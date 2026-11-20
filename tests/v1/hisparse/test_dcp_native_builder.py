# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.hisparse.runtime import (
    HiSparseCacheHandle,
    HiSparseRuntime,
    PagedCacheView,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.utils import AttentionGroup


def _builder(block_size, *, hisparse=True, dcp_size=8):
    builder = object.__new__(FlashInferMLASparseMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=object() if hisparse else None)
    )
    builder.kv_cache_spec = SimpleNamespace(block_size=block_size, dcp_sharded=True)
    builder.use_pcp = False
    builder.use_shardq = False
    builder.dcp_world_size = dcp_size
    builder.dcp_rank = 1 if dcp_size > 1 else 0
    builder.cp_kv_cache_interleave_size = 4
    builder.topk_tokens = 2048
    builder.device = torch.device("cpu")
    builder._prefill_backend = None
    builder.reorder_batch_threshold = 4
    builder.req_id_per_token_buffer = torch.empty(16, dtype=torch.int32)
    return builder


@pytest.mark.parametrize("for_capture", [False, True])
def test_native_builder_gets_explicit_dummy_from_worker(monkeypatch, for_capture):
    from vllm.v1.attention.ops import metadata as metadata_ops

    def cpu_token_to_req_indices(starts, buffer, num_mapped, num_output):
        assert num_mapped == num_output
        req_ids = torch.repeat_interleave(
            torch.arange(starts.numel() - 1, dtype=buffer.dtype), starts.diff()
        )
        return buffer[:num_output].copy_(req_ids)

    monkeypatch.setattr(
        metadata_ops, "compute_token_to_req_indices", cpu_token_to_req_indices
    )
    batch = InputBatch.make_dummy(1, 4, InputBuffers(2, 8, torch.device("cpu")))
    # Exercise a one-chunk synthetic prefill rather than a synthetic decode.
    builder = _builder(64)
    builder.reorder_batch_threshold = 1
    state = object.__new__(DefaultModelState)
    state.model_config = SimpleNamespace(max_model_len=4096)
    state.mm_prefix_ranges = None
    group = AttentionGroup(None, ["sparse"], builder.kv_cache_spec, 0)
    group.metadata_builders = [builder]
    metadata = state.prepare_attn(
        batch,
        CUDAGraphMode.PIECEWISE,
        (torch.zeros((1, 1), dtype=torch.int32),),
        torch.full((1, 4), -1, dtype=torch.int64),
        [[group]],
        SimpleNamespace(kv_cache_groups=[object()]),
        for_capture=for_capture,
    )["sparse"]
    plan = metadata.hisparse_prefill_plan
    assert plan is not None and plan.writer_rows is None
    assert plan.query_context_lens.tolist() == [1, 2, 3, 4]
    assert metadata.hisparse_dcp_decode is None
    batch.is_dummy = False
    with pytest.raises(RuntimeError, match="do not cover the current query suffix"):
        state.prepare_attn(
            batch,
            CUDAGraphMode.PIECEWISE,
            (torch.zeros((1, 1), dtype=torch.int32),),
            torch.full((1, 4), -1, dtype=torch.int64),
            [[group]],
            SimpleNamespace(kv_cache_groups=[object()]),
            for_capture=for_capture,
        )


def _native_mixed_batch(block_size):
    span = block_size * 8
    starts = torch.tensor([0, 1, 5], dtype=torch.int32)
    seq_lens = torch.tensor([8, span + 7], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=seq_lens,
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=4,
        max_seq_len=span + 7,
        block_table_tensor=torch.tensor([[2, 0], [3, 5]], dtype=torch.int32),
        slot_mapping=torch.tensor(
            [
                2 * block_size + 3,
                -1,
                5 * block_size,
                5 * block_size + 1,
                5 * block_size + 2,
            ]
        ),
        positions=torch.tensor([7, span + 3, span + 4, span + 5, span + 6]),
        is_prefilling=torch.tensor([False, True]),
        seq_lens_cpu_upper_bound=seq_lens,
        dcp_local_seq_lens_cpu_upper_bound=torch.tensor([4, block_size + 3]),
        _token_to_req_indices_cache=torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
    )


@pytest.mark.parametrize("block_size", [64, 256])
def test_native_builder_and_cache_handle_use_current_resident_writer_slice(block_size):
    common = _native_mixed_batch(block_size)
    metadata = _builder(block_size).build(0, common)
    assert (
        metadata.num_decodes,
        metadata.num_prefills,
        metadata.num_decode_tokens,
    ) == (1, 1, 1)
    assert metadata.hisparse_dcp_decode.seq_lens.tolist() == [8]
    plan = metadata.hisparse_prefill_plan
    assert plan is not None and int(plan.writer_mask.sum()) == 3
    assert plan.query_context_lens.tolist() == (common.positions[1:] + 1).tolist()

    handle = object.__new__(HiSparseCacheHandle)
    handle.block_table = torch.tensor([[0, 9], [8, 0]], dtype=torch.int32)
    handle.slot_mapping = torch.tensor(
        [8 * block_size, -1, 9 * block_size, 9 * block_size + 1, 9 * block_size + 2]
    )
    raw = torch.zeros(10 * block_size, dtype=torch.uint8)
    handle.view = PagedCacheView.bind(
        raw,
        dtype=raw.dtype,
        row_width=1,
        byte_offset=0,
        block_stride=block_size,
        num_blocks=10,
        block_size=block_size,
    )

    def gather(source_cache, staging_plan, *, resident_cache):
        assert source_cache is raw
        assert staging_plan is plan
        assert resident_cache is handle.view.cache
        return resident_cache

    handle.runtime = SimpleNamespace(
        request_state_indices=torch.tensor([1, 0]), gather_prefill_cache=gather
    )
    group = object.__new__(HiSparseMLAIndexGroup)
    group.caches = [handle]
    cache, table, req_ids = group.stage_prefill_rows(0, raw, metadata)
    assert cache is handle.view.cache
    assert table is plan.block_table
    assert req_ids.tolist() == [0, 0, 0, 0]
    assert (
        plan.gpu_row_ids[plan.writer_mask].tolist() == handle.slot_mapping[2:].tolist()
    )
    assert torch.count_nonzero(plan.miss_mask[plan.writer_mask]) == 0
    assert int(plan.miss_mask.sum()) == block_size


@pytest.mark.parametrize("hisparse,dcp_size", [(False, 8), (True, 1)])
def test_shared_builder_retains_non_hisparse_dcp_split(hisparse, dcp_size):
    builder = _builder(64, hisparse=hisparse, dcp_size=dcp_size)
    common = _native_mixed_batch(64)
    metadata = builder.build(0, common)
    assert (
        metadata.num_decodes,
        metadata.num_prefills,
        metadata.num_decode_tokens,
    ) == (2, 0, 5)
    assert metadata.hisparse_prefill_plan is None
    assert metadata.hisparse_dcp_decode is None


def test_native_builder_localizes_only_decode_before_the_resolver():
    block_size = 64
    metadata = _builder(block_size).build(0, _native_mixed_batch(block_size))
    logical_topk = torch.tensor([[4, 0, 7, 2, -1]], dtype=torch.int32)
    rows = torch.empty_like(logical_topk)
    shared = SimpleNamespace(
        device_topk_rows=rows.clone(),
        valid_topk_counts=torch.empty(1, dtype=torch.int32),
        physical_topk_indices=rows.clone(),
        swap_host_physical_rows=rows.clone(),
        swap_device_physical_rows=rows.clone(),
        swap_counts=torch.empty(1, dtype=torch.int32),
    )
    physical = torch.tensor([[31, -1, -1, 8, -1]], dtype=torch.int32)

    def resolve_residency(*args):
        assert args[3].tolist() == [[0, -1, 3, -1, -1]]
        assert args[-1].tolist() == [1]
        shared.physical_topk_indices.copy_(physical)

    runtime = SimpleNamespace(
        hot=SimpleNamespace(
            block_size=block_size, cache=None, attention_block_stride=64
        ),
        hot_block_table=None,
        host_cache=None,
        request_state_indices=torch.tensor([0, 1]),
        index_group=SimpleNamespace(
            shared_topk=shared, device_global_indices=None, lru_slots=None
        ),
        region_stride=block_size,
        max_union_rows=5,
        cache_ops=SimpleNamespace(hisparse_resolve_residency=resolve_residency),
    )
    runtime.index_group.swap_stats = None
    handle = HiSparseCacheHandle(runtime)
    handle.source_block_table = metadata.block_table

    def call_resolver(**kwargs):
        kwargs["topk_indices"] = kwargs.pop("logical_topk_indices")
        return_valid_counts = kwargs["return_valid_counts"]
        HiSparseRuntime._resolve_residency(runtime, shared_rows=slice(0, 1), **kwargs)
        if return_valid_counts:
            return shared.physical_topk_indices, shared.valid_topk_counts
        return shared.physical_topk_indices

    runtime.swap_in = call_resolver
    group = object.__new__(HiSparseMLAIndexGroup)
    group.caches = [handle]
    rows, counts = group.convert_logical_to_physical_topk(
        0, logical_topk, metadata, block_stride_rows=None, return_valid_counts=True
    )
    assert rows.tolist() == [[31, 8, -1, -1, -1]]
    assert counts.tolist() == [2]
    with pytest.raises(ValueError, match="cannot consume prefill"):
        group.convert_logical_to_physical_topk(
            0,
            logical_topk.expand(2, -1),
            metadata,
            block_stride_rows=None,
            return_valid_counts=True,
        )
