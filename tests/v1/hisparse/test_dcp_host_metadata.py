# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    _localize_dcp_row_mirrors,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.hisparse.runtime import (
    HiSparseDCPDecodeMetadata,
    build_hisparse_prefill_staging_plan,
)
from vllm.v1.hisparse.types import SparseKVRowMirror


@pytest.mark.parametrize("block_size", [64, 256])
def test_rank_local_host_plan_never_recalls_the_current_writer(block_size):
    span = block_size * 8
    positions = torch.tensor([span + 3, span + 4, span + 5, span + 6])
    source_slots = torch.tensor(
        [-1, 5 * block_size, 5 * block_size + 1, 5 * block_size + 2]
    )
    plan = build_hisparse_prefill_staging_plan(
        torch.tensor([[3, 5]], dtype=torch.int32),
        torch.tensor([span + 7]),
        block_size,
        2,
        dcp_size=8,
        dcp_rank=1,
        interleave=4,
    )
    plan.bind_dcp_writer_rows(
        torch.zeros(4, dtype=torch.int64),
        positions,
        source_slots,
        dcp_size=8,
        dcp_rank=1,
        interleave=4,
    )
    assert plan.writer_rows is not None
    assert plan.writer_rows[0] == -1
    assert torch.count_nonzero(plan.writer_mask) == 3
    assert plan.query_context_lens.tolist() == (positions + 1).tolist()

    resident_slots = torch.tensor(
        [-1, 9 * block_size, 9 * block_size + 1, 9 * block_size + 2]
    )
    plan.ensure_gpu_sources(
        torch.tensor([[0, 9]], dtype=torch.int32),
        torch.tensor([0]),
        block_size,
        writer_slots=resident_slots,
        resident_num_blocks=10,
    )
    assert plan.gpu_row_ids[plan.writer_mask].tolist() == resident_slots[1:].tolist()
    assert torch.count_nonzero(plan.miss_mask[plan.writer_mask]) == 0
    assert int(plan.miss_mask.sum()) == block_size


def test_rank_local_host_plan_rejects_an_incorrect_source_slot():
    plan = build_hisparse_prefill_staging_plan(
        torch.tensor([[4]], dtype=torch.int32),
        torch.tensor([8]),
        64,
        1,
        dcp_size=8,
        dcp_rank=1,
        interleave=4,
    )
    with pytest.raises(RuntimeError, match="native source slot"):
        plan.bind_dcp_writer_rows(
            torch.tensor([0]),
            torch.tensor([4]),
            torch.tensor([4 * 64 + 1]),
            dcp_size=8,
            dcp_rank=1,
            interleave=4,
        )


@pytest.mark.parametrize("rank", range(8))
def test_decode_indices_use_causal_owned_rows_and_ignore_graph_padding(rank):
    starts = torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([40, 6, 8], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=4,
        max_query_len=2,
        max_seq_len=40,
        block_table_tensor=torch.ones((3, 1), dtype=torch.int32),
        slot_mapping=torch.full((4,), -1, dtype=torch.int64),
    )
    metadata = HiSparseDCPDecodeMetadata.from_common(
        common, 2, size=8, rank=rank, interleave=4
    )
    query_requests = torch.tensor([0, 0, 1, 0], dtype=torch.int32)
    topk = torch.tensor(
        [
            [4, 38, 39, 35, -1],
            [7, 39, 37, 8, -1],
            [5, 6, 7, 1, -1],
            [0, 7, 39, 1, -1],
        ],
        dtype=torch.int32,
    )
    actual = metadata.localize(query_requests, topk)
    bounds = [38, 39, 5, -1]
    expected = [
        [
            (token // 32) * 4 + token % 4
            if 0 <= token <= bound and token // 4 % 8 == rank
            else -1
            for token in row
        ]
        for row, bound in zip(topk.tolist(), bounds, strict=True)
    ]
    assert actual.tolist() == expected


@pytest.mark.parametrize("rank", range(8))
def test_connector_mirror_uses_only_its_rank_local_physical_span(rank):
    block_size = 64
    span = block_size * 8
    start = 30
    num_rows = 10
    mirrors = (
        SparseKVRowMirror(
            source_starts=(3 * span + start, 5 * span + start),
            destination_start=7 * span + start,
            num_rows=num_rows,
        ),
    )
    localized = _localize_dcp_row_mirrors(
        mirrors, block_size=block_size, dcp_size=8, dcp_rank=rank, interleave=4
    )
    owned = [
        (position // 32) * 4 + position % 4
        for position in range(start, start + num_rows)
        if position // 4 % 8 == rank
    ]
    if not owned:
        assert localized == ()
        return
    assert localized == (
        SparseKVRowMirror(
            source_starts=(3 * block_size + owned[0], 5 * block_size + owned[0]),
            destination_start=7 * block_size + owned[0],
            num_rows=len(owned),
        ),
    )
