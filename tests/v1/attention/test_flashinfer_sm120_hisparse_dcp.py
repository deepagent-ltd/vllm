# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.ops.dcp import _lse_weighted_combine


@pytest.fixture
def backend(monkeypatch):
    impl = object.__new__(sm120.FlashInferMLASparseSM120Impl)
    impl.dcp_world_size = 8
    impl.dcp_rank = 1
    impl.num_heads = 1
    impl.qk_nope_head_dim = 256
    impl.qk_rope_head_dim = 0
    impl.kv_lora_rank = 4
    impl.kv_scale_format = "arbitrary_fp32"
    impl.scale = 0.125
    impl.index_group_index = 0
    impl._workspace_buffer = torch.empty(1, dtype=torch.uint8)
    impl.index_group = Mock(spec=HiSparseMLAIndexGroup)
    kernel = Mock()

    def run(**args):
        lse = args["lse"]
        assert lse.dtype == torch.float32
        assert lse.shape == (args["query"].shape[0], args["query"].shape[2])
        assert args["backend"] == "sparse"
        assert args["return_lse"] and args["return_lse_base"] == "base2"
        assert args["out"].shape == (*args["query"].shape[:-1], impl.kv_lora_rank)
        args["out"].fill_(6)
        lse.fill_(2)
        return args["out"], lse

    kernel.side_effect = run
    monkeypatch.setattr(
        fi_utils, "flashinfer_trtllm_batch_decode_with_kv_cache_mla", kernel
    )
    return impl, kernel


@pytest.mark.parametrize("synthetic", [False, True])
def test_hisparse_dcp_prefill_and_decode_partials_combine(
    backend, monkeypatch, synthetic
):
    impl, kernel = backend
    impl.topk_indices_buffer = torch.tensor(
        [[999, 999, 999, 999], [9, 1, -1, -1], [4, 5, 8, -1], [4, 5, 7, -1]],
        dtype=torch.int32,
    )
    hot = torch.empty((2, 64, 4), dtype=torch.uint8)
    staged = torch.empty((3, 64, 4), dtype=torch.uint8)
    table = torch.tensor([[0 if synthetic else 1]], dtype=torch.int32)
    reqs = torch.zeros(2, dtype=torch.int32)
    metadata = SimpleNamespace(
        topk_start=1,
        num_decode_tokens=1,
        block_size=64,
        hisparse_dcp_decode=object(),
        hisparse_prefill_plan=SimpleNamespace(
            writer_rows=None if synthetic else torch.tensor([-1, 64]),
            query_context_lens=torch.tensor([4, 6]),
        ),
        cp_kv_cache_interleave_size=4,
    )
    impl.index_group.physical_kv_cache.return_value = hot
    impl.index_group.stage_prefill_rows.return_value = staged, table, reqs
    impl.index_group.convert_logical_to_physical_topk.return_value = (
        torch.tensor([[71, -1, -1, -1]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )

    def remap(indices, pages, topk, **kwargs):
        assert indices is reqs and pages is table
        expected = [[-1] * 4, [-1] * 4 if synthetic else [4, 5, -1, -1]]
        assert topk.tolist() == expected
        assert (kwargs["dcp_size"], kwargs["dcp_rank"]) == (8, 1)
        assert kwargs["cp_kv_cache_interleave_size"] == 4
        assert kwargs["BLOCK_STRIDE_ROWS"] == 64 and kwargs["return_valid_counts"]
        return (
            torch.tensor(
                [[-1] * 4, [-1] * 4 if synthetic else [64, 65, -1, -1]],
                dtype=torch.int32,
            ),
            torch.tensor([0, 0 if synthetic else 2], dtype=torch.int32),
        )

    monkeypatch.setattr(sm120, "triton_filter_and_convert_dcp_index", remap)
    # The DCP query has eight gathered heads, whereas each rank owns just one.
    q = torch.zeros((3, 8, 4), dtype=torch.bfloat16)
    out, lse = impl.forward_mqa(q, hot, metadata, None)
    assert impl.can_return_lse_for_decode and not impl.lse_base_on_e
    assert out.shape == (3, 8, 4)
    assert lse.shape == (3, 8)
    assert kernel.call_args_list[0].kwargs["seq_lens"].tolist() == [1]
    assert kernel.call_args_list[1].kwargs["seq_lens"].tolist() == [
        0,
        0 if synthetic else 2,
    ]
    assert impl.index_group.convert_logical_to_physical_topk.call_args.args[
        1
    ].tolist() == [[9, 1, -1, -1]]
    assert out[0].eq(6).all() and lse[0].eq(2).all()
    assert out[1].eq(0).all() and lse[1].isneginf().all()
    if synthetic:
        assert out[2].eq(0).all() and lse[2].isneginf().all()
    else:
        assert out[2].eq(6).all() and lse[2].eq(2).all()

    other_out = torch.full_like(out, 2)
    other_lse = torch.full_like(lse, 1)
    merged, merged_lse = _lse_weighted_combine(
        torch.stack((out, other_out)),
        torch.stack((lse, other_lse)),
        return_lse=True,
        is_lse_base_on_e=impl.lse_base_on_e,
    )
    torch.testing.assert_close(
        merged[0].float(), torch.full_like(merged[0].float(), 14 / 3), atol=0.02, rtol=0
    )
    assert merged[1].eq(2).all() and merged_lse[1].eq(1).all()


def test_sm120_resident_dcp_does_not_use_hisparse_path(backend):
    impl, kernel = backend
    impl.index_group = None
    with pytest.raises(NotImplementedError, match="requires HiSparse"):
        impl.forward_mqa(torch.empty((1, 8, 4)), torch.empty(0), None, None)
    kernel.assert_not_called()
