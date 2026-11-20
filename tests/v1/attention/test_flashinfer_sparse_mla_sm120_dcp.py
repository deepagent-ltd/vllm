# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 GLM NoPE numerical coverage for rank-local HiSparse and DCP reduce."""

import math
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

WORLD_SIZE = 8
INTERLEAVE = 4
BLOCK_SIZE = 256
TOPK = 2048
HEADS = 64
LATENT = 512
ROW_BYTES = 528
SCALE = 256**-0.5
LSE_ATOL = 0.02
OUT_ATOL = 0.015


def _owned(position: int, rank: int) -> bool:
    return position // INTERLEAVE % WORLD_SIZE == rank


def _local(position: int) -> int:
    return position // (WORLD_SIZE * INTERLEAVE) * INTERLEAVE + position % INTERLEAVE


def _effective_query(query: torch.Tensor) -> torch.Tensor:
    # SM120 converts the BF16 query to E4M3 using one power-of-two scale per
    # 128 features. Scoring the unrounded query gives a different LSE.
    tiles = query.float().reshape(*query.shape[:-1], 4, 128)
    scales = (tiles.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2()
    scales = scales.ceil().exp2()
    q8 = (tiles / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q8.float() * scales).reshape_as(query).double()


def _unpack_rows(rows: torch.Tensor) -> torch.Tensor:
    fp8 = rows[:, :LATENT].contiguous().view(torch.float8_e4m3fn)
    scales = rows[:, LATENT:].contiguous().view(torch.float32)
    return (fp8.float().view(-1, 4, 128) * scales.unsqueeze(-1)).flatten(1).double()


def _dense_reference(
    query: torch.Tensor, keys: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = torch.zeros(query.shape, dtype=torch.float64)
    lses = torch.full(query.shape[:-1], -math.inf, dtype=torch.float64)
    for index, rows in enumerate(keys):
        if len(rows):
            scores = (query[index] @ rows.T) * SCALE
            lses[index] = scores.logsumexp(-1) / math.log(2)
            outputs[index] = scores.softmax(-1) @ rows
    return outputs, lses


def _run_case(rank, group, seq_lens, query_lens, num_decodes):
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseMetadata,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
        FlashInferMLASparseSM120Impl,
    )
    from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
    from vllm.v1.attention.ops.dcp import cp_lse_ag_out_rs
    from vllm.v1.hisparse.runtime import (
        HiSparseDCPDecodeMetadata,
        build_hisparse_prefill_staging_plan,
    )

    device = torch.device("cuda", rank)
    num_reqs = len(seq_lens)
    num_prefills = num_reqs - num_decodes
    num_dec_tokens = sum(query_lens[:num_decodes])
    req_ids, positions, starts, candidates = [], [], [0], []
    for request, (seq_len, q_len) in enumerate(zip(seq_lens, query_lens, strict=True)):
        for position in range(seq_len - q_len, seq_len):
            req_ids.append(request)
            positions.append(position)
            # Include future and out-of-range entries deliberately. Reverse
            # order also ensures sparse validity is not a contiguous prefix.
            candidates.append([*range(seq_len, -1, -1), *([-1] * (TOPK - seq_len - 1))])
        starts.append(len(req_ids))
    num_tokens = len(positions)
    req_tensor = torch.tensor(req_ids, dtype=torch.int32, device=device)
    pos_tensor = torch.tensor(positions, dtype=torch.int32, device=device)
    starts_tensor = torch.tensor(starts, dtype=torch.int32, device=device)
    lengths_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    logical = torch.tensor(candidates, dtype=torch.int32, device=device)
    block_table = torch.arange(1, num_reqs + 1, device=device, dtype=torch.int32)
    block_table = block_table[:, None]
    source = torch.zeros(
        (num_reqs + 1, BLOCK_SIZE, ROW_BYTES), device=device, dtype=torch.uint8
    )

    # Make every addressed cache row with the production writer. Arbitrary
    # FP32 scale factors distinguish GLM from the power-of-two DeepSeek path.
    lanes = torch.arange(LATENT, device=device, dtype=torch.float32)
    for req, seq_len in enumerate(seq_lens):
        local = [pos for pos in range(seq_len) if _owned(pos, rank)]
        if not local:
            continue
        indices = torch.tensor(local, device=device, dtype=torch.float32)
        features = (
            0.16 * (indices[:, None] * 0.13 + lanes * 0.027 + req).sin()
            + 0.11 * (indices[:, None] * 0.07 - lanes * 0.011 + rank).cos()
        ).bfloat16()
        slots = torch.arange(len(local), device=device) + (req + 1) * BLOCK_SIZE
        ops.concat_and_cache_mla(
            features,
            features[:, :0].contiguous(),
            source,
            slots,
            "fp8_ds_mla",
            torch.ones(1, device=device),
        )
    source[..., LATENT:].view(torch.float32).mul_(0.93 + 0.017 * rank)
    host = source.cpu().pin_memory()
    writer_slots = torch.tensor(
        [
            (req + 1) * BLOCK_SIZE + _local(pos) if _owned(pos, rank) else -1
            for req, pos in zip(req_ids, positions, strict=True)
        ],
        dtype=torch.int64,
        device=device,
    )
    flat_host = host.view(-1, ROW_BYTES)
    local_keys = [
        _unpack_rows(
            flat_host[
                [
                    (req + 1) * BLOCK_SIZE + _local(i)
                    for i in row
                    if 0 <= i <= pos and _owned(i, rank)
                ]
            ]
        )
        for req, pos, row in zip(req_ids, positions, candidates, strict=True)
    ]
    if num_prefills:
        own_writes = writer_slots[num_dec_tokens:].cpu().tolist()
        # Staging must read active writes from the device cache. The host
        # source deliberately contains different bytes for these exact rows.
        flat_host[[slot for slot in own_writes if slot >= 0]] = 0

    impl = FlashInferMLASparseSM120Impl(
        num_heads=HEADS // WORLD_SIZE,
        head_size=LATENT,
        scale=SCALE,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_ds_mla",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=LATENT,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        qk_head_dim=256,
        v_head_dim=256,
        # Projection happens after rank-local MQA and is outside this test.
        kv_b_proj=None,
        topk_indices_buffer=logical,
    )
    assert impl.dcp_rank == rank and impl.dcp_world_size == WORLD_SIZE
    # The fixture writes GLM's arbitrary FP32 scales explicitly.
    impl.kv_scale_format = "arbitrary_fp32"
    assert impl.get_fp8_ds_mla_row_bytes() == ROW_BYTES
    assert isinstance(impl.index_group, HiSparseMLAIndexGroup)
    cache = impl.index_group.cache(0)
    cache.runtime.request_state_indices = torch.arange(
        num_reqs, dtype=torch.int32, device=device
    )
    cache.runtime.bind_source_cache(host)
    blocks_per_req = 4096 // BLOCK_SIZE
    hot_blocks = num_reqs * blocks_per_req
    cache.runtime.bind_hot_cache(
        torch.zeros(
            hot_blocks * BLOCK_SIZE * ROW_BYTES, dtype=torch.int8, device=device
        ),
        byte_offset=0,
        block_stride=BLOCK_SIZE * ROW_BYTES,
        num_blocks=hot_blocks,
        block_size=BLOCK_SIZE,
        block_table=torch.arange(hot_blocks, dtype=torch.int32, device=device).view(
            num_reqs, blocks_per_req
        ),
    )
    cache.bind_cache(
        source.view(-1),
        byte_offset=0,
        block_stride=BLOCK_SIZE * ROW_BYTES,
        num_blocks=num_reqs + 1,
        block_size=BLOCK_SIZE,
        block_table=torch.zeros_like(block_table),
        slot_mapping=writer_slots,
    )
    cache.source_block_table = block_table
    cache.all_context_pages_resident = False

    decode = None
    if num_dec_tokens:
        decode = HiSparseDCPDecodeMetadata(
            query_start_loc=starts_tensor[: num_decodes + 1],
            seq_lens=lengths_tensor[:num_decodes],
            size=WORLD_SIZE,
            rank=rank,
            interleave=INTERLEAVE,
        )
    plan = None
    if num_prefills:
        plan = build_hisparse_prefill_staging_plan(
            block_table[num_decodes:],
            lengths_tensor[num_decodes:],
            BLOCK_SIZE,
            num_prefills,
            dcp_size=WORLD_SIZE,
            dcp_rank=rank,
            interleave=INTERLEAVE,
        )
        plan.bind_dcp_writer_rows(
            req_tensor[num_dec_tokens:] - num_decodes,
            pos_tensor[num_dec_tokens:],
            writer_slots[num_dec_tokens:],
            dcp_size=WORLD_SIZE,
            dcp_rank=rank,
            interleave=INTERLEAVE,
        )
    meta = FlashInferMLASparseMetadata(
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        query_start_loc=starts_tensor,
        slot_mapping=writer_slots,
        block_table=block_table,
        req_id_per_token=req_tensor,
        seq_lens=lengths_tensor,
        num_decodes=num_decodes,
        num_decode_tokens=num_dec_tokens,
        num_prefills=num_prefills,
        block_size=BLOCK_SIZE,
        topk_tokens=TOPK,
        hisparse_prefill_plan=plan,
        hisparse_dcp_decode=decode,
        cp_kv_cache_interleave_size=INTERLEAVE,
    )
    impl.prepare_for_batch(meta)
    q = (
        (
            0.19
            * (
                torch.arange(num_tokens * HEADS * LATENT, dtype=torch.float32) * 0.039
            ).sin()
            + 0.07
            * (torch.arange(LATENT, dtype=torch.float32) * 0.023)
            .cos()
            .repeat(num_tokens * HEADS)
        )
        .reshape(num_tokens, HEADS, LATENT)
        .bfloat16()
    )
    local_heads = slice(
        rank * (HEADS // WORLD_SIZE), (rank + 1) * (HEADS // WORLD_SIZE)
    )
    all_heads = group.all_gather(q[:, local_heads].contiguous().to(device), dim=1)

    out, lse = impl.forward_mqa(all_heads, host, meta, layer=None)
    assert lse is not None and lse.dtype is torch.float32
    assert out.shape == (num_tokens, HEADS, LATENT) and lse.shape == (num_tokens, HEADS)
    assert not impl.lse_base_on_e
    expected_out, expected_lse = _dense_reference(_effective_query(q), local_keys)
    empty = torch.tensor([len(keys) == 0 for keys in local_keys])
    assert torch.isneginf(lse.cpu()[empty]).all()
    assert torch.count_nonzero(out.cpu()[empty]) == 0
    torch.testing.assert_close(lse.cpu().double(), expected_lse, atol=LSE_ATOL, rtol=0)
    torch.testing.assert_close(out.cpu().double(), expected_out, atol=OUT_ATOL, rtol=0)

    # cp_lse_ag_out_rs scales its input in place. The CPU oracle consumes
    # the independent per-rank reference and cannot double-apply that scale.
    packed_reference = torch.cat((expected_out, expected_lse.unsqueeze(-1)), dim=-1)
    refs = [torch.empty_like(packed_reference) for _ in range(WORLD_SIZE)]
    dist.all_gather(refs, packed_reference, group=group.cpu_group)
    stack = torch.stack(refs)
    weights = torch.softmax(stack[..., -1] * math.log(2), dim=0)
    combined_ref = (weights.unsqueeze(-1) * stack[..., :-1]).sum(0)
    combined_lse_ref = (stack[..., -1] * math.log(2)).logsumexp(0) / math.log(2)
    merged, merged_lse = cp_lse_ag_out_rs(
        out, lse, group, return_lse=True, is_lse_base_on_e=False
    )
    torch.testing.assert_close(
        merged.cpu().double(), combined_ref[:, local_heads], atol=OUT_ATOL, rtol=0
    )
    torch.testing.assert_close(
        merged_lse.cpu().double(),
        combined_lse_ref[:, local_heads],
        atol=LSE_ATOL,
        rtol=0,
    )
    out_error = (merged.cpu().double() - combined_ref[:, local_heads]).abs().amax()
    lse_error = (
        (merged_lse.cpu().double() - combined_lse_ref[:, local_heads]).abs().amax()
    )
    return int(empty.sum()), out_error.item(), lse_error.item()


def _run_rank(rank: int, address: str):
    from vllm.config import (
        HiSparseConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    cfg = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=WORLD_SIZE,
            decode_context_parallel_size=WORLD_SIZE,
            cp_kv_cache_interleave_size=INTERLEAVE,
            dcp_comm_backend="ag_rs",
            disable_custom_all_reduce=True,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=2048,
            is_encoder_decoder=False,
            max_num_seqs=8,
            max_num_batched_tokens=1024,
        ),
    )
    # Kernel qualification stays separate from the serving eligibility gate.
    cfg.attention_config.hisparse_config = HiSparseConfig(device_buffer_size=4096)
    with set_current_vllm_config(cfg):
        init_distributed_environment(
            world_size=WORLD_SIZE,
            rank=rank,
            local_rank=rank,
            distributed_init_method=address,
            backend="nccl",
            timeout=timedelta(seconds=120),
        )
        initialize_model_parallel(
            tensor_model_parallel_size=WORLD_SIZE,
            decode_context_model_parallel_size=WORLD_SIZE,
            backend="nccl",
        )
        try:
            group = get_dcp_group()
            assert group.rank_in_group == rank and len(group.ranks) == WORLD_SIZE
            for case, seq_lens, query_lens, num_decodes in (
                ("decode", (47, 3), (1, 1), 2),
                ("prefill", (34, 3), (4, 3), 0),
                ("mixed", (47, 3, 69), (1, 1, 5), 2),
            ):
                empty, out_error, lse_error = _run_case(
                    rank, group, seq_lens, query_lens, num_decodes
                )
                empty_rows = torch.tensor([empty])
                dist.all_reduce(empty_rows, group=group.cpu_group)
                assert empty_rows.item() > 0
                errors = torch.tensor([out_error, lse_error], dtype=torch.float64)
                dist.all_reduce(errors, op=dist.ReduceOp.MAX, group=group.cpu_group)
                if rank == 0:
                    print(
                        f"{case}: empty rank rows={empty_rows.item()}, "
                        f"merged output max abs={errors[0]:.6g}, "
                        f"merged LSE max abs={errors[1]:.6g}",
                        flush=True,
                    )
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def test_sm120_hisparse_dcp_local_attention_and_lse(tmp_path):
    if torch.version.cuda is None or torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip("requires eight CUDA devices")
    if any(torch.cuda.get_device_capability(i) != (12, 0) for i in range(WORLD_SIZE)):
        pytest.skip("requires eight SM120 devices")

    from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

    assert has_flashinfer_sparse_mla_sm120(), "requires the native FlashInfer SM120 API"
    torch.multiprocessing.spawn(
        _run_rank,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=WORLD_SIZE,
        join=True,
    )
