# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import torch

from vllm import SamplingParams
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.worker.worker_base import CompilationTimes


class CpuFrameWorker:
    """CPU fixture for the external model execution boundary."""

    def __init__(self):
        self.calls = []
        self.current_step: SchedulerOutput | None = None
        self.frame = None

    def get_kv_cache_layer_configs(self):
        return {
            "decoder": KVCacheLayerConfig(
                manager_config=FullAttentionManager.config(),
                spec=AttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=16, dtype=torch.float32
                ),
            )
        }

    def get_supported_kv_cache_layouts(self):
        return ["LBHNC"]

    def set_kv_cache_layout(self, layout, cp_interleave):
        assert layout == "LBHNC"
        assert cp_interleave == 1

    def determine_available_memory(self):
        return 1024 * 1024

    def initialize_from_config(self, kv_cache_configs):
        assert all(config.num_blocks > 0 for config in kv_cache_configs)
        self.cache_initialized = True

    def cache_ready(self):
        return self.cache_initialized

    def compile_or_warm_up_model(self):
        return CompilationTimes(language_model=0, encoder=0)

    def execute_model(self, output):
        self.frame = None
        self.current_step = output
        self.calls.append(("forward", tuple(output.num_scheduled_tokens)))
        return None

    def sample_tokens(self, grammar):
        assert self.current_step is not None
        request_ids = list(self.current_step.num_scheduled_tokens)
        self.calls.append(("sample", tuple(request_ids)))
        self.frame = ModelRunnerOutput(
            req_ids=request_ids,
            sampled_token_ids=[[33] for _ in request_ids],
        )
        return self.frame

    def capture_committed_frame(self, request_ids):
        assert self.frame is not None and self.frame.req_ids == list(request_ids)
        self.calls.append(("capture", tuple(request_ids)))
        if request_ids == ("rpc-timeout",):
            raise TimeoutError("No fixture worker response within the deadline")
        return self.frame.sampled_token_ids

    def synchronize_device(self):
        pass

    def shutdown(self):
        pass


class CpuStepExecutor(Executor):
    @classmethod
    def supports_async_scheduling(cls):
        return True

    supports_pp = True
    supports_timed_collective_rpc = True

    def _init_executor(self):
        self.worker = CpuFrameWorker()

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
    ) -> list[Any]:
        assert isinstance(method, str)
        if method == "capture_committed_frame":
            assert timeout == 1.0 and not non_block
        result = getattr(self.worker, method)(*args, **(kwargs or {}))
        if non_block:
            future: Future[Any] = Future()
            future.set_result(result)
            return [future]
        return [result]

    def check_health(self):
        pass


class RecordingScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_lookups = 0
        self.frames = []
        self.writer_rpc = None
        self.last_outputs: tuple[SchedulerOutput, ModelRunnerOutput] | None = None
        self.requests_committed = set()

    def get_synchronous_step_callback(self):
        self.callback_lookups += 1
        return self.capture_committed_sample

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        self.last_outputs = (scheduler_output, model_runner_output)
        self.requests_committed = set(self.finished_req_ids)
        return result

    def capture_committed_sample(
        self, scheduler_output, model_runner_output, worker_rpc
    ):
        assert self.last_outputs is not None
        assert self.last_outputs[0] is scheduler_output
        assert self.last_outputs[1] is model_runner_output
        request_ids = tuple(model_runner_output.req_ids)
        assert set(request_ids) <= self.requests_committed
        if self.writer_rpc is not None:
            with pytest.raises(RuntimeError, match="outside its synchronous step"):
                self.writer_rpc("capture_committed_frame", timeout=1.0)
        self.writer_rpc = worker_rpc
        [samples] = worker_rpc(
            "capture_committed_frame", timeout=1.0, args=(request_ids,)
        )
        self.frames.append((request_ids, samples))


@pytest.fixture
def make_core(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        """{
            "model_type": "opt", "architectures": ["OPTForCausalLM"],
            "vocab_size": 64, "hidden_size": 16, "word_embed_proj_dim": 16,
            "ffn_dim": 32, "num_attention_heads": 2, "num_hidden_layers": 2,
            "max_position_embeddings": 128, "pad_token_id": 0
        }""",
        encoding="utf-8",
    )
    engines = []

    def create(
        scheduler_cls,
        *,
        async_scheduling=False,
        pipeline_size=1,
        executor_class=CpuStepExecutor,
        core_class=EngineCore,
    ):
        config = VllmConfig(
            model_config=ModelConfig(
                model=str(tmp_path),
                skip_tokenizer_init=True,
                dtype="float32",
                max_model_len=128,
                enforce_eager=True,
            ),
            cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
            parallel_config=ParallelConfig(
                pipeline_parallel_size=pipeline_size,
                distributed_executor_backend=executor_class,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=128,
                max_num_batched_tokens=128,
                max_num_seqs=2,
                is_encoder_decoder=False,
                async_scheduling=async_scheduling,
                scheduler_cls=scheduler_cls,
            ),
        )
        # CPU platform normalization disables async model execution. This
        # native test executor does not run the model and supports the queue.
        config.scheduler_config.async_scheduling = async_scheduling
        engine = core_class(config, executor_class, log_stats=False)
        engines.append(engine)
        return engine

    yield create
    for engine in engines:
        engine.shutdown()


def add_request(core, request_id):
    core.add_request(
        Request(
            request_id=request_id,
            prompt_token_ids=list(range(1, 17)),
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
        )
    )


def test_callback_sees_the_committed_step_before_frame_reuse(make_core):
    core = make_core(RecordingScheduler)
    for request in ("first", "second"):
        add_request(core, request)
        outputs, executed = core.step()
        assert executed and outputs[0].outputs[0].new_token_ids == [33]
    assert core.scheduler.callback_lookups == 1
    assert core.scheduler.frames == [(("first",), [[33]]), (("second",), [[33]])]
    assert core.model_executor.worker.calls == [
        ("forward", ("first",)),
        ("sample", ("first",)),
        ("capture", ("first",)),
        ("forward", ("second",)),
        ("sample", ("second",)),
        ("capture", ("second",)),
    ]
    with pytest.raises(RuntimeError, match="outside its synchronous step"):
        core.scheduler.writer_rpc("capture_committed_frame", timeout=1.0)


def test_unchanged_stock_scheduler_never_enters_callback(make_core):
    core = make_core(Scheduler)
    add_request(core, "stock")
    outputs, executed = core.step()
    assert executed and outputs[0].outputs[0].new_token_ids == [33]
    assert core.model_executor.worker.calls == [
        ("forward", ("stock",)),
        ("sample", ("stock",)),
    ]


@pytest.mark.parametrize("async_scheduling, pipeline_size", [(True, 1), (False, 2)])
def test_opt_in_rejects_concurrent_batches_without_blocking_stock(
    make_core, async_scheduling, pipeline_size
):
    stock = make_core(
        Scheduler, async_scheduling=async_scheduling, pipeline_size=pipeline_size
    )
    assert stock.batch_queue is not None
    with pytest.raises(ValueError, match="exactly one in-flight batch"):
        make_core(
            RecordingScheduler,
            async_scheduling=async_scheduling,
            pipeline_size=pipeline_size,
        )


class FailingScheduler(RecordingScheduler):
    def capture_committed_sample(
        self, scheduler_output, model_runner_output, worker_rpc
    ):
        self.writer_rpc = worker_rpc
        with ThreadPoolExecutor(max_workers=1) as thread:
            call = thread.submit(worker_rpc, "capture_committed_frame", timeout=1.0)
            with pytest.raises(RuntimeError, match="outside its synchronous step"):
                call.result()
        worker_rpc("capture_committed_frame", timeout=float("inf"))


def test_callback_rpc_fails_closed_without_owner_or_finite_deadline(make_core):
    core = make_core(FailingScheduler)
    add_request(core, "rejected")
    with pytest.raises(ValueError, match="finite timeout"):
        core.step()
    assert core.model_executor.worker.calls == [
        ("forward", ("rejected",)),
        ("sample", ("rejected",)),
    ]
    with pytest.raises(RuntimeError, match="outside its synchronous step"):
        core.scheduler.writer_rpc("capture_committed_frame", timeout=1.0)


class UntimedStepExecutor(CpuStepExecutor):
    supports_timed_collective_rpc = False


def test_opt_in_rejects_executor_that_cannot_enforce_the_deadline(make_core):
    core = make_core(Scheduler, executor_class=UntimedStepExecutor)
    add_request(core, "stock")
    assert core.step()[1]
    with pytest.raises(ValueError, match="with timed worker RPC"):
        make_core(RecordingScheduler, executor_class=UntimedStepExecutor)


def test_failed_native_rpc_cannot_emit_the_current_step(make_core):
    core = make_core(RecordingScheduler)
    add_request(core, "rpc-timeout")
    with pytest.raises(TimeoutError, match="within the deadline"):
        core.step()
    assert core.scheduler.frames == []
    assert core.model_executor.worker.calls == [
        ("forward", ("rpc-timeout",)),
        ("sample", ("rpc-timeout",)),
        ("capture", ("rpc-timeout",)),
    ]
    with pytest.raises(RuntimeError, match="outside its synchronous step"):
        core.scheduler.writer_rpc("capture_committed_frame", timeout=1.0)


class StartupScheduler(Scheduler):
    def get_worker_startup_callback(self):
        return self.record_startup

    def record_startup(self, worker_rpc):
        self.startup_rpc = worker_rpc
        self.cache_ready = worker_rpc("cache_ready", timeout=1.0)

    def get_owner_worker_control_registration(self):
        return ("cache_ready",), self.bind_owner_rpc

    def bind_owner_rpc(self, worker_rpc):
        self.owner_rpc = worker_rpc


def test_startup_observes_native_cache_and_ends_its_rpc_scope(make_core):
    core = make_core(StartupScheduler)
    scheduler = core.scheduler
    assert scheduler.cache_ready == [True]
    assert scheduler.owner_rpc("cache_ready", timeout=1.0) == [True]
    with pytest.raises(RuntimeError, match="outside its synchronous step"):
        scheduler.startup_rpc("cache_ready", timeout=1.0)
    with pytest.raises(ValueError, match="not registered"):
        scheduler.owner_rpc("get_kv_cache_layer_configs", timeout=1.0)
    with ThreadPoolExecutor(max_workers=1) as thread:
        call = thread.submit(scheduler.owner_rpc, "cache_ready", timeout=1.0)
        with pytest.raises(RuntimeError, match="EngineCore owner"):
            call.result()
