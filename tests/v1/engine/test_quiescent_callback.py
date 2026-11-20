# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from concurrent.futures import Future
from contextlib import contextmanager
from threading import Event, Thread, get_ident

import pytest

from vllm import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import EngineZmqAddresses
from vllm.v1.request import Request

from .test_synchronous_step_callback import (
    CpuFrameWorker,
    CpuStepExecutor,
    RecordingScheduler,
    StartupScheduler,
    add_request,
)
from .test_synchronous_step_callback import (
    make_core as shared_core_fixture,
)

make_core = shared_core_fixture

INTERVAL_NS = 20_000_000
INTERVAL = INTERVAL_NS / 1_000_000_000


class ObservedInputQueue(queue.Queue):
    """Record the actual blocking queue calls and the input/timeout race."""

    def __init__(self):
        super().__init__()
        self.waits = queue.SimpleQueue()
        self.on_timeout = None

    def get(self, block=True, timeout=None):
        self.waits.put((block, timeout))
        try:
            return super().get(block=block, timeout=timeout)
        except queue.Empty:
            if self.on_timeout is not None:
                inject, self.on_timeout = self.on_timeout, None
                inject()
            raise


class QueueCoreProc(EngineCoreProc):
    """Real engine and scheduler with the sockets supplied by the test queue."""

    def __init__(self, config, executor, log_stats):
        super().__init__(config, True, "", executor, log_stats)
        self.input_queue = ObservedInputQueue()

    @contextmanager
    def _perform_handshakes(self, *args):
        yield EngineZmqAddresses(inputs=[], outputs=[])

    def process_input_sockets(self, _, __, ___, ready_event):
        ready_event.set()

    def process_output_sockets(self, *args):
        pass


class IdleScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callbacks = queue.SimpleQueue()

    def get_quiescent_callback(self):
        return INTERVAL_NS, self.poll_quiescent_work

    def poll_quiescent_work(self):
        assert not self.has_requests()
        self.callbacks.put(get_ident())


class WaitingCpuWorker(CpuFrameWorker):
    def __init__(self):
        super().__init__()
        self.inflight = Event()
        self.finish_forward = Event()

    def execute_model(self, output):
        self.inflight.set()
        assert self.finish_forward.wait(3)
        return super().execute_model(output)


class WaitingCpuExecutor(CpuStepExecutor):
    def _init_executor(self):
        self.worker = WaitingCpuWorker()


@contextmanager
def busy_loop(engine):
    result = Future()

    def run():
        try:
            engine.run_busy_loop()
        except SystemExit:
            result.set_result(None)
        except BaseException as exc:
            result.set_exception(exc)

    thread = Thread(target=run, daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        engine.shutdown_state = EngineShutdownState.REQUESTED
        engine.input_queue.put((EngineCoreRequestType.WAKEUP, None))
        thread.join(5)
        assert not thread.is_alive()
        result.result(timeout=0)


def test_default_idle_blocks_without_callback(make_core):
    engine = make_core(Scheduler, core_class=QueueCoreProc)
    with busy_loop(engine):
        assert engine.input_queue.waits.get(timeout=3) == (True, None)
        with pytest.raises(queue.Empty):
            engine.input_queue.waits.get(timeout=3 * INTERVAL)
        assert engine.model_executor.worker.calls == []


def test_quiescent_callback_wakes_without_a_client_on_the_engine_thread(make_core):
    engine = make_core(IdleScheduler, core_class=QueueCoreProc)
    with busy_loop(engine) as owner:
        assert engine.input_queue.waits.get(timeout=3) == (True, INTERVAL)
        assert engine.scheduler.callbacks.get(timeout=3) == owner.ident
        assert engine.scheduler.callbacks.get(timeout=3) == owner.ident
        assert engine.model_executor.worker.calls == []


def test_input_appearing_at_timeout_has_priority_over_callback(make_core):
    engine = make_core(
        IdleScheduler,
        core_class=QueueCoreProc,
        executor_class=WaitingCpuExecutor,
    )
    request = Request(
        request_id="arriving-at-timeout",
        prompt_token_ids=list(range(1, 17)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    engine.input_queue.on_timeout = lambda: engine.input_queue.put(
        (EngineCoreRequestType.ADD, (request, 0))
    )
    worker = engine.model_executor.worker
    with busy_loop(engine) as owner:
        try:
            assert worker.inflight.wait(3)
            assert engine.scheduler.has_requests()
            with pytest.raises(queue.Empty):
                engine.scheduler.callbacks.get(timeout=3 * INTERVAL)
        finally:
            worker.finish_forward.set()
        assert engine.scheduler.callbacks.get(timeout=3) == owner.ident
        assert worker.calls == [
            ("forward", (request.request_id,)),
            ("sample", (request.request_id,)),
            ("forward", ()),
            ("sample", ()),
        ]


def test_paused_engine_keeps_untimed_wait_even_with_opt_in(make_core):
    engine = make_core(IdleScheduler, core_class=QueueCoreProc)
    engine.pause_scheduler(mode="keep", clear_cache=False)
    with busy_loop(engine):
        assert engine.input_queue.waits.get(timeout=3) == (True, None)
        with pytest.raises(queue.Empty):
            engine.scheduler.callbacks.get(timeout=3 * INTERVAL)


class OnePollFuture(Future):
    def __init__(self, value):
        super().__init__()
        self._polled = False
        self.set_result(value)

    def poll(self):
        prior, self._polled = self._polled, True
        return prior

    def wait_briefly(self, pending_inputs, deadline):
        raise AssertionError("The fixture always presents an input on the first poll")


class OverlapExecutor(CpuStepExecutor):
    def execute_model(self, scheduler_output, non_block=False):
        result = super().execute_model(scheduler_output, non_block)
        return OnePollFuture(result.result()) if non_block else result


@pytest.mark.parametrize(
    ("scheduler_cls", "deferred"),
    [
        (Scheduler, False),
        (RecordingScheduler, True),
        (StartupScheduler, True),
    ],
)
def test_step_input_order_respects_owner_callback_boundary(
    make_core, scheduler_cls, deferred
):
    core = make_core(
        scheduler_cls, core_class=QueueCoreProc, executor_class=OverlapExecutor
    )
    add_request(core, "active")
    next_request = Request(
        request_id="pending-next",
        prompt_token_ids=list(range(1, 17)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    core.input_queue.put((EngineCoreRequestType.ADD, (next_request, 0)))
    output, executed = core.step()
    assert executed and output[0].outputs[0].request_id == "active"
    assert core.scheduler.has_request(next_request.request_id) is not deferred
    if deferred:
        assert len(core._deferred_inputs) == 1
    core._process_input_queue()
    assert core.scheduler.has_request(next_request.request_id)
    assert not core._deferred_inputs
