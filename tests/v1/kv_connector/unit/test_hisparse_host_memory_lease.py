# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.hisparse import runtime


def _config(*, nested=False, tp=1):
    spec = {
        "kv_connector": "HiSparseConnector",
        "kv_connector_extra_config": {
            "host_pool_gib": 1,
            "host_pool_allocator": "my_plugin.acquire",
        },
    }
    if nested:
        spec = {
            "kv_connector": "MultiConnector",
            "kv_connector_extra_config": {"connectors": [spec]},
        }
    return SimpleNamespace(
        instance_id="dp-owner",
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_index=3,
        ),
        kv_transfer_config=SimpleNamespace(**spec),
    )


def _make_pool(monkeypatch, *, nested=False, tp=1, size_adjustment=0):
    config = _config(nested=nested, tp=tp)
    pool = runtime.HiSparseHostPool(
        config, SimpleNamespace(hisparse_shared_host_pool=False)
    )
    size = mmap.PAGESIZE
    base = torch.empty(2 * size, dtype=torch.int8)
    offset = (-base.data_ptr()) % size
    backing = base[offset : offset + size + size_adjustment]
    release = Mock()
    allocator = Mock(return_value=runtime.HiSparseHostMemoryLease(backing, release))
    pin = Mock()
    monkeypatch.setattr(runtime, "_host_pools", {config.instance_id: pool})
    monkeypatch.setattr(
        runtime, "resolve_obj_by_qualname", Mock(return_value=allocator)
    )
    monkeypatch.setattr(runtime, "pin_tensor", pin)
    return config, pool, backing, allocator, pin, release


@pytest.mark.parametrize("nested", [False, True])
def test_external_backing_stays_leased_until_unregistered(monkeypatch, nested):
    config, pool, backing, allocator, pin, release = _make_pool(
        monkeypatch, nested=nested
    )
    view = pool.allocate(17)
    assert view.shape == (17,)
    assert view.data_ptr() == backing.data_ptr()
    assert pool.allocate(17) is view
    allocator.assert_called_once_with(
        instance_id="dp-owner", dp_rank=3, size_bytes=mmap.PAGESIZE
    )
    pin.assert_called_once_with(backing)

    runtime.drop_hisparse_host_pool(config)
    release.assert_not_called()
    assert runtime._host_pools[config.instance_id] is pool

    # Emulate a successful release_pinned_state device sync and unregister.
    pool.pinned.clear()
    runtime.drop_hisparse_host_pool(config)
    runtime.drop_hisparse_host_pool(config)
    release.assert_called_once()
    assert config.instance_id not in runtime._host_pools


def test_invalid_backing_is_released_without_registration(monkeypatch):
    _, pool, _, _, pin, release = _make_pool(monkeypatch, size_adjustment=-1)
    with pytest.raises(ValueError, match="page-aligned CPU int8"):
        pool.allocate(17)
    pin.assert_not_called()
    release.assert_called_once()
    assert pool.backing is None


def test_registration_failure_releases_lease(monkeypatch):
    _, pool, _, _, pin, release = _make_pool(monkeypatch)
    pin.side_effect = RuntimeError("registration failed")
    with pytest.raises(RuntimeError, match="registration failed"):
        pool.allocate(17)
    release.assert_called_once()
    assert pool.pinned == []
    assert pool.backing is None


def test_external_backing_rejects_multi_worker_group(monkeypatch):
    _, pool, _, allocator, pin, release = _make_pool(monkeypatch, tp=2)
    with pytest.raises(ValueError, match="TP1/PP1/PCP1/DCP1"):
        pool.allocate(17)
    allocator.assert_not_called()
    pin.assert_not_called()
    release.assert_not_called()


def test_failed_external_release_retains_lease_for_retry(monkeypatch):
    config, pool, _, _, _, release = _make_pool(monkeypatch)
    pool.allocate(17)
    pool.pinned.clear()
    release.side_effect = [RuntimeError("release failed"), None]

    with pytest.raises(RuntimeError, match="release failed"):
        runtime.drop_hisparse_host_pool(config)
    assert runtime._host_pools[config.instance_id] is pool
    with pytest.raises(RuntimeError, match="retiring"):
        pool.allocate(17)

    runtime.drop_hisparse_host_pool(config)
    assert release.call_count == 2
    assert config.instance_id not in runtime._host_pools


def test_failed_registration_and_cleanup_retains_lease_for_retry(monkeypatch):
    config, pool, _, _, pin, release = _make_pool(monkeypatch)
    pin.side_effect = RuntimeError("registration failed")
    release.side_effect = [RuntimeError("release failed"), None]

    with pytest.raises(RuntimeError, match="registration failed"):
        pool.allocate(17)
    assert not pool.pinned
    with pytest.raises(RuntimeError, match="retiring"):
        pool.allocate(17)

    runtime.drop_hisparse_host_pool(config)
    assert release.call_count == 2
    assert config.instance_id not in runtime._host_pools
