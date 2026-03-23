import numpy as np
import torch

from iron.common.aie_base import AIEOperatorBase


class _DummyOperator(AIEOperatorBase):
    def set_up_artifacts(self):
        return None

    def set_up_runtime(self):
        return None


def test_read_buffer_copy_uses_mapped_copy_path():
    operator = object.__new__(_DummyOperator)

    class FakeBO:
        def __init__(self):
            self.read_calls = []
            self.map_calls = 0
            self.payload = np.arange(8, dtype=np.uint8)

        def read(self, size, offset):
            self.read_calls.append((size, offset))
            return np.arange(size, dtype=np.int8)

        def map(self):
            self.map_calls += 1
            return memoryview(self.payload)

    fake_bo = FakeBO()
    operator.buffer_bos = {"output": fake_bo}

    result = _DummyOperator.read_buffer(
        operator,
        "output",
        (2, 4),
        copy=True,
        dtype=np.uint8,
    )

    assert fake_bo.read_calls == []
    assert fake_bo.map_calls == 1
    assert np.array_equal(result, np.arange(8, dtype=np.uint8).reshape(2, 4))


def test_read_buffer_as_torch_copies_bo_backed_arrays():
    operator = object.__new__(_DummyOperator)
    captured = {}

    def fake_read_buffer(buffer_name, shape, copy=False, dtype=None):
        captured["buffer_name"] = buffer_name
        captured["shape"] = shape
        captured["copy"] = copy
        captured["dtype"] = dtype
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    operator.read_buffer = fake_read_buffer

    result = _DummyOperator.read_buffer_as_torch(
        operator,
        "output",
        (2, 3),
        dtype=np.float32,
    )

    assert captured == {
        "buffer_name": "output",
        "shape": (2, 3),
        "copy": True,
        "dtype": np.float32,
    }
    assert torch.equal(result, torch.arange(6, dtype=torch.float32).reshape(2, 3))
