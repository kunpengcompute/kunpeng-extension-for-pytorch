# Copyright and license: see pytorch_cpu_upstream.LICENSE (PyTorch contributors).
"""Unmodified selected CPU collective test methods from pytorch/pytorch.

Only the imports/class harness and selected matrix are adapted by our runner.
See pytorch_cpu_upstream.json for the immutable source and method locations.
"""

import torch
import torch.distributed as dist

COUNTS = (0, 4)
ASYNC_OPS = (False, True)


class UpstreamCpuCollectives:
    def _value(self, rank, dtype):
        if dtype == torch.bool:
            return bool(rank % 2)
        return rank + 1

    def _tensor(self, count, dtype, rank=None):
        rank = self.rank if rank is None else rank
        value = self._value(rank, dtype)
        return torch.full((count,), value, dtype=torch.float32, device=self.device).to(
            dtype
        )

    def _expected_tensor(self, count, dtype, rank):
        return self._tensor(count, dtype, rank=rank)

    def _wait(self, work, async_op):
        if async_op:
            self.assertIsNotNone(work)
            work.wait()
        else:
            self.assertIsNone(work)

    def _expected_reduce_value(self, dtype, op):
        values = [self._value(rank, dtype) for rank in range(self.world_size)]
        if op == dist.ReduceOp.SUM:
            result = sum(values)
        elif op == dist.ReduceOp.PRODUCT:
            result = 1
            for value in values:
                result *= value
        elif op == dist.ReduceOp.MIN:
            result = min(values)
        elif op == dist.ReduceOp.MAX:
            result = max(values)
        elif op == dist.ReduceOp.AVG:
            result = sum(values) / self.world_size
        elif op == dist.ReduceOp.BAND:
            result = values[0]
            for value in values[1:]:
                result &= value
        elif op == dist.ReduceOp.BOR:
            result = values[0]
            for value in values[1:]:
                result |= value
        elif op == dist.ReduceOp.BXOR:
            result = values[0]
            for value in values[1:]:
                result ^= value
        else:
            raise AssertionError(f"Unhandled reduce op: {op}")
        return result

    def _expected_reduce(self, count, dtype, op):
        value = self._expected_reduce_value(dtype, op)
        return torch.full((count,), value, dtype=torch.float32, device=self.device).to(
            dtype
        )

    def _test_broadcast(self, count, dtype, async_op):
        tensor = self._tensor(count, dtype)
        work = dist.broadcast(tensor, src=0, async_op=async_op)
        self._wait(work, async_op)
        self.assertEqual(tensor, self._expected_tensor(count, dtype, rank=0))

    def _test_all_gather(self, count, dtype, async_op):
        tensor = self._tensor(count, dtype)
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        work = dist.all_gather(output, tensor, async_op=async_op)
        self._wait(work, async_op)
        for rank, result in enumerate(output):
            self.assertEqual(result, self._expected_tensor(count, dtype, rank))

    def _test_all_gather_single(self, count, dtype, async_op):
        tensor = self._tensor(count, dtype)
        output = torch.empty(self.world_size * count, dtype=dtype, device=self.device)
        work = dist.all_gather_single(output, tensor, async_op=async_op)
        self._wait(work, async_op)
        expected = torch.cat(
            [
                self._expected_tensor(count, dtype, rank)
                for rank in range(self.world_size)
            ]
        )
        self.assertEqual(output, expected)

    def _test_all_to_all_single(self, count, dtype, async_op):
        inputs = torch.cat(
            [
                self._expected_tensor(count, dtype, self.rank)
                for _ in range(self.world_size)
            ]
        )
        output = torch.empty_like(inputs)
        work = dist.all_to_all_single(output, inputs, async_op=async_op)
        self._wait(work, async_op)
        expected = torch.cat(
            [
                self._expected_tensor(count, dtype, rank)
                for rank in range(self.world_size)
            ]
        )
        self.assertEqual(output, expected)

    def _test_transport_matrix(self, test):
        for count in COUNTS:
            for dtype in self.dtypes:
                for async_op in ASYNC_OPS:
                    with self.subTest(count=count, dtype=dtype, async_op=async_op):
                        test(count, dtype, async_op)

    def test_broadcast(self):
        self._test_transport_matrix(self._test_broadcast)

    def test_all_gather(self):
        self._test_transport_matrix(self._test_all_gather)

    def test_all_gather_single(self):
        self._test_transport_matrix(self._test_all_gather_single)

    def test_all_to_all_single(self):
        self._test_transport_matrix(self._test_all_to_all_single)

    def test_all_reduce(self):
        for count in COUNTS:
            for dtype in self.dtypes:
                for op in self._reduce_ops(dtype):
                    for async_op in ASYNC_OPS:
                        with self.subTest(
                            count=count, dtype=dtype, op=op, async_op=async_op
                        ):
                            tensor = self._tensor(count, dtype)
                            work = dist.all_reduce(tensor, op=op, async_op=async_op)
                            self._wait(work, async_op)
                            self.assertEqual(
                                tensor, self._expected_reduce(count, dtype, op)
                            )
