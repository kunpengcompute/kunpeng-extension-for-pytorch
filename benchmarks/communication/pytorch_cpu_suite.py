"""CPU collective cases: frozen PyTorch methods plus KuCCL-specific coverage.

Planning is deliberately torch-free so --list and the host regressions work
without a compiled extension. Runtime imports happen only in rank workers.
"""

from dataclasses import asdict, dataclass
import itertools
import unittest


DTYPES = ("int8", "uint8", "int32", "int64", "float32", "float64")
DTYPE_BYTES = dict(zip(DTYPES, (1, 1, 4, 8, 4, 8)))
BYTE_TRANSPORT_DTYPES = (
    "int16",
    "float16",
    "bfloat16",
    "bool",
    "complex64",
    "complex128",
)
DTYPE_BYTES.update(zip(BYTE_TRANSPORT_DTYPES, (2, 2, 2, 1, 8, 16)))
METHODS = {
    "broadcast": "test_broadcast",
    "list": "test_all_gather",
    "tensor": "test_all_gather_single",
    "alltoall": "test_all_to_all_single",
    "allreduce": "test_all_reduce",
}


@dataclass(frozen=True)
class Case:
    op: str
    variant: str
    dtype: str = "float32"
    count: int = 17
    async_op: bool = False
    reduce_op: str = "sum"
    origin: str = "kuccl-extension"
    repeats: int = 3

    @property
    def id(self):
        return (
            f"{self.origin}/{self.op}/{self.variant}/{self.dtype}/n{self.count}"
            f"/{self.reduce_op}/{'async' if self.async_op else 'sync'}"
        )

    def to_dict(self):
        return dict(asdict(self), id=self.id)


def make_cases(suite, world_size, repeats=3):
    if suite not in ("smoke", "full") or world_size < 1 or repeats < 1:
        raise ValueError("suite, world size or repeats out of range")
    counts = (4,) if suite == "smoke" else (0, 1, 4, 17, 4097, 16384, 131073, 262145)
    dtypes = ("int32", "float32") if suite == "smoke" else DTYPES
    cases = []
    variants = (
        ("allreduce", "allreduce", "sum"),
        ("allreduce", "allreduce", "min"),
        ("allreduce", "allreduce", "max"),
        ("broadcast", "broadcast", "sum"),
        ("allgather", "list", "sum"),
        ("allgather", "tensor", "sum"),
        ("alltoall", "alltoall", "sum"),
    )
    for (op, variant, reduction), dtype, count, asynchronous in itertools.product(
        variants, dtypes, counts, (False, True)
    ):
        cases.append(
            Case(op, variant, dtype, count, asynchronous, reduction, "pytorch", repeats)
        )
    for op in ("allreduce", "broadcast", "allgather", "alltoall"):
        for asynchronous in (False, True):
            for reduction in ("sum", "min", "max") if op == "allreduce" else ("sum",):
                cases.append(
                    Case(
                        op,
                        "varying",
                        async_op=asynchronous,
                        reduce_op=reduction,
                        repeats=repeats,
                    )
                )
    for variant, op in (
        ("all_roots", "broadcast"),
        ("equal_splits", "alltoall"),
        ("mixed", "allreduce"),
        ("subgroups", "allreduce"),
        ("coalesced", "allreduce"),
    ):
        cases.append(
            Case(
                op,
                variant,
                count=18 if variant == "equal_splits" else 17,
                repeats=repeats,
            )
        )
    if suite == "full":
        for op in ("allgather", "alltoall"):
            cases.append(Case(op, "strided_input", count=18, async_op=True))
        for dtype in DTYPES:
            for op in ("allreduce", "broadcast", "allgather", "alltoall"):
                for reduction in (
                    ("sum", "min", "max") if op == "allreduce" else ("sum",)
                ):
                    cases.append(
                        Case(
                            op,
                            "varying",
                            dtype=dtype,
                            count=4097,
                            async_op=True,
                            reduce_op=reduction,
                            repeats=repeats,
                        )
                    )
        for dtype in BYTE_TRANSPORT_DTYPES:
            cases.append(
                Case(
                    "alltoall",
                    "byte_transport",
                    dtype=dtype,
                    count=37,
                    async_op=True,
                    repeats=repeats,
                )
            )
        cases.append(Case("allreduce", "coalesced", async_op=True))
        cases.append(Case("allreduce", "subgroups", async_op=True))
        for variant, op in (
            ("unequal_splits_rejected", "alltoall"),
            ("product_rejected", "allreduce"),
            ("noncontiguous_output_rejected", "alltoall"),
        ):
            cases.append(Case(op, variant))
    return cases


def memory_estimate(case, world_size):
    """Conservative peak tensor bytes per rank; excludes library scratch/storage."""
    count = max(case.count, 262145 if case.variant == "mixed" else 0)
    return DTYPE_BYTES[case.dtype] * max(count, 1) * (8 * world_size + 16)


def run_upstream(case, rank, world_size):
    import torch
    import torch.distributed as dist
    import pytorch_cpu_upstream as upstream

    class DistAdapter:
        def __getattr__(self, name):
            if name == "all_gather_single" and not hasattr(dist, name):
                return dist.all_gather_into_tensor
            return getattr(dist, name)

    class Adapter(upstream.UpstreamCpuCollectives, unittest.TestCase):
        def assertEqual(self, left, right, *args, **kwargs):
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            else:
                super().assertEqual(left, right, *args, **kwargs)

        def _reduce_ops(self, dtype):
            return (getattr(dist.ReduceOp, case.reduce_op.upper()),)

    upstream.dist = DistAdapter()
    upstream.COUNTS = (case.count,)
    upstream.ASYNC_OPS = (case.async_op,)
    adapter = Adapter()
    adapter.rank, adapter.world_size, adapter.device = (
        rank,
        world_size,
        torch.device("cpu"),
    )
    adapter.dtypes = (getattr(torch, case.dtype),)
    getattr(adapter, METHODS[case.variant])()


def run_extension(case, rank, world_size, profile):
    import torch
    import torch.distributed as dist

    dtype = getattr(torch, case.dtype)
    if profile == "gloo" and (
        case.variant == "byte_transport"
        or (case.variant == "strided_input" and case.op == "alltoall")
    ):
        raise unittest.SkipTest(
            "KuCCL byte-transport/input-normalization extension; outside Gloo baseline"
        )

    def pattern(source, destination, count, epoch):
        # Each source/destination/element/iteration matters; values remain small
        # enough for int8 SUM at supported world sizes (including overflow cast).
        index = torch.arange(count, dtype=torch.int64)
        if dtype == torch.bool:
            # Encode all four bits of both ranks, rather than turning almost
            # every nonzero integer into True. Each 8-element block carries
            # a distinct source/destination address for up to 16 ranks.
            owner = torch.where(index % 8 < 4, source, destination)
            return (
                (owner.bitwise_right_shift(index % 4) & 1) ^ ((index // 8 + epoch) % 2)
            ).bool()
        value = (index * 7 + source * 3 + destination * 5 + epoch * 11) % 97
        if dtype.is_complex:
            imaginary = (index * 11 + source * 7 + destination * 3 + epoch * 13) % 89
            return value.to(dtype) + imaginary.to(dtype) * 1j
        return value.to(dtype)

    def check(actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def wait(work, asynchronous):
        if asynchronous:
            if work is None or work.wait() is False:
                raise AssertionError(
                    "async collective did not return a successful Work"
                )
            if not work.is_completed():
                raise AssertionError("Work incomplete after wait")
        elif work is not None:
            raise AssertionError("synchronous collective returned a Work")

    def exercise(
        op,
        count,
        epoch,
        group=None,
        members=None,
        root=None,
        splits=False,
        strided=False,
    ):
        members = list(range(world_size)) if members is None else members
        size = len(members)
        asynchronous = case.async_op
        if op == "allreduce":
            tensor = pattern(rank, 0, count, epoch)
            contributions = torch.stack(
                [pattern(r, 0, count, epoch).to(torch.int64) for r in members]
            )
            if case.reduce_op == "sum":
                expected = contributions.sum(0).to(dtype)
            elif case.reduce_op == "min":
                expected = contributions.amin(0).to(dtype)
            else:
                expected = contributions.amax(0).to(dtype)
            reduction = getattr(dist.ReduceOp, case.reduce_op.upper())
            wait(
                dist.all_reduce(
                    tensor, op=reduction, group=group, async_op=asynchronous
                ),
                asynchronous,
            )
            check(tensor, expected)
        elif op == "broadcast":
            root = members[-1] if root is None else root
            tensor = pattern(rank, 0, count, epoch)
            wait(
                dist.broadcast(tensor, src=root, group=group, async_op=asynchronous),
                asynchronous,
            )
            check(tensor, pattern(root, 0, count, epoch))
        elif op == "allgather":
            tensor = pattern(rank, 0, count, epoch)
            if strided:
                storage = torch.zeros(count * 2, dtype=dtype)
                storage[::2] = tensor
                tensor = storage[::2]
            output = torch.empty(size * count, dtype=dtype)
            wait(
                dist.all_gather_into_tensor(
                    output, tensor, group=group, async_op=asynchronous
                ),
                asynchronous,
            )
            check(output, torch.cat([pattern(r, 0, count, epoch) for r in members]))
        else:
            tensor = torch.cat([pattern(rank, d, count, epoch) for d in members])
            if strided:
                storage = torch.zeros(tensor.numel() * 2, dtype=dtype)
                storage[::2] = tensor
                tensor = storage[::2]
            # Exercise row-based split units and multidimensional trailing shape.
            width = 2 if count % 2 == 0 else 1
            tensor = tensor.reshape(size * count // width, width)
            output = torch.empty_like(tensor, memory_format=torch.contiguous_format)
            kwargs = {}
            if splits:
                kwargs = dict(
                    input_split_sizes=[count // width] * size,
                    output_split_sizes=[count // width] * size,
                )
            wait(
                dist.all_to_all_single(
                    output, tensor, group=group, async_op=asynchronous, **kwargs
                ),
                asynchronous,
            )
            check(
                output.flatten(),
                torch.cat([pattern(s, rank, count, epoch) for s in members]),
            )
            # Neither transport may modify its send tensor.
            check(
                tensor.flatten(),
                torch.cat([pattern(rank, d, count, epoch) for d in members]),
            )

    if case.variant == "subgroups":
        # Every world rank calls new_group in identical order, including nonmembers.
        # Repeated membership deliberately creates separate communicator contexts.
        memberships = [
            list(range(world_size)),
            list(range(0, world_size, 2)),
            list(range(1, world_size, 2)),
            list(range(world_size)),
        ]
        groups = []
        try:
            for members in memberships:
                if members:
                    groups.append((members, dist.new_group(members)))
            for index, (members, group) in enumerate(groups):
                if rank in members:
                    for op in ("allreduce", "broadcast", "allgather", "alltoall"):
                        exercise(op, 18, index + 1, group, members)
        finally:
            for members, group in reversed(groups):
                if rank in members:
                    dist.destroy_process_group(group)
    elif case.variant == "coalesced":
        tensors = [pattern(rank, 0, count, 1) for count in (1, 17, 257)]
        expected = [
            torch.stack([pattern(r, 0, t.numel(), 1) for r in range(world_size)])
            .sum(0)
            .to(dtype)
            for t in tensors
        ]
        # This public API returns a Future (unlike all_reduce's Work) on 2.9.
        completion = dist.all_reduce_coalesced(tensors, async_op=case.async_op)
        if case.async_op:
            if completion is None:
                raise AssertionError(
                    "coalesced async call returned no completion handle"
                )
            completion.wait()
            done = (
                completion.done()
                if hasattr(completion, "done")
                else completion.is_completed()
            )
            if not done:
                raise AssertionError(
                    "coalesced completion handle incomplete after wait"
                )
        elif completion is not None:
            raise AssertionError("coalesced sync call returned a completion handle")
        for actual, reference in zip(tensors, expected):
            check(actual, reference)
    elif case.variant.endswith("_rejected"):
        if profile == "gloo":
            raise unittest.SkipTest(
                "KuCCL adapter rejection contract; Gloo has different capabilities"
            )
        if case.variant == "unequal_splits_rejected" and world_size == 1:
            raise unittest.SkipTest("unequal peer splits require at least two ranks")
        try:
            if case.variant == "product_rejected":
                dist.all_reduce(torch.ones(17), op=dist.ReduceOp.PRODUCT)
            elif case.variant == "noncontiguous_output_rejected":
                source = torch.ones(world_size * 17)
                dist.all_to_all_single(torch.empty(world_size * 34)[::2], source)
            else:
                ins = [rank + destination + 1 for destination in range(world_size)]
                outs = [source + rank + 1 for source in range(world_size)]
                dist.all_to_all_single(
                    torch.empty(sum(outs)), torch.ones(sum(ins)), outs, ins
                )
        except (RuntimeError, ValueError) as error:
            expected_text = {
                "product_rejected": ("reduce", "Reduce", "op", "Op"),
                "noncontiguous_output_rejected": ("contiguous",),
                "unequal_splits_rejected": ("split",),
            }[case.variant]
            if not any(text in str(error) for text in expected_text):
                raise AssertionError(
                    f"unrelated error while checking rejection: {error}"
                ) from error
        else:
            raise AssertionError("unsupported input was silently accepted")
    elif case.variant == "all_roots":
        for root in range(world_size):
            exercise("broadcast", case.count, root + 1, root=root)
    elif case.variant == "mixed":
        for epoch in range(case.repeats):
            # Reuse communicator storage across shrinking/growing buffers and ops.
            for count in (1, 4097, 262145, 17):
                for op in ("allreduce", "allgather", "broadcast", "alltoall"):
                    exercise(op, count, epoch)
    else:
        for epoch in range(case.repeats):
            exercise(
                case.op,
                case.count,
                epoch,
                splits=case.variant == "equal_splits",
                strided=case.variant == "strided_input",
            )


def run_case(case, rank, world_size, profile):
    if case.origin == "pytorch":
        run_upstream(case, rank, world_size)
    else:
        run_extension(case, rank, world_size, profile)
