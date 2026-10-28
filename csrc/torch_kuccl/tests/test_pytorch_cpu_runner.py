"""Host regressions for the standalone PyTorch CPU collective runner."""

import json  # noqa: E402  # noqa: E402
import ast
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest  # noqa: E402  # noqa: E402
from unittest import mock

COMM = Path(__file__).resolve().parents[3] / "benchmarks" / "communication"
sys.path.insert(0, str(COMM))
import pytorch_cpu_suite as suite  # noqa: E402
import test_pytorch_collectives as runner  # noqa: E402


class PlanningTests(unittest.TestCase):
    def test_smoke_covers_four_operations_and_both_work_modes(self):
        cases = suite.make_cases("smoke", 2, repeats=2)
        official = [c for c in cases if c.origin == "pytorch"]
        self.assertEqual(
            {c.op for c in official},
            {"allreduce", "broadcast", "allgather", "alltoall"},
        )
        self.assertEqual({c.async_op for c in official}, {False, True})
        self.assertEqual(
            {c.reduce_op for c in official if c.op == "allreduce"},
            {"sum", "min", "max"},
        )
        self.assertEqual(len({c.id for c in cases}), len(cases))

    def test_full_retains_upstream_empty_cases(self):
        cases = suite.make_cases("full", 4, repeats=3)
        empty = {c.op for c in cases if c.origin == "pytorch" and c.count == 0}
        self.assertEqual(empty, {"allreduce", "broadcast", "allgather", "alltoall"})
        self.assertTrue(any(c.variant == "subgroups" for c in cases))
        self.assertTrue(any(c.variant == "mixed" for c in cases))
        self.assertTrue(any(c.variant == "coalesced" for c in cases))
        self.assertTrue(any(c.count > 131072 for c in cases))

    def test_elementwise_patterns_cover_every_reduction(self):
        varying = [
            c
            for c in suite.make_cases("full", 4)
            if c.op == "allreduce" and c.variant == "varying"
        ]
        self.assertEqual({c.reduce_op for c in varying}, {"sum", "min", "max"})

    def test_frozen_official_method_bodies_match_provenance(self):
        source = (COMM / "pytorch_cpu_upstream.py").read_text(encoding="utf-8")
        metadata = json.loads(
            (COMM / "pytorch_cpu_upstream.json").read_text(encoding="utf-8")
        )
        lines = source.splitlines(keepends=True)
        methods = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
        ]
        self.assertEqual({method.name for method in methods}, set(metadata["methods"]))
        for method in methods:
            body = "".join(lines[method.lineno - 1 : method.end_lineno])
            self.assertEqual(
                hashlib.sha256(body.encode()).hexdigest(),
                metadata["methods"][method.name]["sha256"],
            )

    def test_memory_estimate_accounts_for_world_size(self):
        case = next(
            c
            for c in suite.make_cases("full", 2)
            if c.op == "alltoall" and c.count > 131072
        )
        self.assertGreater(
            suite.memory_estimate(case, 16), suite.memory_estimate(case, 2)
        )

    def test_profile_selection_is_explicit(self):
        self.assertEqual(
            runner.profile_environment("xucg", 4)["KUCCL_BACKEND_TYPE"], "xucg"
        )
        self.assertEqual(
            runner.profile_environment("kupl-hier", 4)["KUCCL_KUPL_AR_ALGO"], "hier"
        )
        self.assertEqual(
            runner.profile_environment("kupl-default", 1)["KUCCL_KUPL_AG_THREADS"], "1"
        )
        with self.assertRaises(ValueError):
            runner.profile_environment("typo", 4)


class VerdictTests(unittest.TestCase):
    def test_partial_case_log_preserves_valid_prefix_and_reports_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            first = {"status": "PASS", "case": {"id": "first"}}
            path.write_text(
                json.dumps(first) + '\n{"status": "FAIL",', encoding="utf-8"
            )
            records, error = runner.read_records(Path(directory))
            self.assertEqual(records, [first])
            self.assertIn("complete case log", error)

    def test_route_evidence_distinguishes_subgroups_and_flat_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank-0.log"
            path.write_text(
                "[AR-route] bytes=4B ws=4 → hier_die\n"
                "[AG-route] chunk=4B ws=4 → hier_die\n"
                "[AG-route] chunk=4B ws=2 → linear(fence)\n",
                encoding="utf-8",
            )
            routes = runner.observed_routes(Path(directory), 4)
            self.assertEqual(
                routes, {"allreduce": ["hier_die"], "allgather": ["hier_die"]}
            )

    def test_missing_or_partial_progress_does_not_abort_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            self.assertIsNone(runner.read_progress(path))
            path.write_text('{"index":', encoding="utf-8")
            self.assertIsNone(runner.read_progress(path))
            path.write_text(json.dumps({"index": 0}), encoding="utf-8")
            self.assertEqual(runner.read_progress(path), 0)

    def test_missing_rank_never_passes(self):
        records = [{"rank": 0, "status": "PASS"}]
        self.assertEqual(runner.merge_rank_results(records, 2)["status"], "FAIL")

    def test_any_rank_failure_is_global_failure(self):
        records = [
            {"rank": 0, "status": "PASS"},
            {"rank": 1, "status": "FAIL", "error": "bad data"},
        ]
        self.assertEqual(runner.merge_rank_results(records, 2)["status"], "FAIL")

    def test_duplicate_or_mixed_skip_rank_cannot_pass(self):
        for records in (
            [{"rank": 0, "status": "PASS"}] * 2,
            [{"rank": 0, "status": "SKIP"}, {"rank": 1, "status": "PASS"}],
        ):
            self.assertEqual(runner.merge_rank_results(records, 2)["status"], "FAIL")

    def test_all_rank_pass_and_all_rank_skip(self):
        for status in ("PASS", "SKIP"):
            records = [{"rank": r, "status": status} for r in range(3)]
            self.assertEqual(runner.merge_rank_results(records, 3)["status"], status)

    def test_missing_or_failed_cases_cannot_make_successful_report(self):
        self.assertFalse(runner.report_passed([]))
        self.assertFalse(runner.report_passed([{"status": "SKIP"}]))
        self.assertFalse(
            runner.report_passed([{"status": "PASS"}, {"status": "NOT_RUN"}])
        )
        self.assertFalse(runner.report_passed([{"status": "PASS"}, {"status": "FAIL"}]))
        self.assertTrue(runner.report_passed([{"status": "PASS"}, {"status": "SKIP"}]))

    def test_child_failure_kills_peers_and_marks_unreported_cases(self):
        class Process:
            def __init__(self, code=None):
                self.code, self.killed = code, False

            def poll(self):
                return self.code

            def kill(self):
                self.code, self.killed = -9, True

            def wait(self):
                return self.code

        failed, peer = Process(7), Process()
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parse_args(["--profiles", "gloo"])
            with mock.patch.object(
                runner.subprocess, "Popen", side_effect=[failed, peer]
            ):
                result = runner.supervise(args, "gloo", 2, 1, Path(directory) / "run")
            self.assertTrue(peer.killed)
            self.assertFalse(result["passed"])
            self.assertIn("NOT_RUN", result["counts"])
            self.assertIn("rank process failed", result["fatal"])

    def test_watchdog_terminates_a_hung_rank(self):
        process = mock.Mock()
        process.poll.return_value = None
        clock = iter((0, 100))
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parse_args(
                ["--profiles", "gloo", "--world-sizes", "1", "--timeout", "1"]
            )
            with (
                mock.patch.object(runner.subprocess, "Popen", return_value=process),
                mock.patch.object(
                    runner.time, "monotonic", side_effect=lambda: next(clock)
                ),
            ):
                result = runner.supervise(args, "gloo", 1, 1, Path(directory) / "run")
            process.kill.assert_called_once()
            self.assertFalse(result["passed"])
            self.assertIn("watchdog", result["fatal"])


if __name__ == "__main__":
    unittest.main()
