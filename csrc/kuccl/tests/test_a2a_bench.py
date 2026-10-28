"""Host-only regressions for A2A payloads and shell verdicts; no KUPL runtime required.

Run: python -m unittest discover -s csrc/kuccl/tests -p test_a2a_bench.py -v
The C++ checks compile the actual payload/oracle assignments against an independent
transpose. Shell checks execute the real scripts with a controlled MPI launcher.
These checks do not validate KUPL memory mappings, fences or SDMA.
"""

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class PayloadTests(unittest.TestCase):
    def test_payload_and_oracle_match_independent_transpose(self):
        compiler = shutil.which("g++")
        if not compiler:
            self.skipTest("g++ required for host payload checks")
        for filename in ("bench_a2a_mpi.cpp",):
            with (
                self.subTest(source=filename),
                tempfile.TemporaryDirectory() as directory,
            ):
                source = (EXAMPLES / filename).read_text(encoding="utf-8")
                payloads = re.findall(r"send\[s \* count \+ m\]\s*=\s*([^;]+);", source)
                oracle = re.search(
                    r"const int32_t expect = ([^;]+);",
                    source[source.index("for (int shift") :],
                ).group(1)
                self.assertTrue(payloads)
                for payload in payloads:
                    program = r"""
#include <cstdint>
#include <cstdio>
#include <vector>
int main() {
  for (int world : {1, 2, 3, 8, 16}) for (size_t count : {1, 2, 7, 1024, 4096}) {
    std::vector<std::vector<int32_t>> sends(world, std::vector<int32_t>(world*count));
    for (int rank=0; rank<world; ++rank) {
      auto &send = sends[rank];
      for (size_t s=0; s<size_t(world); ++s) for (size_t m=0; m<count; ++m)
        send[s*count+m] = PAYLOAD;
    }
    for (int rank=0; rank<world; ++rank) {
      for (size_t s=0; s<size_t(world); ++s) for (size_t m=0; m<count; ++m) {
        size_t i=s*count+m;
        const int32_t expect = ORACLE;
        // MPI semantics: receiver rank gets sender s's block addressed to rank.
        const int32_t actual = sends[s][rank*count+m];
        if (actual != expect) {
          std::fprintf(stderr,"oracle mismatch: W=%d C=%zu receiver=%d source=%zu m=%zu got=%d expect=%d\n",
                       world,count,rank,s,m,actual,expect);
          return 1;
        }
        // Source identities must differ, even at the same destination and index.
        if (world>1 && sends[0][rank*count+m] == sends[1][rank*count+m]) {
          std::fputs("payload cannot distinguish source ranks\n",stderr); return 2;
        }
      }
    }
  }
}
""".replace("PAYLOAD", payload).replace("ORACLE", oracle)
                    cpp = Path(directory) / "payload.cpp"
                    exe = Path(directory) / (
                        "payload.exe" if os.name == "nt" else "payload"
                    )
                    cpp.write_text(program, encoding="utf-8")
                    subprocess.run(
                        [
                            compiler,
                            "-std=c++17",
                            "-O2",
                            *(["-static"] if os.name == "nt" else []),
                            str(cpp),
                            "-o",
                            str(exe),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    result = subprocess.run([str(exe)], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)


class BenchmarkControlFlowTests(unittest.TestCase):
    """Run the actual C++ benchmark with a deterministic transport double.

    The double only makes completion/error paths executable without MPI/KUPL;
    it is not evidence that either real transport works.
    """

    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("g++ required for benchmark control-flow checks")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        directory = Path(cls.temp.name)
        (directory / "mpi.h").write_text(
            r"""
#pragma once
using MPI_Comm = int;
constexpr int MPI_COMM_WORLD=0, MPI_SUCCESS=0, MPI_BYTE=1, MPI_INT=2, MPI_DOUBLE=3, MPI_MAX=4;
int MPI_Init(int*,char***); int MPI_Finalize();
int MPI_Comm_rank(MPI_Comm,int*); int MPI_Comm_size(MPI_Comm,int*);
int MPI_Abort(MPI_Comm,int); int MPI_Bcast(void*,int,int,int,MPI_Comm);
int MPI_Allgather(const void*,int,int,void*,int,int,MPI_Comm);
int MPI_Allreduce(const void*,void*,int,int,int,MPI_Comm);
int MPI_Barrier(MPI_Comm);
""",
            encoding="utf-8",
        )
        (directory / "runtime.cpp").write_text(
            r"""
#include <mpi.h>
#include <kuccl.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
struct kucclComm {};
struct kucclRequest {};
static int calls=0;
static bool fault(const char* name) {
  const char* f=std::getenv("TEST_FAULT"); return f && std::strcmp(f,name)==0;
}
int MPI_Init(int*,char***) { return 0; }
int MPI_Finalize() { return 0; }
int MPI_Comm_rank(MPI_Comm,int* r) { *r=0; return 0; }
int MPI_Comm_size(MPI_Comm,int* w) { *w=2; return 0; }
int MPI_Abort(MPI_Comm,int code) { std::exit(code); }
int MPI_Bcast(void*,int,int,int,MPI_Comm) { return 0; }
int MPI_Barrier(MPI_Comm) { return 0; }
int MPI_Allgather(const void* s,int n,int,void* r,int,int,MPI_Comm) {
  std::memcpy(r,s,n); std::memcpy(static_cast<char*>(r)+n,s,n); return 0;
}
int MPI_Allreduce(const void* s,void* r,int n,int dt,int,MPI_Comm) {
  std::memcpy(r,s,n*(dt==MPI_DOUBLE?sizeof(double):sizeof(int)));
  // Inject an error reported by the other rank, although local data is correct.
  if (dt==MPI_INT && fault("remote_failure")) *static_cast<int*>(r)=1;
  return 0;
}
kucclResult_t kucclGetUniqueId(kucclUniqueId*) { return KuCCL_SUCCESS; }
kucclResult_t kucclCommInitRankConfig(kucclComm_t* c,int,kucclUniqueId,int,kucclConfig_t*) {
  *c=new kucclComm; return KuCCL_SUCCESS;
}
kucclResult_t kucclCommDestroy(kucclComm_t c) { delete c; return KuCCL_SUCCESS; }
const char* kucclGetLastError() { return "injected transport failure"; }
kucclResult_t kucclAllToAll(const void* s,void* r,size_t count,kucclDataType_t,kucclComm_t,kucclRequest_t* req) {
  ++calls;
  if (calls>1 && fault("timed_start_error")) return KuCCL_FAIL;
  const auto* in=static_cast<const int32_t*>(s); auto* out=static_cast<int32_t*>(r);
  for (size_t sender=0;sender<2;++sender) for (size_t m=0;m<count;++m)
    out[sender*count+m]=in[m]+int32_t(sender*2*count);
  if (calls>1 && fault("timed_corruption")) out[0]=-7;
  *req=new kucclRequest; return KuCCL_SUCCESS;
}
kucclResult_t kucclTest(kucclRequest_t* req,int* flag,kucclStatus_t* st) {
  if (fault("test_error")) return KuCCL_FAIL;
  delete *req; *req=nullptr; *flag=1;
  *st=fault("completion_error")?KuCCL_STATUS_ERROR:KuCCL_STATUS_COMPLETE;
  return KuCCL_SUCCESS;
}
""",
            encoding="utf-8",
        )
        cls.exe = directory / ("benchmark.exe" if os.name == "nt" else "benchmark")
        result = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                *(["-static"] if os.name == "nt" else []),
                "-I",
                str(directory),
                "-I",
                str(ROOT / "include"),
                str(EXAMPLES / "bench_a2a_mpi.cpp"),
                str(directory / "runtime.cpp"),
                "-o",
                str(cls.exe),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise AssertionError(result.stderr)

    def run_bench(self, fault="", args=(3, 4, 2, 1)):
        return subprocess.run(
            [str(self.exe), *map(str, args)],
            env=dict(os.environ, TEST_FAULT=fault),
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_correct_exchange_and_zero_warmup_pass(self):
        for warmup in (0, 1):
            with self.subTest(warmup=warmup):
                result = self.run_bench(args=(3, 4, 2, warmup))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("VERIFY: ALL PASSED", result.stdout)

    def test_other_rank_failure_cannot_print_global_pass(self):
        result = self.run_bench("remote_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("VERIFY: ALL PASSED", result.stdout)

    def test_request_errors_terminate_without_polling_forever(self):
        for fault in ("test_error", "completion_error", "timed_start_error"):
            with self.subTest(fault=fault):
                result = self.run_bench(fault)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("FAILED: injected transport failure", result.stderr)

    def test_timed_data_corruption_fails(self):
        result = self.run_bench("timed_corruption")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VERIFY FAIL", result.stderr)

    def test_invalid_and_oversized_shapes_fail_before_allocation(self):
        for args in ((1, 4, 2, 1), (3, 4, 0, 1), (3, 4, 2, -1), (30, 30, 1, 0)):
            with self.subTest(args=args):
                self.assertNotEqual(self.run_bench(args=args).returncode, 0)


class RunnerTests(unittest.TestCase):
    def run_runner(self, filename, mode):
        bash = shutil.which("bash")
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if os.name == "nt" and git_bash.exists():
            bash = str(git_bash)
        if not bash:
            self.skipTest("Bash required for launcher checks")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = root / "examples"
            binary = root / "bin"
            examples.mkdir()
            binary.mkdir()
            shutil.copyfile(EXAMPLES / filename, examples / filename)
            (root / "kuccl_env.sh").write_text(
                'export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bin:$PATH"\n'
                "export KUPL_DIR=/unused\n",
                encoding="utf-8",
            )
            compiler = binary / "mpicxx"
            compiler.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            launcher = binary / "mpirun"
            launcher.write_text(
                """#!/usr/bin/env bash
echo "native_algorithm=${KUPL_SHM_ALLTOALL_ALGORITHM-unset}"
case "$FAKE_MPI_MODE" in
  pass) echo 'VERIFY: ALL PASSED'; echo '[a2a] alltoall verify: ALL PASSED'; exit 0;;
  false_success) echo 'VERIFY: ALL PASSED'; echo '[a2a] alltoall verify: ALL PASSED'; exit 7;;
  native_failure)
    if [ "${KUCCL_KUPL_A2A_ALGO-}" = native ]; then echo 'VERIFY: FAILED'; exit 1; fi
    echo 'VERIFY: ALL PASSED'; exit 0;;
  fail) echo 'SDMA ERROR: injected transport failure'; echo 'VERIFY: FAILED'; exit 1;;
esac
""",
                encoding="utf-8",
            )
            compiler.chmod(0o755)
            launcher.chmod(0o755)
            env = dict(os.environ, FAKE_MPI_MODE=mode, KUPL_SHM_ALLTOALL_ALGORITHM="2")
            result = subprocess.run(
                [bash, "--noprofile", "--norc", filename, "2"],
                cwd=examples,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=300,
            )
            logs = {
                p.name: p.read_text(encoding="utf-8", errors="replace")
                for p in examples.glob("logs/**/*.txt")
            }
            return result, logs

    def test_successful_launcher_passes(self):
        result, _ = self.run_runner("bench_a2a_mpi.sh", "pass")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_success_text_does_not_hide_nonzero_exit(self):
        result, _ = self.run_runner("bench_a2a_mpi.sh", "false_success")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_failure_fails_runner_and_preserves_diagnostics(self):
        result, logs = self.run_runner("bench_a2a_mpi.sh", "fail")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(
            any("SDMA ERROR: injected" in text for text in logs.values()), logs
        )

    def test_native_failure_is_not_treated_as_expected_success(self):
        result, _ = self.run_runner("bench_a2a_mpi.sh", "native_failure")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
