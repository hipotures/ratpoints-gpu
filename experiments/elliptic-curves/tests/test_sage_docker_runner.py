from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("sage_docker_runner", "run_sage_docker.py")
sage_script = load("rank31_sage", "rank31_t_minus_47_80_sage.py")


class SageDockerRunnerTests(unittest.TestCase):
    def test_backend_modes_are_explicit(self):
        for mode in ("invariants", "pari-bound", "mwrank-bound"):
            self.assertEqual(sage_script.parse_args(["--mode", mode]).mode, mode)
            args = runner.parse_args(["--timeout", "2", str(HERE / "rank31_t_minus_47_80_sage.py"),
                                      "--mode", mode])
            self.assertEqual(args.mode, mode)
        for parser in (lambda: sage_script.parse_args(["--mode", "descent"]),
                       lambda: runner.parse_args([str(HERE / "rank31_t_minus_47_80_sage.py"),
                                                  "--mode", "descent"])):
            with self.assertRaises(SystemExit):
                parser()

    def test_result_file_is_host_owned_and_atomically_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            runner.write_json(path, {"status": "completed"})
            self.assertEqual(path.stat().st_uid, os.getuid())
            runner.write_json(path, {"status": "timeout", "timed_out": True})
            self.assertEqual(json.loads(path.read_text())["status"], "timeout")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_timeout_forces_container_cleanup_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            here = root / "experiments" / "elliptic-curves"
            (here / "results").mkdir(parents=True)
            script = here / "rank31_t_minus_47_80_sage.py"
            script.touch()
            args = SimpleNamespace(script=script, script_args=["--mode", "pari-bound"],
                                   mode="pari-bound", timeout=0.01)

            class HungProcess:
                def wait(self, timeout=None):
                    if timeout == args.timeout:
                        raise subprocess.TimeoutExpired("docker", timeout)
                    return -15

                def poll(self):
                    return 0

            with patch.object(runner, "HERE", here), patch.object(runner, "ROOT", root), \
                 patch.object(runner.subprocess, "Popen", return_value=HungProcess()) as launch, \
                 patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout="abc123\n")), \
                 patch.object(runner, "remove_container") as cleanup:
                self.assertEqual(runner.run(args), 124)
            self.assertEqual(cleanup.call_count, 1)
            command = launch.call_args.args[0]
            self.assertIn("--entrypoint", command)
            self.assertIn("/usr/bin/sage", command)
            self.assertNotIn("mwrank-bound", command)
            result = json.loads((here / "results" / "rank31-t-minus-47-80-sage-pari-bound.json").read_text())
            self.assertEqual(result["status"], "timeout")
            self.assertTrue(result["timed_out"])
            self.assertEqual(result["source_commit"], "abc123")
            self.assertIsNone(result["value"])
            self.assertEqual(list((here / "results").glob("sage-*.json")), [])


if __name__ == "__main__":
    unittest.main()
