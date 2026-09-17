#!/usr/bin/env python3
"""Run a repository Sage script in pinned Docker with a hard wall-clock limit."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IMAGE = "sagemath/sagemath:10.10.beta10"
MODES = ("invariants", "pari-bound", "pari-init", "mwrank-bound", "simon-bound", "simon-shifted", "small-isogenies", "saturation", "analytic", "sections", "section-scan", "section-scan-lll", "rational-section-scan", "verify-gpu", "search")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30,
                        help="hard wall-clock limit in seconds (default: 30)")
    parser.add_argument("--tag", help="distinct result suffix for bounded experiments")
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    script = args.script.resolve()
    if script != HERE / "rank31_t_minus_47_80_sage.py":
        parser.error("this runner currently supports rank31_t_minus_47_80_sage.py")
    if "--mode" not in args.script_args:
        parser.error("Sage script requires --mode")
    try:
        mode = args.script_args[args.script_args.index("--mode") + 1]
    except IndexError:
        parser.error("--mode requires a value")
    if mode not in MODES:
        parser.error(f"unknown mode {mode!r}")
    if args.tag and (not args.tag.replace('-', '').replace('_', '').isalnum()):
        parser.error("--tag must contain only letters, digits, hyphens or underscores")
    if "--output" in args.script_args:
        parser.error("--output is managed by the runner")
    args.script = script
    args.mode = mode
    return args


def write_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def remove_container(docker: str, name: str) -> None:
    # Force removal kills the container's PID namespace and all Sage descendants.
    subprocess.run([docker, "rm", "-f", name], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=10, check=False)


def run(args, docker="docker") -> int:
    tag = getattr(args, "tag", None)
    suffix = f"-{tag}" if tag else ""
    output = HERE / "results" / f"rank31-t-minus-47-80-sage-{args.mode}{suffix}.json"
    name = "rank31-sage-" + uuid.uuid4().hex
    started = time.monotonic()
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True, check=True).stdout.strip()
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix="sage-", suffix=".json",
                                     delete=False) as stream:
        temporary = Path(stream.name)
    relative_script = "/work/" + str(args.script.relative_to(ROOT))
    relative_output = "/work/" + str(temporary.relative_to(ROOT))
    sage_command = ["/usr/bin/sage", "-python", relative_script, *args.script_args,
                    "--output", relative_output]
    # Argument values are passed as positional shell parameters, never interpolated.
    shell = '"$@"; rc=$?; chown "$HOST_UID:$HOST_GID" "$RESULT_PATH"; exit "$rc"'
    command = [docker, "run", "--rm", "--name", name, "--user", "0:0",
               "--mount", f"type=bind,src={ROOT},dst=/work", "--workdir", "/work",
               "--env", f"HOST_UID={os.getuid()}", "--env", f"HOST_GID={os.getgid()}",
               "--env", f"RESULT_PATH={relative_output}", "--entrypoint", "/bin/sh",
               IMAGE, "-c", shell, "sage-runner", *sage_command]
    invocation = shlex.join([str((HERE / Path(__file__).name).relative_to(ROOT)), "--timeout", str(args.timeout),
                             *(["--tag", tag] if tag else []),
                             str(args.script.relative_to(ROOT)), *args.script_args])
    status = "error"
    error = None
    returncode = 1
    process = None
    try:
        process = subprocess.Popen(command, cwd=ROOT)
        try:
            returncode = process.wait(timeout=args.timeout)
            if returncode == 0:
                status = "completed"
            else:
                error = f"Docker/Sage exited with code {returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
            error = f"wall-clock timeout after {args.timeout:g} seconds"
    except (OSError, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # Also runs on normal completion; --rm makes this a no-op in that case.
        try:
            remove_container(docker, name)
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = f"container cleanup failed: {exc}"
            status = "error"
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    elapsed = time.monotonic() - started
    result = {}
    if status == "completed":
        try:
            result = json.loads(temporary.read_text(encoding="utf-8"))
            if result.get("status") == "error":
                status = "error"
                error = f"Sage step failed: {result.get('error')}"
        except (OSError, ValueError) as exc:
            status = "error"
            error = f"Sage produced no valid JSON: {exc}"
    temporary.unlink(missing_ok=True)
    result.update({"mode": args.mode, "backend": args.mode.removesuffix("-bound") if args.mode.endswith("-bound") else None,
                   "runner_invocation": invocation, "sage_command": shlex.join(sage_command),
                   "image": IMAGE, "source_commit": source_commit,
                   "elapsed_seconds": elapsed, "status": status, "timed_out": status == "timeout"})
    if error:
        result["error"] = error
    if status != "completed":
        result.setdefault("sage_version", None)
        result.setdefault("value", None)
    else:
        result["value"] = result.get("steps")
    write_json(output, result)
    print(f"{status}: {output}" + (f" ({error})" if error else ""), file=sys.stderr)
    return 0 if status == "completed" else 124 if status == "timeout" else 130 if status == "interrupted" else 1


if __name__ == "__main__":
    sys.exit(run(parse_args(), docker=os.environ.get("RANK31_DOCKER_BIN", "docker")))
