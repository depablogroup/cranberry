#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cranberry.data import data_path


DEFAULT_PDB = "examples/ggcGCAAgcc_cg_vs_conect.pdb"


@dataclass(frozen=True)
class GpuSample:
    timestamp: float
    utilization_percent: float
    memory_mib: float
    power_watt: float | None


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    wall_seconds: float
    stdout_path: str
    stderr_path: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark local CUDA/MPS MD and REMD modes and write YAML.")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/local_cuda_modes.yaml"))
    parser.add_argument("--work-dir", type=Path, default=Path("benchmarks/results/local_cuda_modes_runs"))
    parser.add_argument("--pdb", type=Path, default=None, help="Prepared CG PDB; defaults to bundled ggcGCAAgcc")
    parser.add_argument("--md-steps", type=int, default=20000)
    parser.add_argument("--remd-steps", type=int, default=20000)
    parser.add_argument("--warmup-note", default="OpenMM context creation is included in wall time; speed is also parsed from logs when available.")
    parser.add_argument("--parallel-md", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--skip-single-md", action="store_true")
    parser.add_argument("--skip-mps-md", action="store_true")
    parser.add_argument("--skip-remd", action="store_true")
    parser.add_argument("--start-mps", action="store_true", help="Start and stop a local CUDA MPS daemon around MPS MD and MPI REMD runs.")
    parser.add_argument("--remd-mpi-ranks", type=int, default=1, help="Launch REMD through mpirun with this many ranks; 1 runs serially.")
    parser.add_argument(
        "--remd-n-analysis",
        type=int,
        nargs="*",
        default=[0, 10],
        help="Target online-analysis writes for REMD benchmark. Defaults to both disabled and about 10 writes.",
    )
    args = parser.parse_args(argv)

    pdb = args.pdb or data_path(DEFAULT_PDB)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    benchmark: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": hostname(),
        "pdb": str(pdb),
        "gpu": nvidia_smi_summary(),
        "notes": [args.warmup_note],
        "runs": [],
    }

    if not args.skip_single_md:
        benchmark["runs"].append(run_single_md(pdb, args.work_dir / "single_md_cuda", args.md_steps, args.sample_interval))

    if not args.skip_mps_md:
        benchmark["runs"].extend(
            run_mps_md_series(
                pdb=pdb,
                work_dir=args.work_dir,
                steps=args.md_steps,
                parallel_counts=args.parallel_md,
                sample_interval=args.sample_interval,
                start_mps=args.start_mps,
            )
        )

    if not args.skip_remd:
        remd_mps_enabled = args.start_mps and args.remd_mpi_ranks > 1
        with maybe_mps_daemon(args.work_dir / "mps_remd", remd_mps_enabled) as remd_mps_env:
            for n_analysis in args.remd_n_analysis:
                benchmark["runs"].append(
                    run_remd(
                        pdb,
                        args.work_dir / f"remd_cuda_8temps_mpi_{args.remd_mpi_ranks}_analysis_{n_analysis}",
                        args.remd_steps,
                        n_analysis,
                        args.sample_interval,
                        mpi_ranks=args.remd_mpi_ranks,
                        mps_enabled=remd_mps_enabled,
                        extra_env=remd_mps_env,
                    )
                )

    args.output.write_text(to_yaml(benchmark))
    print(f"wrote {args.output}")
    return 0


def run_single_md(pdb: Path, outdir: Path, steps: int, sample_interval: float) -> dict[str, Any]:
    command = md_command(pdb, outdir, steps)
    result, samples = run_monitored(command, outdir, sample_interval)
    return md_run_summary("single-md-cuda", result, samples, [outdir], steps, parallel_processes=1, mps_enabled=False)


def run_mps_md_series(
    *,
    pdb: Path,
    work_dir: Path,
    steps: int,
    parallel_counts: list[int],
    sample_interval: float,
    start_mps: bool,
) -> list[dict[str, Any]]:
    summaries = []
    with maybe_mps_daemon(work_dir / "mps", start_mps) as mps_env:
        for count in parallel_counts:
            outdirs = [work_dir / f"mps_md_{count}" / f"replica_{index}" for index in range(count)]
            commands = [md_command(pdb, outdir, steps) for outdir in outdirs]
            result, samples = run_parallel_monitored(commands, outdirs, sample_interval, extra_env=mps_env)
            summaries.append(md_run_summary(f"mps-md-cuda-{count}", result, samples, outdirs, steps, parallel_processes=count, mps_enabled=True))
    return summaries


def run_remd(
    pdb: Path,
    outdir: Path,
    steps: int,
    n_analysis: int,
    sample_interval: float,
    *,
    mpi_ranks: int = 1,
    mps_enabled: bool = False,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if mpi_ranks < 1:
        raise ValueError("mpi_ranks must be at least 1")
    base_command = [
        sys.executable,
        "-m",
        "cranberry.cli.main",
        "remd",
        str(pdb),
        "--steps",
        str(steps),
        "--swap-steps",
        "100",
        "--n-replicas",
        "8",
        "--t-min",
        "298",
        "--t-max",
        "600",
        "--n-analysis",
        str(n_analysis),
        "--output-dir",
        str(outdir),
        "--platform",
        "CUDA",
        "--overwrite",
    ]
    command = base_command
    env = dict(extra_env or {})
    if mpi_ranks > 1:
        mpirun = shutil.which("mpirun")
        if mpirun is None:
            raise RuntimeError("mpirun is required when --remd-mpi-ranks is greater than 1")
        command = [mpirun, "--oversubscribe", "-n", str(mpi_ranks)] + base_command
        env["OPENMMTOOLS_ENABLE_MPI"] = "1"
    result, samples = run_monitored(command, outdir, sample_interval, extra_env=env)
    online_analysis = parse_openmmtools_yaml(outdir / "output_real_time_analysis.yaml")
    summary = base_summary(f"remd-cuda-8temps-mpi-{mpi_ranks}-analysis-{n_analysis}", result, samples)
    summary.update(
        {
            "steps": steps,
            "parallel_processes": mpi_ranks,
            "mps_enabled": mps_enabled,
            "n_replicas": 8,
            "n_analysis": n_analysis,
            "args_json": read_json_like(outdir / "args.json"),
            "online_analysis": online_analysis,
        }
    )
    if online_analysis is not None:
        speed_values = online_analysis["timing_data_ns_per_day_values"]
        last_aggregate_speed = speed_values[-1] if speed_values else None
        mean_aggregate_speed = statistics.mean(speed_values) if speed_values else None
        summary.update(
            {
                "online_analysis_aggregate_speed_ns_per_day_first": speed_values[0] if speed_values else None,
                "online_analysis_aggregate_speed_ns_per_day_last": last_aggregate_speed,
                "online_analysis_aggregate_speed_ns_per_day_mean": mean_aggregate_speed,
                "online_analysis_speed_ns_per_day_first": speed_values[0] if speed_values else None,
                "online_analysis_speed_ns_per_day_last": last_aggregate_speed,
                "online_analysis_speed_ns_per_day_mean": mean_aggregate_speed,
                "online_analysis_per_runner_speed_ns_per_day_last": last_aggregate_speed / mpi_ranks if last_aggregate_speed is not None else None,
                "online_analysis_per_runner_speed_ns_per_day_mean": mean_aggregate_speed / mpi_ranks if mean_aggregate_speed is not None else None,
                "online_analysis_per_replica_speed_ns_per_day_last": last_aggregate_speed / 8 if last_aggregate_speed is not None else None,
                "online_analysis_per_replica_speed_ns_per_day_mean": mean_aggregate_speed / 8 if mean_aggregate_speed is not None else None,
            }
        )
    return summary


def md_command(pdb: Path, outdir: Path, steps: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "cranberry.cli.main",
        "md",
        str(pdb),
        "--steps",
        str(steps),
        "--n-record",
        "20",
        "--checkpoint-interval",
        str(max(1, steps)),
        "--output-dir",
        str(outdir),
        "--platform",
        "CUDA",
    ]


def run_monitored(command: list[str], outdir: Path, sample_interval: float, extra_env: dict[str, str] | None = None) -> tuple[CommandResult, list[GpuSample]]:
    outdir.mkdir(parents=True, exist_ok=True)
    stdout_path = outdir / "stdout.txt"
    stderr_path = outdir / "stderr.txt"
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    monitor = GpuMonitor(sample_interval)
    monitor.start()
    start = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(command, cwd=Path.cwd(), env=env, stdout=stdout, stderr=stderr, text=True, check=False)
    wall_seconds = time.perf_counter() - start
    samples = monitor.stop()
    return CommandResult(command, completed.returncode, wall_seconds, str(stdout_path), str(stderr_path)), samples


def run_parallel_monitored(commands: list[list[str]], outdirs: list[Path], sample_interval: float, extra_env: dict[str, str] | None = None) -> tuple[CommandResult, list[GpuSample]]:
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    monitor = GpuMonitor(sample_interval)
    monitor.start()
    start = time.perf_counter()
    processes = []
    for command, outdir in zip(commands, outdirs, strict=True):
        stdout = (outdir / "stdout.txt").open("w")
        stderr = (outdir / "stderr.txt").open("w")
        process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=stdout, stderr=stderr, text=True)
        processes.append((process, stdout, stderr))
    returncodes = []
    try:
        for process, stdout, stderr in processes:
            returncodes.append(process.wait())
            stdout.close()
            stderr.close()
    finally:
        for process, stdout, stderr in processes:
            if process.poll() is None:
                process.terminate()
            stdout.close()
            stderr.close()
    wall_seconds = time.perf_counter() - start
    samples = monitor.stop()
    result = CommandResult(
        command=["parallel"] + [str(len(commands))] + commands[0],
        returncode=max(returncodes) if returncodes else 1,
        wall_seconds=wall_seconds,
        stdout_path=str(outdirs[0].parent / "replica_*/stdout.txt"),
        stderr_path=str(outdirs[0].parent / "replica_*/stderr.txt"),
    )
    return result, samples


def md_run_summary(kind: str, result: CommandResult, samples: list[GpuSample], outdirs: list[Path], steps: int, *, parallel_processes: int, mps_enabled: bool) -> dict[str, Any]:
    speeds = [parse_md_log_speed(outdir / "log") for outdir in outdirs]
    speed_values = [speed for speed in speeds if speed is not None]
    wall_ns_per_day_per_process = steps * 5.0e-6 / result.wall_seconds * 86400.0
    summary = base_summary(kind, result, samples)
    summary.update(
        {
            "steps_per_process": steps,
            "parallel_processes": parallel_processes,
            "mps_enabled": mps_enabled,
            "wall_ns_per_day_per_process": wall_ns_per_day_per_process,
            "aggregate_wall_ns_per_day": wall_ns_per_day_per_process * parallel_processes,
            "log_speed_ns_per_day_per_runner_mean": statistics.mean(speed_values) if speed_values else None,
            "log_speed_ns_per_day_aggregate": sum(speed_values) if speed_values else None,
            "log_speed_ns_per_day_mean": statistics.mean(speed_values) if speed_values else None,
            "log_speed_ns_per_day_values": speed_values,
        }
    )
    return summary


def base_summary(kind: str, result: CommandResult, samples: list[GpuSample]) -> dict[str, Any]:
    util = [sample.utilization_percent for sample in samples]
    memory = [sample.memory_mib for sample in samples]
    return {
        "kind": kind,
        "returncode": result.returncode,
        "command": result.command,
        "wall_seconds": result.wall_seconds,
        "stdout_path": result.stdout_path,
        "stderr_path": result.stderr_path,
        "gpu_samples": len(samples),
        "gpu_utilization_percent_mean": statistics.mean(util) if util else None,
        "gpu_utilization_percent_max": max(util) if util else None,
        "gpu_memory_mib_max": max(memory) if memory else None,
    }


class GpuMonitor:
    def __init__(self, interval: float):
        self.interval = interval
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[GpuSample]:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 2))
        return self.samples

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = query_gpu_sample()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval)


def query_gpu_sample() -> GpuSample | None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    row = next(csv.reader([completed.stdout.strip().splitlines()[0]]))
    power = parse_float(row[2]) if len(row) > 2 else None
    return GpuSample(time.time(), parse_float(row[0]) or 0.0, parse_float(row[1]) or 0.0, power)


def parse_md_log_speed(path: Path) -> float | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    header = next(csv.reader([lines[0].lstrip("#")]))
    values = next(csv.reader([lines[-1]]))
    for index, column in enumerate(header):
        if column.startswith("Speed"):
            return parse_float(values[index])
    return None


def parse_openmmtools_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    iterations = []
    ns_per_day_values = []
    average_seconds_per_iteration_values = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- iteration:"):
            value = stripped.split(":", 1)[1].strip()
            parsed = parse_float(value)
            if parsed is not None:
                iterations.append(int(parsed))
        elif stripped.startswith("ns_per_day:"):
            parsed = parse_float(stripped.split(":", 1)[1].strip())
            if parsed is not None:
                ns_per_day_values.append(parsed)
        elif stripped.startswith("average_seconds_per_iteration:"):
            parsed = parse_float(stripped.split(":", 1)[1].strip())
            if parsed is not None:
                average_seconds_per_iteration_values.append(parsed)
    return {
        "path": str(path),
        "iteration_records": len(iterations),
        "last_iteration": iterations[-1] if iterations else None,
        "timing_data_ns_per_day_values": ns_per_day_values,
        "timing_data_average_seconds_per_iteration_values": average_seconds_per_iteration_values,
    }


def read_json_like(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text())


class maybe_mps_daemon:
    def __init__(self, directory: Path, enabled: bool):
        self.directory = directory
        self.enabled = enabled
        self.env: dict[str, str] = {}

    def __enter__(self) -> dict[str, str]:
        if not self.enabled:
            return {}
        self.directory.mkdir(parents=True, exist_ok=True)
        runtime_dir = Path(tempfile.mkdtemp(prefix="cranberry-mps-", dir="/tmp"))
        pipe_dir = runtime_dir / "pipe"
        log_dir = runtime_dir / "log"
        pipe_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.env = {
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(log_dir),
        }
        env = os.environ.copy()
        env.update(self.env)
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
        time.sleep(1.0)
        return self.env

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.enabled:
            return
        subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True, env={**os.environ, **self.env}, check=False)


def nvidia_smi_summary() -> dict[str, Any]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {}
    row = next(csv.reader([completed.stdout.strip().splitlines()[0]]))
    return {
        "name": row[0].strip() if row else None,
        "driver_version": row[1].strip() if len(row) > 1 else None,
        "memory_total_mib": parse_float(row[2]) if len(row) > 2 else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def hostname() -> str:
    return subprocess.run(["hostname"], text=True, capture_output=True, check=False).stdout.strip()


def parse_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def to_yaml(value: Any, indent: int = 0) -> str:
    lines = _yaml_lines(value, indent)
    return "\n".join(lines) + "\n"


def _yaml_lines(value: Any, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in ":#[]{}&,*?|-<>=!%@`\\n"):
        return repr(text)
    return text


if __name__ == "__main__":
    raise SystemExit(main())
