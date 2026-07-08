#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from openmm import app, unit

from cranberry import __version__
from cranberry.data import data_path
from cranberry.md import create_simulation

SYSTEMS = {
    "157d": "examples/157d_cg_vs_conect.pdb",
    "1l2x": "examples/1l2x_cg_vs_conect.pdb",
    "2ntCG": "examples/2ntCG_cg_vs_conect.pdb",
    "1zih": "examples/1zih_cg_vs_conect.pdb",
}
RESULTS_DIR = Path("benchmarks/results")
DOCS_DIR = Path("docs/benchmarks")
CURRENT_MD = DOCS_DIR / "current.md"
CURRENT_SVG = DOCS_DIR / "current.svg"
INDEX_MD = DOCS_DIR / "index.md"


@dataclass(frozen=True)
class BenchmarkRow:
    system_id: str
    pdb_path: str
    nucleotides: int
    atoms: int
    wall_seconds: float
    steps: int
    timestep_fs: float
    ns_per_day: float
    seconds_per_step: float


@dataclass(frozen=True)
class BenchmarkSnapshot:
    schema_version: int
    benchmark_kind: str
    generated_at: str
    hostname: str
    platform: str
    mps_enabled: bool
    series_name: str
    steps: int
    warmup_steps: int
    timestep_fs: float
    model: str
    temperature_kelvin: float
    salt_millimolar: float
    cranberry_version: str
    openmm_version: str | None
    gpu_name: str | None
    driver_version: str | None
    cuda_visible_devices: str | None
    rows: list[BenchmarkRow]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CRANBERRY MD benchmarks and publish docs snapshots.")
    parser.add_argument("--platform", required=True, help="OpenMM platform name, for example CUDA or CPU")
    parser.add_argument("--series-name", default=None, help="Label for this benchmark series")
    parser.add_argument("--steps", type=int, default=1000, help="Timed MD steps per system")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Untimed warm-up steps per system")
    parser.add_argument("--timestep-fs", type=float, default=10.0, help="MD timestep in femtoseconds")
    parser.add_argument("--temperature-kelvin", type=float, default=298.0, help="System temperature in kelvin")
    parser.add_argument("--salt-millimolar", type=float, default=150.0, help="Salt concentration in millimolar")
    parser.add_argument("--model", default="default", help="CRANBERRY model name")
    parser.add_argument("--results-json", type=Path, default=None, help="Path for the raw JSON snapshot")
    parser.add_argument("--mps-enabled", action="store_true", help="Mark the run as MPS-enabled in the metadata")
    parser.add_argument("--publish-only", action="store_true", help="Skip running simulations and just regenerate docs from existing JSON snapshots")
    parser.add_argument("systems", nargs="*", default=list(SYSTEMS), help="System IDs to benchmark; defaults to the bundled four fixtures")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.publish_only:
        platform = args.platform
        series_name = args.series_name or default_series_name(platform, args.mps_enabled)
        results_json = args.results_json or default_results_json(series_name, platform, args.mps_enabled)
        rows = []
        for system_id in args.systems:
            if system_id not in SYSTEMS:
                raise SystemExit(f"Unknown benchmark system: {system_id}")
            rows.append(
                benchmark_system(
                    system_id=system_id,
                    pdb_relative_path=SYSTEMS[system_id],
                    platform=platform,
                    steps=args.steps,
                    warmup_steps=args.warmup_steps,
                    timestep_fs=args.timestep_fs,
                    temperature_kelvin=args.temperature_kelvin,
                    salt_millimolar=args.salt_millimolar,
                    model=args.model,
                )
            )
        rows.sort(key=lambda row: (row.nucleotides, row.system_id))
        snapshot = BenchmarkSnapshot(
            schema_version=1,
            benchmark_kind="md-single-system-suite",
            generated_at=datetime.now(timezone.utc).isoformat(),
            hostname=socket.gethostname(),
            platform=platform,
            mps_enabled=bool(args.mps_enabled),
            series_name=series_name,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            timestep_fs=args.timestep_fs,
            model=args.model,
            temperature_kelvin=args.temperature_kelvin,
            salt_millimolar=args.salt_millimolar,
            cranberry_version=__version__,
            openmm_version=openmm_version(),
            gpu_name=gpu_name() if platform.upper() == "CUDA" else None,
            driver_version=driver_version() if platform.upper() == "CUDA" else None,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            rows=rows,
        )
        write_snapshot(results_json, snapshot)
        print(f"wrote {results_json}")

    publish_docs()
    print(f"wrote {INDEX_MD}")
    if CURRENT_MD.exists():
        print(f"wrote {CURRENT_MD}")
    if CURRENT_SVG.exists():
        print(f"wrote {CURRENT_SVG}")
    return 0


def benchmark_system(*, system_id: str, pdb_relative_path: str, platform: str, steps: int, warmup_steps: int, timestep_fs: float, temperature_kelvin: float, salt_millimolar: float, model: str) -> BenchmarkRow:
    pdb_path = data_path(pdb_relative_path)
    pdb = app.PDBFile(str(pdb_path))
    residue_count = sum(1 for _ in pdb.topology.residues())
    atom_count = pdb.topology.getNumAtoms()
    simulation = create_simulation(
        pdb_path,
        model=model,
        temperature=temperature_kelvin * unit.kelvin,
        salt_concentration=salt_millimolar * unit.millimolar,
        timestep=timestep_fs * unit.femtosecond,
        platform=platform,
    )
    if warmup_steps:
        simulation.step(warmup_steps)
    start = time.perf_counter()
    simulation.step(steps)
    wall_seconds = time.perf_counter() - start
    simulated_ns = steps * timestep_fs * 1e-6
    ns_per_day = simulated_ns / wall_seconds * 86400.0 if wall_seconds > 0 else math.inf
    return BenchmarkRow(
        system_id=system_id,
        pdb_path=str(pdb_path),
        nucleotides=residue_count,
        atoms=atom_count,
        wall_seconds=wall_seconds,
        steps=steps,
        timestep_fs=timestep_fs,
        ns_per_day=ns_per_day,
        seconds_per_step=wall_seconds / steps,
    )


def write_snapshot(path: Path, snapshot: BenchmarkSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot_to_dict(snapshot), indent=2, sort_keys=True) + "\n")


def publish_docs() -> None:
    snapshots = load_snapshots()
    INDEX_MD.write_text(render_index(snapshots))
    latest = latest_snapshot(snapshots)
    if latest is None:
        CURRENT_MD.write_text(render_empty_current())
        if CURRENT_SVG.exists():
            CURRENT_SVG.unlink()
        return
    CURRENT_MD.write_text(render_current(latest, CURRENT_SVG.name))
    CURRENT_SVG.write_text(render_svg(latest))


def load_snapshots() -> list[tuple[Path, BenchmarkSnapshot]]:
    snapshots: list[tuple[Path, BenchmarkSnapshot]] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        rows = [BenchmarkRow(**row) for row in payload["rows"]]
        payload["rows"] = rows
        snapshots.append((path, BenchmarkSnapshot(**payload)))
    snapshots.sort(key=lambda item: item[1].generated_at, reverse=True)
    return snapshots


def latest_snapshot(snapshots: list[tuple[Path, BenchmarkSnapshot]]) -> tuple[Path, BenchmarkSnapshot] | None:
    return snapshots[0] if snapshots else None


def render_index(snapshots: list[tuple[Path, BenchmarkSnapshot]]) -> str:
    lines = [
        "# Benchmarks",
        "",
        "Benchmarks are separate from tests. Continuous integration should only run tiny benchmark smoke checks; full CPU/GPU benchmarks should be manual or scheduled and published here as snapshots.",
        "",
        "The benchmark runner writes JSON snapshots under `benchmarks/results/` and regenerates this docs section. That means a new benchmark on another machine can update the published plots by committing the new JSON plus the generated benchmark docs artifacts.",
        "",
        "```{toctree}",
        ":maxdepth: 1",
        "",
        "current",
        "```",
        "",
        "## Available snapshots",
        "",
    ]
    if not snapshots:
        lines.append("No benchmark snapshots have been published yet.")
        lines.append("")
    else:
        lines.extend([
            "| Series | Generated | Host | Platform | MPS | GPU | Raw JSON |",
            "| --- | --- | --- | --- | ---: | --- | --- |",
        ])
        for path, snapshot in snapshots:
            lines.append(
                f"| `{snapshot.series_name}` | `{snapshot.generated_at}` | `{snapshot.hostname}` | `{snapshot.platform}` | `{snapshot.mps_enabled}` | `{snapshot.gpu_name or 'n/a'}` | `{path.name}` |"
            )
        lines.append("")
    lines.extend([
        "## Planned expansion",
        "",
        "- Add explicit CPU, CUDA, and later MPS series side by side for the same four canonical systems.",
        "- Add multi-process MPS demonstrations later as separate benchmark kinds rather than mixing them into the first MD baseline.",
        "- Add REMD benchmark series after the REMD workflow exists.",
    ])
    return "\n".join(lines) + "\n"


def render_empty_current() -> str:
    return """# CRANBERRY Benchmark Snapshot\n\nNo benchmark snapshots have been published yet. Run `benchmarks/benchmark_md.py` to generate the first baseline.\n"""


def render_current(item: tuple[Path, BenchmarkSnapshot], plot_name: str) -> str:
    path, snapshot = item
    lines = [
        "# CRANBERRY Benchmark Snapshot",
        "",
        "This page is generated from the latest published benchmark JSON snapshot. The x-axis is system size measured in nucleotides, and the y-axis is MD throughput in ns/day.",
        "",
        f"- Source JSON: `{path.name}`",
        f"- Benchmark kind: `{snapshot.benchmark_kind}`",
        f"- Generated at: `{snapshot.generated_at}`",
        f"- Host: `{snapshot.hostname}`",
        f"- Platform: `{snapshot.platform}`",
        f"- MPS enabled: `{snapshot.mps_enabled}`",
        f"- Series: `{snapshot.series_name}`",
        f"- Timed steps per system: `{snapshot.steps}`",
        f"- Warm-up steps per system: `{snapshot.warmup_steps}`",
        f"- Timestep: `{snapshot.timestep_fs} fs`",
        f"- Model: `{snapshot.model}`",
        f"- Temperature: `{snapshot.temperature_kelvin} K`",
        f"- Salt: `{snapshot.salt_millimolar} mM`",
        f"- Cranberry: `{snapshot.cranberry_version}`",
        f"- OpenMM: `{snapshot.openmm_version}`",
        f"- GPU: `{snapshot.gpu_name or 'n/a'}`",
        f"- Driver: `{snapshot.driver_version or 'n/a'}`",
        f"- CUDA_VISIBLE_DEVICES: `{snapshot.cuda_visible_devices or 'n/a'}`",
        "",
        f"![Speed vs system size]({plot_name})",
        "",
        "## Results",
        "",
        "| System | Nucleotides | Atoms | Wall seconds | ns/day |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in snapshot.rows:
        lines.append(
            f"| `{row.system_id}` | {row.nucleotides} | {row.atoms} | {row.wall_seconds:.3f} | {row.ns_per_day:.2f} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "This first published slice is MD for one GPU and one process. Multi-process MPS comparisons and REMD should be added later as distinct benchmark series.",
    ])
    return "\n".join(lines) + "\n"


def render_svg(item: tuple[Path, BenchmarkSnapshot]) -> str:
    _, snapshot = item
    width, height = 900, 520
    left, right, top, bottom = 100, 30, 40, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    xs = [row.nucleotides for row in snapshot.rows]
    ys = [row.ns_per_day for row in snapshot.rows]
    x_min = min(xs)
    x_max = max(xs)
    y_min = 0.0
    y_max = max(ys)
    if x_max == x_min:
        x_max += 1
    if y_max == y_min:
        y_max = 1.0
    x_pad = max(1.0, (x_max - x_min) * 0.1)
    y_pad = max(1.0, y_max * 0.12)
    x_min -= x_pad
    x_max += x_pad
    y_max += y_pad

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_pos(value: float) -> float:
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    grid_lines = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_height * frac
        label = y_max * (1.0 - frac)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d9e0e7" stroke-width="1" />\n'
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#5b6670">{label:.1f}</text>'
        )

    x_ticks = []
    for row in snapshot.rows:
        x = x_pos(row.nucleotides)
        x_ticks.append(
            f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#18222c" stroke-width="1" />\n'
            f'<text x="{x:.1f}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12" fill="#5b6670">{row.nucleotides}</text>'
        )

    palette = ["#0b66c3", "#137348", "#9a5b00", "#7a3e9d"]
    points = []
    for index, row in enumerate(snapshot.rows):
        x = x_pos(row.nucleotides)
        y = y_pos(row.ns_per_day)
        color = palette[index % len(palette)]
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" />\n'
            f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" font-size="12" fill="#18222c">{escape(row.system_id)} {row.ns_per_day:.1f}</text>'
        )

    title = escape(f"CRANBERRY MD throughput: {snapshot.platform}{' + MPS' if snapshot.mps_enabled else ''}")
    subtitle = escape(f"{snapshot.hostname} | {snapshot.gpu_name or 'n/a'} | OpenMM {snapshot.openmm_version or 'unknown'}")
    return f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-labelledby=\"title desc\">\n  <title id=\"title\">{title}</title>\n  <desc id=\"desc\">Scatter plot of MD throughput in ns/day versus nucleotides across the bundled canonical benchmark systems.</desc>\n  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\" />\n  <text x=\"{left}\" y=\"24\" font-size=\"18\" font-weight=\"700\" fill=\"#18222c\">{title}</text>\n  <text x=\"{left}\" y=\"{height - 18}\" font-size=\"12\" fill=\"#5b6670\">x-axis: nucleotides</text>\n  <text x=\"18\" y=\"{top + plot_height / 2:.1f}\" transform=\"rotate(-90 18 {top + plot_height / 2:.1f})\" font-size=\"12\" fill=\"#5b6670\">y-axis: ns/day</text>\n  <text x=\"{left}\" y=\"{height - 4}\" font-size=\"11\" fill=\"#5b6670\">{subtitle}</text>\n  <g>\n    {''.join(grid_lines)}\n  </g>\n  <line x1=\"{left}\" y1=\"{top + plot_height}\" x2=\"{width - right}\" y2=\"{top + plot_height}\" stroke=\"#18222c\" stroke-width=\"1.5\" />\n  <line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{top + plot_height}\" stroke=\"#18222c\" stroke-width=\"1.5\" />\n  <g>\n    {''.join(x_ticks)}\n  </g>\n  <g>\n    {''.join(points)}\n  </g>\n</svg>\n"""


def snapshot_to_dict(snapshot: BenchmarkSnapshot) -> dict:
    payload = asdict(snapshot)
    payload["rows"] = [asdict(row) for row in snapshot.rows]
    return payload


def default_series_name(platform: str, mps_enabled: bool) -> str:
    host = socket.gethostname().split(".")[0]
    gpu = gpu_name() if platform.upper() == "CUDA" else None
    gpu_label = normalize_label(gpu) if gpu else platform.lower()
    suffix = "-mps" if mps_enabled else ""
    return f"{host}-{gpu_label}{suffix}"


def default_results_json(series_name: str, platform: str, mps_enabled: bool) -> Path:
    suffix = "-mps" if mps_enabled else ""
    safe_platform = platform.lower()
    return RESULTS_DIR / f"{series_name}-{safe_platform}{suffix}.json"


def normalize_label(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def openmm_version() -> str | None:
    import openmm as mm
    return getattr(mm, "__version__", None)


def gpu_name() -> str | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def driver_version() -> str | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
