"""Standalone OpenMM reproducer for collinear-dihedral force NaNs.

This script intentionally does not import Cranberry.  It constructs a
four-particle OpenMM system and evaluates custom expressions containing a
singular dihedral at exactly collinear coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import openmm as mm
from openmm import unit


EXPRESSIONS = {
    "direct_cos_dihedral": {
        "compound": "cos(dihedral(p1,p2,p3,p4))",
        "hbond": "cos(dihedral(d2,d1,a1,a2))",
    },
    "unselected_select_branch": {
        "compound": (
            "select(condition, cos(dihedral(p1,p2,p3,p4)), 0); "
            "condition=distance(p1,p2)-distance(p1,p2)"
        ),
        "hbond": (
            "select(condition, cos(dihedral(d2,d1,a1,a2)), 0); "
            "condition=distance(d1,a1)-distance(d1,a1)"
        ),
    },
    "cranberry_shaped_orientation": {
        "compound": (
            "select(sin(A)*sin(B), full, partial); "
            "full=sin(A)*sin(B)*cos(PHI)-cos(A)*cos(B); "
            "partial=-cos(A)*cos(B); "
            "A=angle(p1,p2,p3); B=angle(p2,p3,p4); "
            "PHI=dihedral(p1,p2,p3,p4)"
        ),
        "hbond": (
            "select(sin(A)*sin(B), full, partial); "
            "full=sin(A)*sin(B)*cos(PHI)-cos(A)*cos(B); "
            "partial=-cos(A)*cos(B); "
            "A=angle(d2,d1,a1); B=angle(d1,a1,a2); "
            "PHI=dihedral(d2,d1,a1,a2)"
        ),
    },
    "distance_dot_orientation": {
        "compound": (
            "(distance(p1,p3)^2+distance(p2,p4)^2-distance(p1,p4)^2-distance(p2,p3)^2)"
            "/(2*distance(p1,p2)*distance(p3,p4))"
        ),
        "hbond": (
            "(distance(d2,a1)^2+distance(d1,a2)^2-distance(d2,a2)^2-distance(d1,a1)^2)"
            "/(2*distance(d1,d2)*distance(a1,a2))"
        ),
    },
}


@dataclass(frozen=True)
class Result:
    platform: str
    force_type: str
    expression: str
    offset_nm: float
    energy: str
    forces_finite: str
    max_force: str
    minimize: str


def positions(offset_nm: float) -> unit.Quantity:
    """Return four coordinates that are collinear when offset_nm is zero."""
    coords = np.array(
        [
            [0.0, offset_nm, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, offset_nm],
        ],
        dtype=float,
    )
    return unit.Quantity(coords, unit.nanometer)


def build_system(force_type: str, expression: str) -> mm.System:
    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0 * unit.dalton)

    if force_type == "compound":
        force = mm.CustomCompoundBondForce(4, expression)
        force.addBond([0, 1, 2, 3], [])
    elif force_type == "hbond":
        force = mm.CustomHbondForce(expression)
        force.addDonor(1, 0, -1, [])
        force.addAcceptor(2, 3, -1, [])
    else:
        raise ValueError(f"Unknown force type: {force_type}")

    system.addForce(force)
    return system


def finite_forces(force_quantity: unit.Quantity) -> tuple[bool, float]:
    forces = force_quantity.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    finite = np.isfinite(forces)
    abs_forces = np.abs(forces)
    if np.isinf(abs_forces).any():
        max_force = math.inf
    elif np.isfinite(abs_forces).any():
        max_force = float(abs_forces[np.isfinite(abs_forces)].max())
    else:
        max_force = math.nan
    return bool(finite.all()), max_force


def evaluate(
    platform_name: str,
    force_type: str,
    expression_name: str,
    expression: str,
    offset_nm: float,
) -> Result:
    platform = mm.Platform.getPlatformByName(platform_name)
    system = build_system(force_type, expression)
    context = mm.Context(system, mm.VerletIntegrator(0.001 * unit.picoseconds), platform)
    context.setPositions(positions(offset_nm))

    try:
        energy_value = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        energy = f"{energy_value:.17g}" if math.isfinite(energy_value) else str(energy_value)
    except Exception as exc:  # pragma: no cover - this is diagnostic output.
        energy = f"ERROR: {type(exc).__name__}: {exc}"

    try:
        forces = context.getState(getForces=True).getForces(asNumpy=True)
        all_finite, max_force = finite_forces(forces)
        forces_finite = "yes" if all_finite else "no"
        max_force_text = f"{max_force:.17g}" if math.isfinite(max_force) else str(max_force)
    except Exception as exc:  # pragma: no cover - this is diagnostic output.
        forces_finite = f"ERROR: {type(exc).__name__}: {exc}"
        max_force_text = "n/a"

    try:
        mm.LocalEnergyMinimizer.minimize(context, maxIterations=1)
        minimize = "pass"
    except Exception as exc:  # pragma: no cover - this is diagnostic output.
        minimize = f"FAIL: {type(exc).__name__}: {exc}"

    del context
    return Result(
        platform=platform_name,
        force_type=force_type,
        expression=expression_name,
        offset_nm=offset_nm,
        energy=energy,
        forces_finite=forces_finite,
        max_force=max_force_text,
        minimize=minimize,
    )


def print_table(results: list[Result]) -> None:
    headers = [
        "platform",
        "force_type",
        "expression",
        "offset_nm",
        "energy",
        "forces_finite",
        "max_force",
        "minimize",
    ]
    rows = [
        [
            result.platform,
            result.force_type,
            result.expression,
            f"{result.offset_nm:.0e}",
            result.energy,
            result.forces_finite,
            result.max_force,
            result.minimize,
        ]
        for result in results
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    print(f"OpenMM {mm.version.version}")
    platforms = [mm.Platform.getPlatform(index).getName() for index in range(mm.Platform.getNumPlatforms())]
    print("Platforms:", ", ".join(platforms))

    offsets = [0.0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6]
    results = [
        evaluate(platform, force_type, name, expressions[force_type], offset)
        for platform in platforms
        for force_type in ["compound", "hbond"]
        for name, expressions in EXPRESSIONS.items()
        for offset in offsets
    ]
    print()
    print_table(results)


if __name__ == "__main__":
    main()
