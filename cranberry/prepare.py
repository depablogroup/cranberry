from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from openmm import Vec3, app, unit

from cranberry.pdbio import write_pdb_with_conect
from cranberry.validation import ValidationResult, validate_canonical_pdb

TERMINAL_PHOSPHATE_DISTANCE_NM = 0.45
TERMINAL_PHOSPHATE_S3_S2_ANGLE_RAD = 2.356
TERMINAL_PHOSPHATE_S3_NEXT_P_ANGLE_RAD = 1.920


@dataclass(frozen=True)
class PreparedStructureResult:
    input_path: Path
    output_path: Path | None
    input_validation: ValidationResult
    output_validation: ValidationResult
    add_terminal_phosphate: bool
    inserted_terminal_phosphates: int


def prepare_structure(
    pdb_path: str | Path,
    *,
    output_path: str | Path | None = None,
    add_terminal_phosphate: bool = False,
    overwrite: bool = True,
) -> PreparedStructureResult:
    pdb_path = Path(pdb_path)
    input_validation = validate_canonical_pdb(pdb_path)
    input_validation.raise_for_errors()

    if not add_terminal_phosphate:
        return PreparedStructureResult(
            input_path=pdb_path,
            output_path=None,
            input_validation=input_validation,
            output_validation=input_validation,
            add_terminal_phosphate=False,
            inserted_terminal_phosphates=0,
        )

    if output_path is None:
        output_path = pdb_path.with_name(f"{pdb_path.stem}_prepared.pdb")
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdb = app.PDBFile(str(pdb_path))
    topology, positions, inserted_terminal_phosphates = _insert_terminal_phosphates(pdb.topology, pdb.positions)
    write_pdb_with_conect(output_path, topology, positions)

    output_validation = validate_canonical_pdb(output_path)
    output_validation.raise_for_errors()
    return PreparedStructureResult(
        input_path=pdb_path,
        output_path=output_path,
        input_validation=input_validation,
        output_validation=output_validation,
        add_terminal_phosphate=True,
        inserted_terminal_phosphates=inserted_terminal_phosphates,
    )


def _insert_terminal_phosphates(topology: app.Topology, positions):
    old_positions = positions.value_in_unit(unit.nanometer)
    new_positions = []
    new_topology = app.Topology()
    box_vectors = topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        new_topology.setPeriodicBoxVectors(box_vectors)

    old_to_new = {}
    inserted_terminal_phosphates = 0

    for chain in topology.chains():
        new_chain = new_topology.addChain(chain.id)
        residues = list(chain.residues())
        for residue_index, residue in enumerate(residues):
            new_residue = new_topology.addResidue(residue.name, new_chain, residue.id)
            atoms = list(residue.atoms())
            first_residue = residue_index == 0
            inserted_p_atom = None
            if first_residue and not any(atom.name == "P" for atom in atoms):
                s2_atom = _require_atom(residue, "S2")
                s3_atom = _require_atom(residue, "S3")
                next_p_atom = _find_next_residue_phosphate(residues, residue_index)
                next_p_position = None if next_p_atom is None else old_positions[next_p_atom.index]
                p_position = _terminal_phosphate_position(
                    old_positions[s2_atom.index],
                    old_positions[s3_atom.index],
                    next_p_position,
                )
                inserted_p_atom = new_topology.addAtom("P", app.element.phosphorus, new_residue)
                new_positions.append(p_position)
                inserted_terminal_phosphates += 1
            s3_new_atom = None
            for atom in atoms:
                new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
                old_to_new[atom] = new_atom
                if atom.name == "S3":
                    s3_new_atom = new_atom
                new_positions.append(old_positions[atom.index])
            if inserted_p_atom is not None:
                if s3_new_atom is None:
                    raise ValueError(f"Residue {residue.index} is missing S3 and cannot receive a terminal phosphate")
                new_topology.addBond(inserted_p_atom, s3_new_atom)

    for bond in topology.bonds():
        new_topology.addBond(old_to_new[bond.atom1], old_to_new[bond.atom2])

    return new_topology, unit.Quantity(new_positions, unit.nanometer), inserted_terminal_phosphates


def _terminal_phosphate_position(s2_position, s3_position, next_p_position=None):
    s3_to_s2 = _unit_vector(s2_position - s3_position)
    if next_p_position is None:
        direction = _fallback_direction(s3_to_s2)
    else:
        s3_to_next_p = _unit_vector(next_p_position - s3_position)
        direction = _solve_two_angle_direction(
            s3_to_s2,
            TERMINAL_PHOSPHATE_S3_S2_ANGLE_RAD,
            s3_to_next_p,
            TERMINAL_PHOSPHATE_S3_NEXT_P_ANGLE_RAD,
        )
    return s3_position + direction * TERMINAL_PHOSPHATE_DISTANCE_NM


def _solve_two_angle_direction(axis_a: Vec3, angle_a: float, axis_b: Vec3, angle_b: float) -> Vec3:
    basis_a = _unit_vector(axis_a)
    basis_b = _unit_vector(axis_b)
    ab_dot = _dot(basis_a, basis_b)
    in_plane = basis_b - basis_a * ab_dot
    in_plane_norm = _norm(in_plane)
    if in_plane_norm < 1e-8:
        return _fallback_direction(basis_a)

    basis_c = in_plane / in_plane_norm
    basis_n = _unit_vector(_cross(basis_a, basis_c))

    x = math.cos(angle_a)
    y = (math.cos(angle_b) - x * ab_dot) / in_plane_norm
    remaining = 1.0 - x * x - y * y
    if remaining < -1e-8:
        return _fallback_direction(basis_a)
    z = math.sqrt(max(0.0, remaining))

    candidate_plus = basis_a * x + basis_c * y + basis_n * z
    candidate_minus = basis_a * x + basis_c * y - basis_n * z
    if candidate_plus.z >= candidate_minus.z:
        return _unit_vector(candidate_plus)
    return _unit_vector(candidate_minus)


def _fallback_direction(s3_to_s2: Vec3) -> Vec3:
    perpendicular = _perpendicular_unit_vector(s3_to_s2)
    direction = (
        s3_to_s2 * math.cos(TERMINAL_PHOSPHATE_S3_S2_ANGLE_RAD)
        + perpendicular * math.sin(TERMINAL_PHOSPHATE_S3_S2_ANGLE_RAD)
    )
    return _unit_vector(direction)


def _unit_vector(vector: Vec3) -> Vec3:
    length = _norm(vector)
    if length == 0:
        return Vec3(1.0, 0.0, 0.0)
    return vector / length


def _norm(vector: Vec3) -> float:
    return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)


def _perpendicular_unit_vector(vector: Vec3) -> Vec3:
    reference = Vec3(1.0, 0.0, 0.0)
    if abs(_dot(vector, reference)) > 0.9:
        reference = Vec3(0.0, 1.0, 0.0)
    perpendicular = _cross(vector, reference)
    if _norm(perpendicular) == 0:
        perpendicular = _cross(vector, Vec3(0.0, 0.0, 1.0))
    return _unit_vector(perpendicular)


def _dot(left: Vec3, right: Vec3) -> float:
    return left.x * right.x + left.y * right.y + left.z * right.z


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return Vec3(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    )


def _find_next_residue_phosphate(residues: list[app.Residue], residue_index: int):
    for next_residue in residues[residue_index + 1:]:
        for atom in next_residue.atoms():
            if atom.name == "P":
                return atom
    return None


def _require_atom(residue: app.Residue, name: str):
    for atom in residue.atoms():
        if atom.name == name:
            return atom
    raise ValueError(f"Residue {residue.index} is missing required bead {name!r}")
