from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openmm import Vec3, app, unit

from cranberry.pdbio import write_pdb_with_conect
from cranberry.validation import ValidationResult, validate_canonical_pdb

_CANONICAL_RESIDUE_NAME = {"rA": "A", "rU": "U", "rG": "G", "rC": "C"}

_OUTPUT_ATOM_ORDER = {
    "A": ("P", "S3", "S2", "R1", "A1", "A2", "BC", "BN"),
    "U": ("P", "S3", "S2", "Y1", "U1", "Y2", "BC", "BN"),
    "G": ("P", "S3", "S2", "R1", "G1", "G2", "BC", "BN"),
    "C": ("P", "S3", "S2", "Y1", "C1", "Y2", "BC", "BN"),
}

_SOURCE_ATOMS = {
    "A": {"P": "P", "S3": "C3'", "S2": "C2'", "R1": "C8", "A1": "N6", "A2": "C2"},
    "U": {"P": "P", "S3": "C3'", "S2": "C2'", "Y1": "C6", "U1": "O4", "Y2": "O2"},
    "G": {"P": "P", "S3": "C3'", "S2": "C2'", "R1": "C8", "G1": "O6", "G2": "N2"},
    "C": {"P": "P", "S3": "C3'", "S2": "C2'", "Y1": "C6", "C1": "N4", "Y2": "O2"},
}

_OUTPUT_ELEMENTS = {
    "P": app.element.phosphorus,
    "S3": app.element.carbon,
    "S2": app.element.carbon,
    "R1": app.element.carbon,
    "A1": app.element.nitrogen,
    "A2": app.element.carbon,
    "G1": app.element.oxygen,
    "G2": app.element.nitrogen,
    "Y1": app.element.carbon,
    "Y2": app.element.hydrogen,
    "U1": app.element.oxygen,
    "C1": app.element.nitrogen,
    "BC": app.element.helium,
    "BN": app.element.neon,
}


@dataclass(frozen=True)
class CoarseGrainResult:
    input_path: Path
    output_path: Path
    output_validation: ValidationResult
    residue_count: int
    inserted_virtual_sites: int
    missing_terminal_phosphates: int


def coarse_grain_structure(
    pdb_path: str | Path,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = True,
) -> CoarseGrainResult:
    pdb_path = Path(pdb_path)
    pdb = app.PDBFile(str(pdb_path))

    if output_path is None:
        output_path = pdb_path.with_name(f"{pdb_path.stem}_cg_vs_conect.pdb")
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topology, positions, residue_count, inserted_virtual_sites, missing_terminal_phosphates = _build_cg_topology(
        pdb.topology,
        pdb.positions,
    )
    write_pdb_with_conect(output_path, topology, positions)

    output_validation = validate_canonical_pdb(output_path)
    output_validation.raise_for_errors()
    return CoarseGrainResult(
        input_path=pdb_path,
        output_path=output_path,
        output_validation=output_validation,
        residue_count=residue_count,
        inserted_virtual_sites=inserted_virtual_sites,
        missing_terminal_phosphates=missing_terminal_phosphates,
    )


def _build_cg_topology(topology: app.Topology, positions):
    source_positions = positions.value_in_unit(unit.nanometer)
    new_topology = app.Topology()
    box_vectors = topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        new_topology.setPeriodicBoxVectors(box_vectors)

    new_positions: list[Vec3] = []
    residue_count = 0
    inserted_virtual_sites = 0
    missing_terminal_phosphates = 0

    for chain in topology.chains():
        new_chain = new_topology.addChain(chain.id)
        previous_s3 = None
        for residue_index, residue in enumerate(chain.residues()):
            residue_name = _normalize_residue_name(residue.name)
            if residue_name not in _SOURCE_ATOMS:
                raise ValueError(f"Unsupported residue {residue.name!r} in chain {chain.id!r}")

            atom_lookup = {atom.name: atom for atom in residue.atoms()}
            for atom_name in list(_SOURCE_ATOMS[residue_name].values()):
                if atom_name == "P" and residue_index == 0 and atom_name not in atom_lookup:
                    continue
                if atom_name not in atom_lookup:
                    raise ValueError(
                        f"Residue {residue.index} {residue_name} is missing required atom {atom_name!r}"
                    )

            source_p = atom_lookup.get("P")
            if residue_index == 0 and source_p is None:
                missing_terminal_phosphates += 1
            if residue_index > 0 and source_p is None:
                raise ValueError(
                    f"Residue {residue.index} {residue_name} is missing phosphate P; only chain-start residues may omit it"
                )

            new_residue = new_topology.addResidue(residue_name, new_chain, residue.id)
            new_atoms: dict[str, app.Atom] = {}
            new_positions_by_name: dict[str, Vec3] = {}

            for atom_name in _OUTPUT_ATOM_ORDER[residue_name]:
                if atom_name == "P":
                    if source_p is None:
                        continue
                    atom = new_topology.addAtom("P", _OUTPUT_ELEMENTS["P"], new_residue)
                    position = source_positions[source_p.index]
                    new_positions.append(position)
                    new_atoms["P"] = atom
                    new_positions_by_name["P"] = position
                    continue

                if atom_name == "BC":
                    atom = new_topology.addAtom("BC", _OUTPUT_ELEMENTS["BC"], new_residue)
                    position = _virtual_site_cog(new_positions_by_name, residue_name)
                    new_positions.append(position)
                    new_atoms["BC"] = atom
                    new_positions_by_name["BC"] = position
                    inserted_virtual_sites += 1
                    continue

                if atom_name == "BN":
                    atom = new_topology.addAtom("BN", _OUTPUT_ELEMENTS["BN"], new_residue)
                    position = _virtual_site_normal(new_positions_by_name, residue_name)
                    new_positions.append(position)
                    new_atoms["BN"] = atom
                    new_positions_by_name["BN"] = position
                    inserted_virtual_sites += 1
                    continue

                source_atom_name = _SOURCE_ATOMS[residue_name][atom_name]
                source_atom = atom_lookup[source_atom_name]
                atom = new_topology.addAtom(atom_name, _OUTPUT_ELEMENTS[atom_name], new_residue)
                position = source_positions[source_atom.index]
                new_positions.append(position)
                new_atoms[atom_name] = atom
                new_positions_by_name[atom_name] = position

            base1, base2, base3 = _base_bead_names(residue_name)
            if "P" in new_atoms:
                new_topology.addBond(new_atoms["P"], new_atoms["S3"])
            new_topology.addBond(new_atoms["S3"], new_atoms["S2"])
            new_topology.addBond(new_atoms["S2"], new_atoms[base1])
            new_topology.addBond(new_atoms[base1], new_atoms[base2])
            new_topology.addBond(new_atoms[base1], new_atoms[base3])
            new_topology.addBond(new_atoms[base2], new_atoms[base3])
            new_topology.addBond(new_atoms[base1], new_atoms["BC"])
            new_topology.addBond(new_atoms[base2], new_atoms["BC"])
            new_topology.addBond(new_atoms["BC"], new_atoms["BN"])

            if previous_s3 is not None and "P" in new_atoms:
                new_topology.addBond(previous_s3, new_atoms["P"])
            previous_s3 = new_atoms["S3"]
            residue_count += 1

    return new_topology, unit.Quantity(new_positions, unit.nanometer), residue_count, inserted_virtual_sites, missing_terminal_phosphates


def _normalize_residue_name(name: str) -> str:
    return _CANONICAL_RESIDUE_NAME.get(name, name)


def _base_bead_names(residue_name: str) -> tuple[str, str, str]:
    return {
        "A": ("R1", "A1", "A2"),
        "U": ("Y1", "U1", "Y2"),
        "G": ("R1", "G1", "G2"),
        "C": ("Y1", "C1", "Y2"),
    }[residue_name]


def _virtual_site_cog(new_positions_by_name: dict[str, Vec3], residue_name: str) -> Vec3:
    b1, b2, b3 = _base_bead_names(residue_name)
    p1 = new_positions_by_name[b1]
    p2 = new_positions_by_name[b2]
    p3 = new_positions_by_name[b3]
    return (p1 + p2 + p3) / 3.0


def _virtual_site_normal(new_positions_by_name: dict[str, Vec3], residue_name: str) -> Vec3:
    b1, b2, b3 = _base_bead_names(residue_name)
    p1 = new_positions_by_name[b1]
    p2 = new_positions_by_name[b2]
    p3 = new_positions_by_name[b3]
    cog = (p1 + p2 + p3) / 3.0
    return cog + _cross(p2 - p1, p3 - p1) * 0.1


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return Vec3(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    )
