from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

RNA_RESIDUES = {"A", "U", "G", "C"}
REQUIRED_VIRTUAL_SITES = {"BC", "BN"}
EXPECTED_BEADS = {
    "A": {"P", "S3", "S2", "R1", "A1", "A2", "BC", "BN"},
    "G": {"P", "S3", "S2", "R1", "G1", "G2", "BC", "BN"},
    "C": {"P", "S3", "S2", "Y1", "C1", "Y2", "BC", "BN"},
    "U": {"P", "S3", "S2", "Y1", "U1", "Y2", "BC", "BN"},
}


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    valid: bool
    atom_count: int = 0
    residue_count: int = 0
    bond_count: int = 0
    frame_count: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def raise_for_errors(self) -> None:
        if not self.valid:
            details = "\n".join(f"- {error}" for error in self.errors)
            raise ValueError(f"Invalid CRANBERRY input {self.path}:\n{details}")


def validate_canonical_pdb(path: str | Path) -> ValidationResult:
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []

    if not path.exists():
        return ValidationResult(path=path, valid=False, errors=("file does not exist",))

    try:
        from openmm import app
    except ImportError:
        return ValidationResult(
            path=path,
            valid=False,
            errors=("OpenMM is required for PDB validation but is not installed",),
        )

    try:
        pdb = app.PDBFile(str(path))
    except Exception as exc:  # OpenMM can raise several parser exceptions.
        return ValidationResult(path=path, valid=False, errors=(f"OpenMM could not parse PDB: {exc}",))

    atoms = list(pdb.topology.atoms())
    residues = list(pdb.topology.residues())
    bonds = list(pdb.topology.bonds())
    chain_start_residues = {next(chain.residues(), None) for chain in pdb.topology.chains()}

    if not atoms:
        errors.append("no atoms found")
    if not residues:
        errors.append("no residues found")
    if not bonds:
        errors.append("no topology bonds found; canonical input must include CONECT records")

    atom_names = {atom.name for atom in atoms}
    missing_vs = sorted(REQUIRED_VIRTUAL_SITES - atom_names)
    if missing_vs:
        errors.append(f"missing required virtual site bead names: {', '.join(missing_vs)}")

    for residue in residues:
        if residue.name not in RNA_RESIDUES:
            errors.append(f"unsupported residue {residue.name!r} at index {residue.index}")
            continue

        names = {atom.name for atom in residue.atoms()}
        expected = EXPECTED_BEADS[residue.name]
        residue_atoms = list(residue.atoms())
        if len(names) != len(residue_atoms):
            errors.append(f"residue {residue.index} {residue.name} has duplicate bead names")
        missing = sorted(expected - names)
        if missing == ["P"] and residue in chain_start_residues:
            warnings.append(f"chain {residue.chain.index} first residue is missing terminal phosphate P. This is usually fine or recomended. You should only use cranberry prepare *.pdb --add-terminal-phosphate to add 5'-phosphate if you want to get access to the sugar puckering state of the 5'-terminal nucleotide.")
        elif missing:
            errors.append(
                f"residue {residue.index} {residue.name} missing expected beads: {', '.join(missing)}"
            )

        unexpected = sorted(names - expected)
        if unexpected:
            errors.append(
                f"residue {residue.index} {residue.name} has unexpected beads: {', '.join(unexpected)}"
            )

    expected_bonds = _expected_canonical_bonds(pdb.topology)
    actual_bonds = {
        tuple(sorted((bond.atom1.index, bond.atom2.index)))
        for bond in bonds
    }
    missing_bonds = expected_bonds - actual_bonds
    extra_bonds = actual_bonds - expected_bonds
    if missing_bonds:
        errors.append(
            "missing canonical topology bonds: "
            + ", ".join(_format_bond(pdb.topology, pair) for pair in sorted(missing_bonds)[:8])
        )
    if extra_bonds:
        errors.append(
            "unexpected canonical topology bonds: "
            + ", ".join(_format_bond(pdb.topology, pair) for pair in sorted(extra_bonds)[:8])
        )

    return ValidationResult(
        path=path,
        valid=not errors,
        atom_count=len(atoms),
        residue_count=len(residues),
        bond_count=len(bonds),
        frame_count=pdb.getNumFrames(),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _expected_canonical_bonds(topology) -> set[tuple[int, int]]:
    expected_bonds: set[tuple[int, int]] = set()
    base_beads = {
        "A": ("R1", "A1", "A2"),
        "U": ("Y1", "U1", "Y2"),
        "G": ("R1", "G1", "G2"),
        "C": ("Y1", "C1", "Y2"),
    }

    for chain in topology.chains():
        previous_s3 = None
        for residue_offset, residue in enumerate(chain.residues()):
            if residue.name not in RNA_RESIDUES:
                previous_s3 = None
                continue
            residue_atoms = list(residue.atoms())
            atom_by_name = {atom.name: atom for atom in residue_atoms}
            expected_names = EXPECTED_BEADS[residue.name]
            missing = expected_names - set(atom_by_name)
            terminal_without_phosphate = (
                residue_offset == 0 and missing == {"P"}
            )
            if missing and not terminal_without_phosphate:
                previous_s3 = None
                continue
            if len(atom_by_name) != len(residue_atoms):
                previous_s3 = None
                continue

            base1, base2, base3 = base_beads[residue.name]
            names = ["S3", "S2", base1, base2, base3, "BC", "BN"]
            pairs = [
                ("S3", "S2"),
                ("S2", base1),
                (base1, base2),
                (base1, base3),
                (base2, base3),
                (base1, "BC"),
                (base2, "BC"),
                ("BC", "BN"),
            ]
            if "P" in atom_by_name:
                names.insert(0, "P")
                pairs.insert(0, ("P", "S3"))
            if not all(name in atom_by_name for name in names):
                previous_s3 = None
                continue
            expected_bonds.update(
                tuple(sorted((atom_by_name[first].index, atom_by_name[second].index)))
                for first, second in pairs
            )
            if previous_s3 is not None and "P" in atom_by_name:
                expected_bonds.add(
                    tuple(sorted((previous_s3.index, atom_by_name["P"].index)))
                )
            previous_s3 = atom_by_name["S3"]

    return expected_bonds


def _format_bond(topology, pair: tuple[int, int]) -> str:
    atoms = list(topology.atoms())
    return f"{_format_atom(atoms[pair[0]])}-{_format_atom(atoms[pair[1]])}"


def _format_atom(atom) -> str:
    return f"{atom.residue.chain.id or atom.residue.chain.index}:{atom.residue.id}:{atom.name}"
