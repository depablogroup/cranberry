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
        missing = sorted(expected - names)
        if missing == ["P"] and residue in chain_start_residues:
            warnings.append(f"chain {residue.chain.index} first residue doesn't have terminal phosphate P. This is usually fine or recomended. You should only use cranberry prepare *.pdb --add-terminal-phosphate to add 5'-phosphate if you want to get access to the sugar puckering state of the 5'-terminal nucleotide.")
        elif missing:
            errors.append(
                f"residue {residue.index} {residue.name} missing expected beads: {', '.join(missing)}"
            )

        unexpected = sorted(names - expected)
        if unexpected:
            errors.append(
                f"residue {residue.index} {residue.name} has unexpected beads: {', '.join(unexpected)}"
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
