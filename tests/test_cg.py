from __future__ import annotations

from openmm import app

from cranberry.cg import coarse_grain_structure
from cranberry.validation import validate_canonical_pdb


def _atom_line(serial, atom_name, residue_name, chain_id, residue_id, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} {chain_id}{residue_id:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
    )


def _write_atomistic_fragment(path):
    lines = [
        _atom_line(1, "P", "A", "A", 1, -2.0, 0.0, 0.0, "P"),
        _atom_line(2, "C3'", "A", "A", 1, -1.0, 0.0, 0.0, "C"),
        _atom_line(3, "C2'", "A", "A", 1, -1.0, 0.2, 0.0, "C"),
        _atom_line(4, "C8", "A", "A", 1, 0.0, 0.0, 0.0, "C"),
        _atom_line(5, "N6", "A", "A", 1, 0.0, 0.1, 0.1, "N"),
        _atom_line(6, "C2", "A", "A", 1, 0.0, 0.2, 0.0, "C"),
        _atom_line(7, "P", "G", "A", 2, 2.0, 0.0, 0.0, "P"),
        _atom_line(8, "C3'", "G", "A", 2, 3.0, 0.0, 0.0, "C"),
        _atom_line(9, "C2'", "G", "A", 2, 3.0, 0.2, 0.0, "C"),
        _atom_line(10, "C8", "G", "A", 2, 4.0, 0.0, 0.0, "C"),
        _atom_line(11, "O6", "G", "A", 2, 4.0, 0.1, 0.1, "O"),
        _atom_line(12, "N2", "G", "A", 2, 4.0, 0.2, 0.0, "N"),
        "END",
    ]
    path.write_text("\n".join(lines) + "\n")


def _atom_names(path):
    pdb = app.PDBFile(str(path))
    names = []
    for residue in pdb.topology.residues():
        names.append([atom.name for atom in residue.atoms()])
    return names


def test_coarse_grain_structure_converts_atomistic_fragment(tmp_path):
    source = tmp_path / "input.pdb"
    _write_atomistic_fragment(source)

    result = coarse_grain_structure(source, output_path=tmp_path / "cg.pdb")

    assert result.output_path.exists()
    assert result.residue_count == 2
    assert result.inserted_virtual_sites == 4
    assert result.missing_terminal_phosphates == 0
    assert result.output_validation.valid
    validate_canonical_pdb(result.output_path).raise_for_errors()

    names = _atom_names(result.output_path)
    assert names[0] == ["P", "S3", "S2", "R1", "A1", "A2", "BC", "BN"]
    assert names[1] == ["P", "S3", "S2", "R1", "G1", "G2", "BC", "BN"]
