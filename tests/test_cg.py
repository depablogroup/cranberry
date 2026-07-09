from __future__ import annotations

import math

from openmm import app, unit

from cranberry.cg import coarse_grain_structure
from cranberry.data import data_path
from cranberry.validation import validate_canonical_pdb


def _atom_key(atom):
    return (atom.residue.chain.id, atom.residue.id, atom.residue.name, atom.name)


def _residue_y_atoms(residue, values):
    if residue.name not in {'C', 'U'}:
        return []
    y_atoms = [atom for atom in residue.atoms() if atom.name in {'Y1', 'Y2'}]
    return sorted(
        y_atoms,
        key=lambda atom: (
            values[atom.index].x,
            values[atom.index].y,
            values[atom.index].z,
        ),
    )


def _normalized_atom_key(atom, residue_y_atoms):
    name = atom.name
    if atom.residue.name in {'C', 'U'} and atom.name in {'Y1', 'Y2'}:
        y_atoms = residue_y_atoms[atom.residue]
        name = 'Y1' if atom is y_atoms[0] else 'Y2'
    return (atom.residue.chain.id, atom.residue.id, atom.residue.name, name)


def _atom_positions(path):
    pdb = app.PDBFile(str(path))
    values = pdb.positions.value_in_unit(unit.nanometer)
    residue_y_atoms = {residue: _residue_y_atoms(residue, values) for residue in pdb.topology.residues()}
    return {
        _normalized_atom_key(atom, residue_y_atoms): values[atom.index]
        for atom in pdb.topology.atoms()
        if atom.name != 'BN'
    }


def _bond_pairs(path):
    pdb = app.PDBFile(str(path))
    values = pdb.positions.value_in_unit(unit.nanometer)
    residue_y_atoms = {residue: _residue_y_atoms(residue, values) for residue in pdb.topology.residues()}
    pairs = set()
    for bond in pdb.topology.bonds():
        pairs.add(
            tuple(
                sorted(
                    (
                        _normalized_atom_key(bond.atom1, residue_y_atoms),
                        _normalized_atom_key(bond.atom2, residue_y_atoms),
                    )
                )
            )
        )
    return pairs


def _residue_atom_names(path):
    pdb = app.PDBFile(str(path))
    values = pdb.positions.value_in_unit(unit.nanometer)
    residue_y_atoms = {residue: _residue_y_atoms(residue, values) for residue in pdb.topology.residues()}
    names = []
    for residue in pdb.topology.residues():
        residue_names = []
        for atom in residue.atoms():
            name = atom.name
            if residue.name in {'C', 'U'} and atom.name in {'Y1', 'Y2'}:
                y_atoms = residue_y_atoms[residue]
                name = 'Y1' if atom is y_atoms[0] else 'Y2'
            residue_names.append(name)
        names.append(sorted(residue_names))
    return names


def test_coarse_grain_structure_matches_packaged_1zih(tmp_path):
    source = data_path('examples/aa/1zih/1zih.pdb')
    reference = data_path('examples/1zih_cg_vs_conect.pdb')
    output = tmp_path / '1zih_cg_vs_conect.pdb'

    result = coarse_grain_structure(source, output_path=output)

    assert result.output_path == output
    assert result.residue_count == 12
    assert result.inserted_virtual_sites == 24
    assert result.output_validation.valid
    validate_canonical_pdb(result.output_path).raise_for_errors()

    assert _residue_atom_names(result.output_path) == _residue_atom_names(reference)
    assert _bond_pairs(result.output_path) == _bond_pairs(reference)

    output_positions = _atom_positions(result.output_path)
    reference_positions = _atom_positions(reference)
    assert output_positions.keys() == reference_positions.keys()
    for key, output_position in output_positions.items():
        reference_position = reference_positions[key]
        distance = math.sqrt(
            (output_position.x - reference_position.x) ** 2
            + (output_position.y - reference_position.y) ** 2
            + (output_position.z - reference_position.z) ** 2
        )
        assert distance <= 1e-3, key
