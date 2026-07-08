import math

from openmm import app, unit

from cranberry.data import data_path
from cranberry.prepare import prepare_structure
from cranberry.validation import validate_canonical_pdb


def _extract_conect_lines(path):
    return [line.rstrip() for line in path.read_text().splitlines() if line.startswith("CONECT")]


def _bonded_atom_pairs(path):
    pdb = app.PDBFile(str(path))
    pairs = set()
    for bond in pdb.topology.bonds():
        atom1 = bond.atom1
        atom2 = bond.atom2
        key1 = (atom1.residue.chain.id, atom1.residue.id, atom1.residue.name, atom1.name)
        key2 = (atom2.residue.chain.id, atom2.residue.id, atom2.residue.name, atom2.name)
        pairs.add(tuple(sorted((key1, key2))))
    return pairs


def _atom_positions(path):
    pdb = app.PDBFile(str(path))
    coords = pdb.positions.value_in_unit(unit.nanometer)
    mapping = {}
    for atom in pdb.topology.atoms():
        key = (atom.residue.chain.id, atom.residue.id, atom.residue.name, atom.name)
        mapping[key] = coords[atom.index]
    return mapping


def _distance(left, right):
    delta = left - right
    return math.sqrt(delta.x**2 + delta.y**2 + delta.z**2)


def _angle(vertex, point1, point2):
    v1 = point1 - vertex
    v2 = point2 - vertex
    dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    n1 = math.sqrt(v1.x**2 + v1.y**2 + v1.z**2)
    n2 = math.sqrt(v2.x**2 + v2.y**2 + v2.z**2)
    cosine = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.acos(cosine)


def test_prepare_structure_is_noop_without_terminal_phosphate(tmp_path):
    source_path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    output = tmp_path / "prepared.pdb"
    result = prepare_structure(
        source_path,
        output_path=output,
    )
    assert result.input_path.name == "2ntCG_cg_vs_conect.pdb"
    assert result.output_path is None
    assert not output.exists()
    assert result.inserted_terminal_phosphates == 0
    assert result.output_validation.valid
    assert result.output_validation == result.input_validation


def test_prepare_structure_can_insert_terminal_phosphate(tmp_path):
    source_path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    result = prepare_structure(
        source_path,
        output_path=tmp_path / "prepared_phosphate.pdb",
        add_terminal_phosphate=True,
    )
    assert result.inserted_terminal_phosphates == 1
    assert result.output_validation.valid
    assert not result.output_validation.warnings

    bonded_pairs = _bonded_atom_pairs(result.output_path)
    inserted_p_s3 = tuple(sorted((("A", "1", "C", "P"), ("A", "1", "C", "S3"))))
    spurious_bn_p = tuple(sorted((("A", "1", "C", "BN"), ("A", "1", "C", "P"))))

    assert inserted_p_s3 in bonded_pairs
    assert spurious_bn_p not in bonded_pairs

    positions = _atom_positions(result.output_path)
    phosphate = positions[("A", "1", "C", "P")]
    s3 = positions[("A", "1", "C", "S3")]
    s2 = positions[("A", "1", "C", "S2")]
    next_p = positions[("A", "2", "G", "P")]
    assert math.isclose(_distance(phosphate, s3), 0.45, rel_tol=0.0, abs_tol=5e-4)
    assert math.isclose(_angle(s3, phosphate, s2), 2.356, rel_tol=0.0, abs_tol=1e-2)
    assert math.isclose(_angle(s3, phosphate, next_p), 1.920, rel_tol=0.0, abs_tol=1e-2)

    validate_canonical_pdb(result.output_path).raise_for_errors()
