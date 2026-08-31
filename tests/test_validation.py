from cranberry.data import data_path
from cranberry.validation import validate_canonical_pdb


def test_packaged_examples_validate():
    for name in [
        "157d_cg_vs_conect.pdb",
        "1l2x_cg_vs_conect.pdb",
        "1zih_cg_vs_conect.pdb",
        "2ntCG_cg_vs_conect.pdb",
    ]:
        result = validate_canonical_pdb(data_path(f"examples/{name}"))
        assert result.valid, (name, result.errors)
        assert result.atom_count > 0
        assert result.residue_count > 0
        assert result.bond_count > 0


def test_invalid_input_reports_missing_file(tmp_path):
    result = validate_canonical_pdb(tmp_path / "missing.pdb")
    assert not result.valid
    assert result.errors == ("file does not exist",)



def test_terminal_phosphate_warnings_are_reported():
    result = validate_canonical_pdb(data_path("examples/2ntCG_cg_vs_conect.pdb"))
    assert result.valid
    assert any("missing terminal phosphate P" in warning for warning in result.warnings)


def test_validation_rejects_incomplete_canonical_bond_graph(tmp_path):
    source = data_path("examples/2ntCG_cg_vs_conect.pdb")
    lines = [line for line in source.read_text().splitlines() if not line.startswith("CONECT")]
    lines.extend(["CONECT    1    2", "END"])
    malformed = tmp_path / "malformed.pdb"
    malformed.write_text("\n".join(lines) + "\n")

    result = validate_canonical_pdb(malformed)

    assert not result.valid
    assert any("missing canonical topology bonds" in error for error in result.errors)
