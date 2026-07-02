from cranberry.cli.main import main
from cranberry.data import data_path


def test_cli_help(capsys):
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "CRANBERRY coarse-grained RNA" in captured.out
    assert "prepare" in captured.out
    assert "inspect" in captured.out


def test_cli_inspect_summary(capsys):
    assert main(["inspect"]) == 0
    captured = capsys.readouterr()
    assert "cranberry-rna" in captured.out
    assert "cranberry-v1-alpha.1" in captured.out


def test_cli_inspect_forcefield(capsys):
    assert main(["inspect", "forcefield"]) == 0
    captured = capsys.readouterr()
    assert "model: cranberry-v1-alpha.1" in captured.out
    assert "electrostatic" in captured.out


def test_cli_inspect_input(capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    assert main(["inspect", "input", str(path)]) == 0
    captured = capsys.readouterr()
    assert "valid: True" in captured.out
    assert "residues: 2" in captured.out
