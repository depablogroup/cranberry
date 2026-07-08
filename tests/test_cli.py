from cranberry.cli.main import main
from cranberry.data import data_path


def test_cli_help(capsys):
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "CRANBERRY coarse-grained RNA" in captured.out
    assert "prepare" in captured.out
    assert "cg" in captured.out
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


def test_cli_md_smoke(tmp_path, capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    assert main(["md", str(path), "--steps", "1", "--report-interval", "1", "--output-dir", str(tmp_path), "--platform", "CPU"]) == 0
    captured = capsys.readouterr()
    assert "output directory:" in captured.out
    assert (tmp_path / "output.dcd").exists()
    assert (tmp_path / "log").exists()
    assert (tmp_path / "detailed.log").exists()
    assert (tmp_path / "args.json").exists()
    assert (tmp_path / "checkpoint.chk").exists()
    assert (tmp_path / "final.pdb").exists()


def test_cli_md_restart_smoke(tmp_path, capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    assert main(["md", str(path), "--steps", "1", "--report-interval", "1", "--output-dir", str(tmp_path), "--platform", "CPU"]) == 0
    capsys.readouterr()
    assert main([
        "md",
        str(path),
        "--steps",
        "1",
        "--report-interval",
        "1",
        "--output-dir",
        str(tmp_path),
        "--restart-from",
        str(tmp_path / "checkpoint.chk"),
        "--platform",
        "CPU",
    ]) == 0
    captured = capsys.readouterr()
    assert "restarted from:" in captured.out
    assert (tmp_path / "checkpoint.chk").exists()
    detailed_steps = [line.split(",", 1)[0] for line in (tmp_path / "detailed.log").read_text().splitlines()[1:]]
    assert detailed_steps == ["1", "2"]


def test_cli_prepare_is_noop_without_terminal_phosphate(tmp_path, capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    output = tmp_path / "prepared.pdb"
    assert main(["prepare", str(path), "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert "Nothing to do:" in captured.out
    assert "--add-terminal-phosphate" in captured.out
    assert not output.exists()


def test_cli_prepare_inserts_terminal_phosphate(tmp_path, capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    output = tmp_path / "prepared.pdb"
    assert main(["prepare", str(path), "--output", str(output), "--add-terminal-phosphate"]) == 0
    captured = capsys.readouterr()
    assert "added terminal phosphates: 1" in captured.out
    assert output.exists()


def test_cli_cg_alias_is_noop_without_terminal_phosphate(tmp_path, capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    output = tmp_path / "cg.pdb"
    assert main(["cg", str(path), "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert "Nothing to do:" in captured.out
    assert not output.exists()
