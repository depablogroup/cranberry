import pytest
from cranberry.cli.main import main
from cranberry.data import data_path
from cranberry.remd import RemdRunConfig, TemperatureLadderSpec, run_remd


def test_cli_help(capsys):
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "CRANBERRY coarse-grained RNA" in captured.out
    assert "prepare" in captured.out
    assert "cg" in captured.out
    assert "inspect" in captured.out
    assert "remd" in captured.out


def test_cli_remd_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["remd", "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--temperature-ladder" in captured.out
    assert "--n-replicas" in captured.out
    assert "--n-record" in captured.out
    assert "--n-analysis" in captured.out
    assert "--extra-start-pdb" in captured.out
    assert "--write-dcd" in captured.out
    assert "--periodic" in captured.out
    assert "--box-padding" in captured.out
    assert "--overwrite" in captured.out
    assert "--by-replica" in captured.out
    assert "--by-temperature" in captured.out




def test_cli_md_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["md", "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--periodic" in captured.out
    assert "--box-padding" in captured.out
    assert "--enforce-periodic-output" in captured.out

def test_cli_inspect_summary(capsys):
    assert main(["inspect"]) == 0
    captured = capsys.readouterr()
    assert "cranberry-rna" in captured.out
    assert "cranberry-v1-alpha.1" in captured.out


@pytest.mark.remd
def test_cli_remd_extract_by_replica(tmp_path, capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    result = run_remd(
        RemdRunConfig(
            pdb_path=path,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            overwrite=True,
        )
    )
    capsys.readouterr()
    assert main(["remd-extract", str(result.output_netcdf_path), str(path), "--output-dir", str(tmp_path), "--overwrite"]) == 0
    captured = capsys.readouterr()
    assert "output:" in captured.out
    assert (tmp_path / "output_0.dcd").exists()
    assert (tmp_path / "output_1.dcd").exists()




@pytest.mark.remd
def test_cli_remd_extract_by_temperature(tmp_path, capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    result = run_remd(
        RemdRunConfig(
            pdb_path=path,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            overwrite=True,
        )
    )
    capsys.readouterr()
    assert main(["remd-extract", str(result.output_netcdf_path), str(path), "--output-dir", str(tmp_path), "--by-temperature", "--overwrite"]) == 0
    captured = capsys.readouterr()
    assert "output:" in captured.out
    assert (tmp_path / "output_T0.dcd").exists()
    assert (tmp_path / "output_T1.dcd").exists()
    assert (tmp_path / "output_temperature_labels.txt").exists()


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


def test_cli_md_smoke(tmp_path, monkeypatch, capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    monkeypatch.chdir(tmp_path)
    assert main(["md", str(path), "--steps", "1"]) == 0
    captured = capsys.readouterr()
    assert "settings: model=cranberry-v1-alpha.1" in captured.out
    assert "temperature=298.0 K" in captured.out
    assert "salt=150.0 mM" in captured.out
    assert "timestep=5.0 fs" in captured.out
    assert "periodic=False" in captured.out
    assert "enforce_periodic_output=False" in captured.out
    assert "platform=CPU" in captured.out
    assert "output directory:" in captured.out
    assert (tmp_path / "output.dcd").exists()
    assert (tmp_path / "log").exists()
    assert (tmp_path / "detailed.log").exists()
    assert (tmp_path / "args.json").exists()
    assert (tmp_path / "checkpoint.chk").exists()
    assert (tmp_path / "final.pdb").exists()


@pytest.mark.remd
def test_cli_remd_smoke_with_default_output_dir(tmp_path, monkeypatch, capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    monkeypatch.chdir(tmp_path)
    assert main(["remd", str(path), "--steps", "1", "--temperature-ladder", "298", "318", "--swap-steps", "1"]) == 0
    captured = capsys.readouterr()
    assert "settings:" in captured.out
    assert "netcdf:" in captured.out
    assert "args:" in captured.out
    assert (tmp_path / "output.nc").exists()
    assert (tmp_path / "args.json").exists()


def test_cli_md_restart_smoke(tmp_path, capsys):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    assert main(["md", str(path), "--steps", "1", "--n-record", "1", "--output-dir", str(tmp_path), "--platform", "CPU"]) == 0
    capsys.readouterr()
    assert main([
        "md",
        str(path),
        "--steps",
        "1",
        "--n-record",
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


def test_cli_md_missing_report_interval_regression(tmp_path):
    path = data_path("examples/2ntCG_cg_vs_conect.pdb")
    assert main(["md", str(path), "--steps", "15", "--output-dir", str(tmp_path), "--platform", "CPU"]) == 0


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


def test_cli_cg_coarse_grains_real_1zih(tmp_path, capsys):
    source = data_path("examples/aa/1zih/1zih.pdb")
    output = tmp_path / "1zih_cg_vs_conect.pdb"
    assert main(["cg", str(source), "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert "coarse-grained residues: 12" in captured.out
    assert "virtual-site atoms: 24" in captured.out
    assert output.exists()
