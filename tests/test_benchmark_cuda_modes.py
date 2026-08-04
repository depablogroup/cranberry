from benchmarks.benchmark_cuda_modes import DEFAULT_PDB, parse_openmmtools_yaml, portable_snapshot, to_yaml


def test_cuda_modes_yaml_writer_handles_nested_values():
    payload = {
        "schema_version": 1,
        "runs": [
            {"kind": "single-md-cuda", "gpu_utilization_percent_mean": 42.5, "notes": ["a:b"]},
        ],
    }

    text = to_yaml(payload)

    assert "schema_version: 1" in text
    assert "kind: 'single-md-cuda'" in text
    assert "gpu_utilization_percent_mean: 42.5" in text
    assert "'a:b'" in text


def test_cuda_modes_default_fixture_is_ggcgcaa():
    assert DEFAULT_PDB == "examples/ggcGCAAgcc_cg_vs_conect.pdb"


def test_cuda_modes_reads_openmmtools_online_analysis_speed(tmp_path):
    analysis = tmp_path / "output_real_time_analysis.yaml"
    analysis.write_text(
        "- iteration: 20\n"
        "  timing_data:\n"
        "    average_seconds_per_iteration: 1.25\n"
        "    ns_per_day: 276.48\n"
        "- iteration: 40\n"
        "  timing_data:\n"
        "    average_seconds_per_iteration: 1.0\n"
        "    ns_per_day: 345.6\n"
    )

    summary = parse_openmmtools_yaml(analysis)

    assert summary == {
        "path": str(analysis),
        "iteration_records": 2,
        "last_iteration": 40,
        "timing_data_ns_per_day_values": [276.48, 345.6],
        "timing_data_average_seconds_per_iteration_values": [1.25, 1.0],
    }


def test_cuda_modes_snapshot_removes_machine_specific_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = {
        "host": "workstation.example",
        "pdb": str(tmp_path / "cranberry" / "data" / "example.pdb"),
        "command": ["/opt/conda/envs/cranberry/bin/python", "--input", "/private/source/input.pdb"],
        "nested": {"output": str(tmp_path / "benchmarks" / "result.yaml")},
    }

    portable = portable_snapshot(payload)

    assert portable["pdb"] == "cranberry/data/example.pdb"
    assert portable["command"] == ["<external>/python", "--input", "<external>/input.pdb"]
    assert portable["nested"]["output"] == "benchmarks/result.yaml"
    assert "/home/" not in to_yaml(portable)
    assert portable["host"] == "<host>"
