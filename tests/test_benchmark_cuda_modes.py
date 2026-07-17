from benchmarks.benchmark_cuda_modes import DEFAULT_PDB, to_yaml


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
