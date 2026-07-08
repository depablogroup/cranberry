from benchmarks.benchmark_md import SYSTEMS, card_label


def test_benchmark_systems_include_new_examples():
    assert SYSTEMS["rU40"] == "examples/rU40_cg_vs_conect.pdb"
    assert SYSTEMS["5ml7"] == "examples/5ml7_cg_vs_conect.pdb"
    assert SYSTEMS["2mi0"] == "examples/2MI0_cg_vs_conect.pdb"


def test_card_label_shortens_rtx_2060():
    assert card_label("NVIDIA GeForce RTX 2060", "CUDA") == "2060"
