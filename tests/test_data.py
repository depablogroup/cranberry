import h5py

from cranberry.data import available_forcefields, data_path
from cranberry.forcefield import available_models, default_model_name, get_model_spec


def test_packaged_forcefield_assets_exist():
    assert "cranberry-v1-alpha.1.h5" in available_forcefields()
    assert data_path("forcefields/cranberry-v1-alpha.1.h5").is_file()
    assert data_path("xml/cranberry.xml").is_file()


def test_model_registry_resolves_default():
    assert default_model_name() == "cranberry-v1-alpha.1"
    assert "cranberry-v1-alpha.1" in available_models()
    spec = get_model_spec("default")
    assert spec.name == "cranberry-v1-alpha.1"
    assert spec.parameter_path.is_file()
    assert spec.xml_path.is_file()


def test_canonical_h5_is_merged_and_annotated():
    with h5py.File(data_path("forcefields/cranberry-v1-alpha.1.h5"), "r") as h5:
        assert {"bond", "angle", "dihedral", "sugar", "wca", "spline", "stacking35", "stacking55", "stacking33", "pairing"} <= set(h5.keys())
        assert h5.attrs["cranberry_model"] == "cranberry-v1-alpha.1"
        assert h5.attrs["angle_scaling_baked_in"] == 0.1
        assert "bonded_source" in h5.attrs
        assert "nonbonded_source" in h5.attrs
