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
