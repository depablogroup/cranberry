import cranberry


def test_version_is_string():
    assert isinstance(cranberry.__version__, str)
    assert cranberry.__version__
