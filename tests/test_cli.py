from cranberry.cli.main import main


def test_cli_help(capsys):
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "CRANBERRY coarse-grained RNA" in captured.out
    assert "prepare" in captured.out
    assert "inspect" in captured.out
