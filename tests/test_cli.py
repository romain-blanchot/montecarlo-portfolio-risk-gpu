from portfolio_risk_engine.cli import main


def test_main_runs(capsys):
    main()
    captured = capsys.readouterr()
    assert "Portfolio simulator running" in captured.out
