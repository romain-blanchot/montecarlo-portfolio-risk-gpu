# useful for running the package as a script, e.g. `python -m portfolio_risk_engine`
# not necessary for production & application code, but can be useful for testing and development

from .cli import main

if __name__ == "__main__":
    main()
