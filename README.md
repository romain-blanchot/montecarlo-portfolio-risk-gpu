# montecarlo-portfolio-risk-gpu

This project implements a GPU-accelerated Monte Carlo engine to simulate multi-asset portfolio dynamics and estimate risk metrics such as Value at Risk (VaR) and Expected Shortfall.


| | |
| --- | --- |
| CI/CD | [![CI - Test](https://github.com/ofek/hatch-vcs/actions/workflows/test.yml/badge.svg)](https://github.com/ofek/hatch-vcs/actions/workflows/test.yml) [![CD - Build](https://github.com/ofek/hatch-vcs/actions/workflows/build.yml/badge.svg)](https://github.com/ofek/hatch-vcs/actions/workflows/build.yml) |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/hatch-vcs.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/hatch-vcs/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hatch-vcs.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/hatch-vcs/) |
| Meta | [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/ambv/black) [![License - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/) [![GitHub Sponsors](https://img.shields.io/github/sponsors/ofek?logo=GitHub%20Sponsors&style=social)](https://github.com/sponsors/ofek) |

-----

# TODO:
- mkdocs
- ci
- cd
- release
- pre commit format
- dev dependancies
- dockerfile



conda env update -f environment.yml --prune
conda activate portfolio-risk-engine
python -m pip install -U pip
python -m pip install --group dev


pip install pre-commit
pre-commit install

