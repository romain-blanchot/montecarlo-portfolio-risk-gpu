import math

import numpy as np

from portfolio_risk_engine.application.use_cases.simulate_portfolio import (
    ScenarioResult,
)

ASSET_COLORS = [
    "\033[36m",  # Cyan
    "\033[33m",  # Yellow
    "\033[32m",  # Green
    "\033[35m",  # Magenta
    "\033[34m",  # Blue
    "\033[91m",  # Light red
    "\033[96m",  # Light cyan
    "\033[93m",  # Light yellow
]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
WHITE_BOLD = "\033[1;97m"
YELLOW = "\033[33m"

_BLOCKS = " \u258f\u258e\u258d\u258c\u258b\u258a\u2589\u2588"
_MAX_BAR = 35
_NUM_BINS = 12
_WIDTH = 70


def render_scenario(result: ScenarioResult) -> str:
    lines: list[str] = []

    _render_header(lines, result)
    _render_model_info(lines, result)
    _render_asset_summary(lines, result)
    _render_asset_distributions(lines, result)
    _render_portfolio_distribution(lines, result)
    _render_risk_metrics(lines, result)

    lines.append(f"{BOLD}{'=' * _WIDTH}{RESET}")
    lines.append("")
    return "\n".join(lines)


def _render_header(lines: list[str], result: ScenarioResult) -> None:
    prices = result.historical_prices
    sim = result.simulation_result

    lines.append("")
    lines.append(f"{BOLD}{'=' * _WIDTH}{RESET}")
    lines.append(f"{BOLD}  PORTFOLIO SCENARIO SIMULATION{RESET}")
    lines.append(f"{BOLD}{'=' * _WIDTH}{RESET}")

    start = prices.dates[0]
    end = prices.dates[-1]
    n_obs = len(prices.dates)

    lines.append(f"  Historical data    {start} \u2192 {end}  ({n_obs} observations)")
    lines.append(
        f"  Simulations        {sim.num_simulations:,} paths  |  "
        f"Horizon: {sim.time_horizon_days} trading days"
    )


def _render_model_info(lines: list[str], result: ScenarioResult) -> None:
    info = result.model_info
    if info is None:
        lines.append("  Model              GBM")
    else:
        lines.append(f"  Model              {BOLD}{info.name}{RESET}")

        if info.name == "Student-t GBM" and "df" in info.parameters:
            df = info.parameters["df"]
            lines.append(f"  {DIM}Degrees of freedom  \u03bd = {df:.1f}{RESET}")

        if info.name == "Heston":
            _render_heston_params(lines, result)

    lines.append(f"{'─' * _WIDTH}")
    lines.append("")


def _render_heston_params(lines: list[str], result: ScenarioResult) -> None:
    info = result.model_info
    if info is None:
        return

    tickers = result.portfolio.tickers
    for i, ticker in enumerate(tickers):
        color = ASSET_COLORS[i % len(ASSET_COLORS)]
        prefix = ticker.value
        kappa = info.parameters.get(f"{prefix}_kappa", 0)
        vol_lr = info.parameters.get(f"{prefix}_vol_lr", 0)
        xi = info.parameters.get(f"{prefix}_xi", 0)
        rho = info.parameters.get(f"{prefix}_rho", 0)
        vol0 = info.parameters.get(f"{prefix}_vol0", 0)
        feller = info.parameters.get(f"{prefix}_feller", 0)

        feller_str = f"{GREEN}\u2713{RESET}" if feller else f"{RED}\u2717{RESET}"

        lines.append(
            f"  {color}\u25cf{RESET} {color}{prefix:<6}{RESET} "
            f"\u03ba={kappa:>5.2f}  "
            f"\u03b8={vol_lr:>5.1%}  "
            f"\u03be={xi:>5.2f}  "
            f"\u03c1={rho:>+5.2f}  "
            f"v\u2080={vol0:>5.1%}  "
            f"{DIM}Feller:{RESET}{feller_str}"
        )


def _render_asset_summary(lines: list[str], result: ScenarioResult) -> None:
    lines.append(f"  {BOLD}ASSET SUMMARY{RESET}")

    portfolio = result.portfolio
    sim = result.simulation_result
    mkt = result.market_parameters
    initial_map = dict(zip(sim.tickers, sim.initial_prices))

    for i, (ticker, weight) in enumerate(zip(portfolio.tickers, portfolio.weights)):
        color = ASSET_COLORS[i % len(ASSET_COLORS)]
        s0 = initial_map[ticker]
        terminal = np.array(sim.terminal_prices[ticker])
        mean_st = float(np.mean(terminal))
        pct = (mean_st / s0 - 1) * 100
        sign = "+" if pct >= 0 else ""

        idx = list(mkt.tickers).index(ticker)
        drift = mkt.drift_vector[idx]
        vol = math.sqrt(mkt.covariance_matrix[idx][idx])

        lines.append(
            f"  {color}\u25cf{RESET} {color}{ticker.value:<6}{RESET} "
            f"{weight:>5.1%}    "
            f"S\u2080 ${s0:>8.2f}    "
            f"E[S_T] ${mean_st:>8.2f}    "
            f"{sign}{pct:.1f}%    "
            f"{DIM}\u03bc {drift:+.1%}  \u03c3 {vol:.1%}{RESET}"
        )

    lines.append("")


def _render_histogram(
    lines: list[str],
    data: np.ndarray,
    color: str,
    fmt_label: str = "price",
) -> None:
    counts, bin_edges = np.histogram(data, bins=_NUM_BINS)
    max_count = int(np.max(counts))

    for j in range(len(counts)):
        lo = bin_edges[j]
        pct = counts[j] / len(data) * 100

        bar_width = counts[j] / max_count * _MAX_BAR if max_count > 0 else 0.0
        bar = _bar(bar_width, color)

        if fmt_label == "price":
            label = f"${lo:>9.2f}"
        else:
            label = f"{lo:>+9.2%}"

        lines.append(f"    {label}  \u2524{bar} {DIM}{pct:>5.1f}%{RESET}")

    # Percentiles
    p5 = float(np.percentile(data, 5))
    p50 = float(np.percentile(data, 50))
    p95 = float(np.percentile(data, 95))

    if fmt_label == "price":
        pline = (
            f"    {DIM}P5 ${p5:.2f}  \u00b7  "
            f"P50 ${p50:.2f}  \u00b7  "
            f"P95 ${p95:.2f}{RESET}"
        )
    else:
        pline = (
            f"    {DIM}P5 {p5:+.2%}  \u00b7  "
            f"P50 {p50:+.2%}  \u00b7  "
            f"P95 {p95:+.2%}{RESET}"
        )

    lines.append(pline)
    lines.append("")


def _render_asset_distributions(lines: list[str], result: ScenarioResult) -> None:
    sim = result.simulation_result
    initial_map = dict(zip(sim.tickers, sim.initial_prices))

    for i, ticker in enumerate(result.portfolio.tickers):
        color = ASSET_COLORS[i % len(ASSET_COLORS)]
        s0 = initial_map[ticker]
        terminal = np.array(sim.terminal_prices[ticker])

        lines.append(
            f"  {color}{BOLD}{ticker.value}{RESET}  "
            f"{DIM}Terminal Price Distribution  "
            f"(S\u2080 = ${s0:.2f}){RESET}"
        )
        _render_histogram(lines, terminal, color, fmt_label="price")


def _compute_portfolio_returns(result: ScenarioResult) -> np.ndarray:
    sim = result.simulation_result
    portfolio = result.portfolio
    initial_map = dict(zip(sim.tickers, sim.initial_prices))
    weight_map = dict(zip(portfolio.tickers, portfolio.weights))

    portfolio_returns = np.zeros(sim.num_simulations)
    for ticker in sim.tickers:
        w = weight_map[ticker]
        s0 = initial_map[ticker]
        st = np.array(sim.terminal_prices[ticker])
        portfolio_returns += w * (st / s0 - 1.0)

    return portfolio_returns


def _render_portfolio_distribution(lines: list[str], result: ScenarioResult) -> None:
    portfolio_returns = _compute_portfolio_returns(result)

    lines.append(f"  {WHITE_BOLD}PORTFOLIO{RESET}  {DIM}Return Distribution{RESET}")
    _render_histogram(lines, portfolio_returns, WHITE_BOLD, fmt_label="return")


def _render_risk_metrics(lines: list[str], result: ScenarioResult) -> None:
    risk = result.risk_metrics
    sep = f"  {'─' * 36}"

    portfolio_returns = _compute_portfolio_returns(result)

    # Tail metrics
    mean = float(np.mean(portfolio_returns))
    std = float(np.std(portfolio_returns, ddof=1))
    if std > 0:
        z = (portfolio_returns - mean) / std
        skewness = float(np.mean(z**3))
        excess_kurtosis = float(np.mean(z**4)) - 3.0
    else:
        skewness = 0.0
        excess_kurtosis = 0.0

    prob_loss = float(np.mean(portfolio_returns < 0))

    lines.append(f"  {BOLD}PORTFOLIO RISK{RESET}")
    lines.append(sep)

    mr_color = GREEN if risk.mean_return >= 0 else RED
    lines.append(f"  Mean Return       {mr_color}{risk.mean_return:>+12.4%}{RESET}")
    lines.append(f"  Volatility        {risk.volatility:>12.4%}")

    # Tail metrics
    skew_color = YELLOW if abs(skewness) > 0.5 else DIM
    kurt_color = YELLOW if excess_kurtosis > 1.0 else DIM
    loss_color = RED if prob_loss > 0.5 else GREEN

    lines.append(f"  Skewness          {skew_color}{skewness:>12.4f}{RESET}")
    lines.append(f"  Excess Kurtosis   {kurt_color}{excess_kurtosis:>12.4f}{RESET}")
    lines.append(f"  Prob(Loss)        {loss_color}{prob_loss:>12.2%}{RESET}")
    lines.append(sep)
    lines.append(f"  VaR  95%          {RED}{risk.var_95:>12.4%}{RESET}")
    lines.append(f"  VaR  99%          {RED}{risk.var_99:>12.4%}{RESET}")
    lines.append(f"  ES   95%          {RED}{risk.es_95:>12.4%}{RESET}")
    lines.append(f"  ES   99%          {RED}{risk.es_99:>12.4%}{RESET}")
    lines.append(sep)
    lines.append("")


def _bar(width: float, color: str) -> str:
    full = int(width)
    frac = width - full
    idx = round(frac * 8)
    bar = "\u2588" * full
    if 0 < idx < 9:
        bar += _BLOCKS[idx]
    return f"{color}{bar}{RESET}"
