"""Command-line interface for KlawTrade."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("klawtrade")
    except Exception:
        return "0.1.0"


def cmd_start(args: argparse.Namespace) -> None:
    """Start the KlawTrade trading system."""
    if args.config:
        os.environ["KLAWTRADE_CONFIG"] = str(args.config)

    if args.port:
        os.environ["KLAWTRADE_PORT"] = str(args.port)

    if args.sim:
        os.environ["KLAWTRADE_FORCE_SIM"] = "1"

    from src.main import main
    main()


def _prompt(question: str, default: str = "") -> str:
    """Prompt the user with a default value."""
    suffix = f" [{default}]" if default else ""
    answer = input(f"  {question}{suffix}: ").strip()
    return answer or default


def _prompt_yn(question: str, default: bool = True) -> bool:
    """Prompt for yes/no."""
    hint = "Y/n" if default else "y/N"
    answer = input(f"  {question} ({hint}): ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def _run_wizard(config_path: Path, env_path: Path) -> None:
    """Interactive setup wizard for first-time users."""
    import yaml

    print("\n--- KlawTrade Setup Wizard ---\n")

    # Starting capital
    capital = _prompt("Starting capital", "100000")
    try:
        capital_float = float(capital)
    except ValueError:
        capital_float = 100000.0

    # Risk tolerance
    print("\n  Risk profile:")
    print("    1) Conservative (2% daily loss, 10% drawdown)")
    print("    2) Moderate (3% daily loss, 15% drawdown) [default]")
    print("    3) Aggressive (5% daily loss, 25% drawdown)")
    risk_choice = _prompt("Choose risk profile", "2")

    risk_profiles = {
        "1": {"max_daily_loss_pct": 0.02, "max_drawdown_pct": 0.10, "max_weekly_loss_pct": 0.05},
        "2": {"max_daily_loss_pct": 0.03, "max_drawdown_pct": 0.15, "max_weekly_loss_pct": 0.07},
        "3": {"max_daily_loss_pct": 0.05, "max_drawdown_pct": 0.25, "max_weekly_loss_pct": 0.12},
    }
    risk = risk_profiles.get(risk_choice, risk_profiles["2"])

    # Dashboard port
    port = _prompt("Dashboard port", "8080")
    try:
        port_int = int(port)
    except ValueError:
        port_int = 8080

    # Alpaca keys
    print()
    use_alpaca = _prompt_yn("Do you have Alpaca API keys for paper trading?", default=False)
    alpaca_key = ""
    alpaca_secret = ""
    if use_alpaca:
        alpaca_key = _prompt("Alpaca API key", "")
        alpaca_secret = _prompt("Alpaca secret key", "")

    # Always-on mode
    always_on = _prompt_yn("Enable always-on mode? (trade outside market hours in simulation)", default=True)

    # Write config
    config = {
        "system": {
            "mode": "paper",
            "timezone": "US/Eastern",
            "heartbeat_interval_seconds": 30,
            "always_on": always_on,
        },
        "portfolio": {"starting_capital": capital_float},
        "risk": risk,
        "dashboard": {"enabled": True, "port": port_int},
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Write .env
    env_lines = [
        f"ALPACA_API_KEY={alpaca_key or 'your_paper_key_here'}",
        f"ALPACA_SECRET_KEY={alpaca_secret or 'your_paper_secret_here'}",
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets",
        "",
        "ANTHROPIC_API_KEY=your_claude_key_here",
        "",
        "LOG_LEVEL=INFO",
        f"DASHBOARD_PORT={port_int}",
    ]
    with open(env_path, "w") as f:
        f.write("\n".join(env_lines) + "\n")

    print(f"\n  Config written to {config_path}")
    print(f"  Environment written to {env_path}")


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize a new KlawTrade project directory."""
    target = Path(args.directory).resolve()
    target.mkdir(parents=True, exist_ok=True)

    print(f"Initializing KlawTrade in {target}")

    # Find the package's bundled config
    pkg_root = Path(__file__).parent.parent

    config_dst = target / "config"
    env_dst = target / ".env"

    # Create data and logs dirs
    for d in ("data", "logs"):
        p = target / d
        p.mkdir(exist_ok=True)
        print(f"  Created {p}/")

    # Run interactive wizard or copy defaults
    if sys.stdin.isatty() and not args.quick:
        _run_wizard(config_dst / "settings.yaml", env_dst)
    else:
        # Non-interactive: copy defaults
        config_src = pkg_root / "config"
        if config_src.exists() and not config_dst.exists():
            shutil.copytree(config_src, config_dst)
            print(f"  Created {config_dst}/")

        env_src = pkg_root / ".env.example"
        if env_src.exists() and not env_dst.exists():
            shutil.copy2(env_src, env_dst)
            print(f"  Created {env_dst}")

    print(f"\nKlawTrade initialized in {target}")
    print("\nTo start trading:")
    print(f"  cd {target}")
    print("  klawtrade start")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a historical backtest."""
    from src.backtesting import BacktestEngine

    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        # Fall back to default watchlist from config
        from src.config import load_config
        config = load_config(Path(args.config) if args.config else None)
        symbols = config.strategy.universe.watchlist

    # Parse strategies
    valid_strategies = ("momentum", "mean_reversion")
    if args.strategy == "all":
        strategy_names = list(valid_strategies)
    else:
        if args.strategy not in valid_strategies:
            print(
                f"Error: Unknown strategy '{args.strategy}'. "
                f"Choose from: {', '.join(valid_strategies)}, all"
            )
            sys.exit(1)
        strategy_names = [args.strategy]

    print(f"\n  Symbols:    {', '.join(symbols)}")
    print(f"  Period:     {args.start_date} to {args.end_date}")
    print(f"  Capital:    ${args.capital:,.2f}")
    print(f"  Strategies: {', '.join(strategy_names)}")

    try:
        engine = BacktestEngine(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            starting_capital=args.capital,
            strategy_names=strategy_names,
        )
        metrics = engine.run()
        print(metrics.summary_report())
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nBacktest failed: {e}")
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    """Check if KlawTrade is running and show basic status."""
    import urllib.request
    import json

    port = args.port or 8080
    url = f"http://localhost:{port}/api/state"

    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read())
            equity = data.get("total_equity", 0)
            cash = data.get("cash", 0)
            daily = data.get("daily_pnl", 0)
            positions = len(data.get("positions", []))
            halted = data.get("circuit_breaker_active", False)

            status = "HALTED" if halted else "RUNNING"
            print(f"KlawTrade is {status}")
            print(f"  Equity:    ${equity:,.2f}")
            print(f"  Cash:      ${cash:,.2f}")
            print(f"  Daily P&L: ${daily:,.2f}")
            print(f"  Positions: {positions}")
    except Exception:
        print("KlawTrade is not running (could not reach dashboard)")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="klawtrade",
        description="KlawTrade -- Autonomous Trading System",
    )
    parser.add_argument(
        "--version", action="version", version=f"klawtrade {_get_version()}"
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # klawtrade start
    p_start = sub.add_parser("start", help="Start the trading system")
    p_start.add_argument(
        "-c", "--config", type=Path, help="Path to settings.yaml"
    )
    p_start.add_argument(
        "-p", "--port", type=int, help="Dashboard port (default: 8080)"
    )
    p_start.add_argument(
        "--sim", action="store_true",
        help="Force simulation mode (ignore Alpaca keys)",
    )
    p_start.set_defaults(func=cmd_start)

    # klawtrade init
    p_init = sub.add_parser("init", help="Initialize project directory")
    p_init.add_argument(
        "directory", nargs="?", default=".",
        help="Target directory (default: current)",
    )
    p_init.add_argument(
        "-q", "--quick", action="store_true",
        help="Skip interactive wizard, use defaults",
    )
    p_init.set_defaults(func=cmd_init)

    # klawtrade backtest
    p_bt = sub.add_parser("backtest", help="Run a historical backtest")
    p_bt.add_argument(
        "--start-date", required=True,
        help="Start date (YYYY-MM-DD)",
    )
    p_bt.add_argument(
        "--end-date", required=True,
        help="End date (YYYY-MM-DD)",
    )
    p_bt.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols (default: config watchlist)",
    )
    p_bt.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Starting capital (default: 100000)",
    )
    p_bt.add_argument(
        "--strategy", type=str, default="all",
        choices=["momentum", "mean_reversion", "all"],
        help="Strategy to backtest (default: all)",
    )
    p_bt.add_argument(
        "-c", "--config", type=str, default=None,
        help="Path to settings.yaml",
    )
    p_bt.set_defaults(func=cmd_backtest)

    # klawtrade status
    p_status = sub.add_parser("status", help="Check if KlawTrade is running")
    p_status.add_argument(
        "-p", "--port", type=int, help="Dashboard port to check"
    )
    p_status.set_defaults(func=cmd_status)

    return parser


def cli() -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    cli()
