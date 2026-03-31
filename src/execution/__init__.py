"""Execution layer – broker factory and order management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.execution.broker import Broker
from src.execution.sim_broker import SimBroker

if TYPE_CHECKING:
    from src.config import BrokerConfig

logger = logging.getLogger(__name__)

# Registry of supported broker providers.
# Each entry maps a provider name to (module_path, class_name, keys_check_fn_name).
_BROKER_REGISTRY: dict[str, tuple[str, str, str]] = {
    "alpaca": (
        "src.execution.alpaca_broker",
        "AlpacaBroker",
        "alpaca_keys_present",
    ),
    "ibkr": (
        "src.execution.ibkr_broker",
        "IBKRBroker",
        "ibkr_keys_present",
    ),
    "coinbase": (
        "src.execution.coinbase_broker",
        "CoinbaseBroker",
        "coinbase_keys_present",
    ),
    "binance": (
        "src.execution.binance_broker",
        "BinanceBroker",
        "binance_keys_present",
    ),
    "kraken": (
        "src.execution.kraken_broker",
        "KrakenBroker",
        "kraken_keys_present",
    ),
    "tradier": (
        "src.execution.tradier_broker",
        "TradierBroker",
        "tradier_keys_present",
    ),
}

SUPPORTED_BROKERS = list(_BROKER_REGISTRY.keys())


def create_broker(provider: str = "alpaca", paper: bool = True) -> Broker:
    """Instantiate a broker by provider name.

    Falls back to SimBroker if API keys are missing or the provider's
    dependency is not installed.
    """
    if provider not in _BROKER_REGISTRY:
        logger.warning(
            "Unknown broker provider '%s'. Supported: %s. Falling back to SimBroker.",
            provider,
            ", ".join(SUPPORTED_BROKERS),
        )
        return SimBroker()

    module_path, class_name, keys_fn_name = _BROKER_REGISTRY[provider]

    try:
        import importlib

        mod = importlib.import_module(module_path)
        keys_check = getattr(mod, keys_fn_name)

        if not keys_check():
            logger.warning(
                "%s API keys not configured. Falling back to SimBroker. "
                "See docs: https://klawtrade.com/docs/%s",
                provider.upper(),
                provider,
            )
            return SimBroker()

        broker_cls = getattr(mod, class_name)
        broker = broker_cls(paper=paper)
        logger.info("Broker initialised: %s (paper=%s)", provider, paper)
        return broker

    except ImportError as e:
        logger.warning(
            "Could not import %s broker (missing dependency: %s). "
            "Install it with: pip install klawtrade[%s]. Falling back to SimBroker.",
            provider,
            e,
            provider,
        )
        return SimBroker()
    except Exception as e:
        logger.error(
            "Failed to initialise %s broker: %s. Falling back to SimBroker.",
            provider,
            e,
        )
        return SimBroker()


def broker_from_config(config: BrokerConfig) -> Broker:
    """Create a broker instance from a BrokerConfig object."""
    return create_broker(provider=config.provider, paper=config.paper)
