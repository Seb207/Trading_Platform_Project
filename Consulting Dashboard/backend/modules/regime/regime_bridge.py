"""
Bridge to Market Regime/{factor_schema,similarity_engine,validation}.py.

Adds Market Regime/ to sys.path so those modules can be imported without
modifying the original project (same pattern as
backend/modules/research/arxiv_bridge.py for Research_LLM/).
"""
import sys
from functools import lru_cache


def _add_market_regime_to_path() -> None:
    from backend.config import MARKET_REGIME_DIR
    path_str = str(MARKET_REGIME_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@lru_cache(maxsize=1)
def get_regime_modules():
    """Return (similarity_engine, validation, factor_schema) modules.

    Imported lazily (not at module load time) so numpy/pandas/scipy are
    only required once a regime endpoint is actually hit.
    """
    _add_market_regime_to_path()
    import factor_schema       # type: ignore[import]
    import similarity_engine   # type: ignore[import]
    import validation          # type: ignore[import]
    return similarity_engine, validation, factor_schema
