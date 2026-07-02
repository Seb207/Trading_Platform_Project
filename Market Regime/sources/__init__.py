"""Data-source clients for the Market Regime Detector — one module per source.

    fred.py           FredClient           (free, no key, full history)
    refinitiv_rdp.py  RefinitivClient      (RDP API, desktop-session fallback;
                                            local Refinitiv Workspace must be
                                            running to authenticate)
    gpr.py            get_gpr_index()      (Geopolitical Risk index, free)
    cboe.py           get_putcall_ratio()  (CBOE archive 2006-19 + OCC splice)
    occ.py            get_occ_putcall(), get_occ_open_interest()  (free OCC API)
    common.py         shared raw-file cache + curl-based http_get

Import from the package root:
    from sources import FredClient, RefinitivClient, get_gpr_index, \
        get_putcall_ratio, get_occ_open_interest
"""
from .cboe import get_putcall_ratio
from .fred import FredClient
from .gpr import get_gpr_index
from .occ import get_occ_open_interest, get_occ_putcall
from .refinitiv_rdp import RefinitivClient

__all__ = [
    "FredClient",
    "RefinitivClient",
    "get_gpr_index",
    "get_putcall_ratio",
    "get_occ_putcall",
    "get_occ_open_interest",
]
