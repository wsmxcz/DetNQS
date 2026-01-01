# file: detnqs/__init__.py

"""
detnqs: Local-Energy Variational Evolution with deterministic Refinement.
A scientific computing package for Fock-space quantum chemistry.
"""

# Import the core components from the core.py module to make them
# available at the top level of the package.
# For example, users can now access the IntCtx class via `detnqs.IntCtx`.
from .core import (
    IntCtx,
    gen_fci_dets,
    gen_excited_dets,
    get_ham_diag,
    get_ham_conn,
    get_ham_block,
    get_ham_conn_amp
)

from . import models
from . import system

# It's also good practice to define a package version here.
__version__ = "0.1.0"

# You can also define __all__ to control `from detnqs import *` behavior.
__all__ = [
    "IntCtx",
    "gen_fci_dets",
    "gen_excited_dets",
    "get_ham_diag",
    "get_ham_conn",
    "get_ham_block",
    "get_ham_conn_amp",
    "COOMatrix",
    "SpaceRep",
    "get_ham_proxy",
]