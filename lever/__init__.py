# file: lever/__init__.py

"""
LEVER: Local-Energy Variational Evolution with deterministic Refinement.
A scientific computing package for Fock-space quantum chemistry.
"""

# Import the core components from the core.py module to make them
# available at the top level of the package.
# For example, users can now access the IntCtx class via `lever.IntCtx`.
from .core import (
    IntCtx,
    gen_fci_dets,
    gen_excited_dets,
    get_ham_diag,
    get_ham_conns_SS,
    get_ham_conns_SC,
    get_ham_conns_ST,
    get_ham_conns_SSSC,
)

from .engine.hamiltonian import (
    HamOp,
    SpaceRep,
    get_ham_proxy,
)

from . import models
from . import engine

# It's also good practice to define a package version here.
__version__ = "0.1.0"

# You can also define __all__ to control `from lever import *` behavior.
__all__ = [
    "IntCtx",
    "gen_fci_dets",
    "gen_excited_dets",
    "get_ham_diag",
    "get_ham_conns_SS",
    "get_ham_conns_SC",
    "get_ham_conns_ST",
    "get_ham_conns_SSSC",
    "HamOp",
    "SpaceRep",
    "get_ham_proxy",
]