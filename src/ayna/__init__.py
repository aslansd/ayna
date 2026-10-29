"""ayna -- know whether your inferred dynamics are real, before you interpret them.

ayna checks the outputs of a fitted data-driven dynamics model (LFADS, a
sequential autoencoder, a neural ODE, ...) for the three failure modes defined
in the Computation-through-Dynamics Toolkit: underfitting, feature invention
and dynamical misattribution. It runs on numpy, never trains anything, and
says what it could not check.
"""

__version__ = "0.3.0"

from .bundle import Bundle, BundleError, load_bundle  # noqa: E402
from .report import Check, Finding, Report  # noqa: E402
from .scan import Thresholds, scan  # noqa: E402
from .select import select  # noqa: E402

__all__ = [
    "Bundle",
    "BundleError",
    "Check",
    "Finding",
    "Report",
    "Thresholds",
    "load_bundle",
    "scan",
    "select",
    "__version__",
]
