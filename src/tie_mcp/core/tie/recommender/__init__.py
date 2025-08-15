# Core recommender implementations.
# Each import is individually guarded so environments without heavy optional
# ML dependencies (tensorflow / keras / implicit) can still import the package.
from .recommender import Recommender  # type: ignore
from .top_items_recommender import TopItemsRecommender  # type: ignore

# TensorFlow / Keras dependent recommenders
try:  # pragma: no cover - optional dependency
    from .factorization_recommender import FactorizationRecommender  # type: ignore
except ImportError:  # pragma: no cover
    FactorizationRecommender = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .wals_recommender import WalsRecommender  # type: ignore
except ImportError:  # pragma: no cover
    WalsRecommender = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .bpr_recommender import BPRRecommender  # type: ignore
except ImportError:  # pragma: no cover
    BPRRecommender = None  # type: ignore

# Optional implementations that depend on the 'implicit' library (native build).
try:  # pragma: no cover - optional dependency
    from .implicit_bpr_recommender import ImplicitBPRRecommender  # type: ignore
except Exception:  # pragma: no cover
    ImplicitBPRRecommender = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .implicit_wals_recommender import ImplicitWalsRecommender  # type: ignore
except Exception:  # pragma: no cover
    ImplicitWalsRecommender = None  # type: ignore

__all__ = [
    "Recommender",
    "TopItemsRecommender",
]

if FactorizationRecommender is not None:  # pragma: no cover
    __all__.append("FactorizationRecommender")
if WalsRecommender is not None:  # pragma: no cover
    __all__.append("WalsRecommender")
if BPRRecommender is not None:  # pragma: no cover
    __all__.append("BPRRecommender")
if ImplicitBPRRecommender is not None:  # pragma: no cover
    __all__.append("ImplicitBPRRecommender")
if ImplicitWalsRecommender is not None:  # pragma: no cover
    __all__.append("ImplicitWalsRecommender")
