# Core recommender implementations (always available)
from tie.recommender.bpr_recommender import BPRRecommender  # type: ignore
from tie.recommender.factorization_recommender import FactorizationRecommender  # type: ignore
from tie.recommender.recommender import Recommender  # type: ignore
from tie.recommender.top_items_recommender import TopItemsRecommender  # type: ignore
from tie.recommender.wals_recommender import WalsRecommender  # type: ignore

# Optional implementations that depend on the 'implicit' library.
# Guard imports so the package remains importable without native build toolchain.
try:  # pragma: no cover - optional dependency
    from tie.recommender.implicit_bpr_recommender import (  # type: ignore
        ImplicitBPRRecommender,
    )
except Exception:  # pragma: no cover
    ImplicitBPRRecommender = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from tie.recommender.implicit_wals_recommender import (  # type: ignore
        ImplicitWalsRecommender,
    )
except Exception:  # pragma: no cover
    ImplicitWalsRecommender = None  # type: ignore

__all__ = [
    "FactorizationRecommender",
    "BPRRecommender",
    "WalsRecommender",
    "TopItemsRecommender",
    "Recommender",
]

# Expose optional classes only if they were imported successfully
if ImplicitBPRRecommender is not None:  # pragma: no cover
    __all__.append("ImplicitBPRRecommender")
if ImplicitWalsRecommender is not None:  # pragma: no cover
    __all__.append("ImplicitWalsRecommender")
