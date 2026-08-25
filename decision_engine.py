from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

from debug_json import debug_json as _dj
from feasibility_utils import SLOT_CLOCK_HOUR_MINUTE
from shop_planning import (
    FlavorCategory,
    ShopProfile,
    TrafficAdapter,
    SnsAdapter,
    TasteAuthorityEngine,
    predict_wait_time,
)

# ---------------------------------------------------------------
# Phase A1: feature registry, phi(), and trip-frozen z-scoring
# ---------------------------------------------------------------

from preference_features import (
    FEATURE_NAMES as FEATURE_NAMES,
    FEATURE_NAME_TO_INDEX as FEATURE_NAME_TO_INDEX,
    TASTE_INDICES as TASTE_INDICES,
    CONTEXT_INDICES as CONTEXT_INDICES,
)

BLOCK_INDEX = {
    "taste": TASTE_INDICES,
    "context": CONTEXT_INDICES,
}


def _safe_float(value: object, default: float = 0.0) -> float:
    """Return float(value) if possible, otherwise default."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def phi(item: object, ctx: dict | None = None) -> np.ndarray:
    """
    Extract raw 6-dimensional features for a candidate shop.

    Parameters
    ----------
    item : object
        Usually a ShopProfile (or any object exposing needed attributes).
    ctx : dict, optional
        Context dictionary may contain:
            - preferred_tags: iterable[str]
            - travel_minutes: float
        Missing attributes are replaced with sensible defaults.

    Returns
    -------
    np.ndarray of shape (6,) with raw feature values.
    """
    ctx = ctx or {}

    # ---- taste-oriented block (w_T) ----
    # 1) cuisine_match: Jaccard similarity between requested tags and shop tags
    wanted_tags = {str(t).lower() for t in ctx.get("preferred_tags", [])}
    shop_tags = {str(t).lower() for t in getattr(item, "tags", [])}
    if not wanted_tags or not shop_tags:
        cuisine_match = 0.0
    else:
        inter = len(wanted_tags & shop_tags)
        union = len(wanted_tags | shop_tags)
        cuisine_match = inter / union if union else 0.0

    # 2) fame_touristy: rises with review count and authority medal presence
    auth = getattr(item, "authority_data", None)
    review_count_raw = None
    if auth is not None and getattr(auth, "review_count", None) is not None:
        review_count_raw = float(auth.review_count)
    else:
        review_count_raw = getattr(item, "review_count", None)
    review_count = 0.0 if review_count_raw is None else max(0.0, float(review_count_raw))
    medal = str(getattr(auth, "tablelog_medal", "") or "")
    fame = min(1.0, review_count / 300.0)
    if medal:
        fame = min(1.0, fame + 0.3)

    # 3) heaviness: composite of flavor intensity and portion strictness
    flavor = _safe_float(getattr(item, "flavor_intensity", None))
    portion = _safe_float(getattr(item, "portion_strictness", None), 0.5)
    heaviness = min(1.0, flavor * 0.6 + portion * 0.4)

    # ---- situational-cost block (θ) ----
    # 4) travel_min: candidate-specific or scalar fallback
    travel_map = ctx.get("travel_minutes_map") or {}
    travel_default = ctx.get("travel_minutes", getattr(item, "default_travel_minutes", 15))
    travel_min = float(travel_map.get(str(getattr(item, "name", "")), travel_default))

    # 5) price_level: direct attribute if present, otherwise contextual default
    price_map = ctx.get("price_level_map") or {}
    price_default = ctx.get("price_level", getattr(item, "price_level", 2.5))
    price_level = _safe_float(price_map.get(str(getattr(item, "name", "")), price_default), 2.5)

    # 6) queue_wait: typical queue length in minutes
    queue_wait = _safe_float(getattr(item, "base_wait_minutes", None))

    return np.array(
        [cuisine_match, fame, heaviness, travel_min, price_level, queue_wait],
        dtype=float,
    )


def compute_trip_frozen_scaling(
    candidates: list[object],
    ctx: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute and freeze per-dimension mean/std on the initial candidate pool.

    Returns (means, stds).  Stored scaling later used by `z_score`.
    """
    if not candidates:
        means = np.zeros(6)
        stds = np.ones(6)
        return means, stds

    matrix = np.array([phi(c, ctx) for c in candidates], dtype=float)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    # Avoid division by zero when a dimension has no variance.
    stds[stds == 0.0] = 1.0
    return means, stds


def z_score(item: object, ctx: dict | None, scaling: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Apply trip-frozen standardization to a single item.

    scaling = (means, stds) previously computed via `compute_trip_frozen_scaling`.
    """
    means, stds = scaling
    raw = phi(item, ctx)
    return (raw - means) / stds


def freeze_candidate_scaling(
    candidates: list[object],
    ctx: dict | None = None,
) -> dict[str, object]:
    """
    Freeze the candidate universe for the current turn.

    Computes mean/std once over the given feasible pool and returns a
    dict containing the raw means/stds as plain lists.  All later
    S̃₀ calculations (including hypothetical B_o) must reuse this
    scaling without recomputing it.

    Returns:
        {"candidates": list[object],
         "means": list[float],
         "stds": list[float],
         "ctx": dict|None}
    """
    means, stds = compute_trip_frozen_scaling(candidates, ctx)
    return {
        "candidates": list(candidates),
        "means": [float(x) for x in means],
        "stds": [float(x) for x in stds],
        "ctx": ctx,
    }


def rerank_by_posterior(
    ranked: list["RankedShop"],
    mu: list[float] | tuple[float, ...] | np.ndarray | None,
    ctx: dict | None = None,
) -> list["RankedShop"]:
    """
    Phase B2: compute S = S0 + muᵀ φ(x, c_s) and reorder `ranked` by S.

    When `mu` is None or all zeros, returns the original order unchanged
    (exact fallback to the pure S0 ranking).
    """
    if ranked is None or len(ranked) == 0:
        return ranked
    if mu is None:
        return ranked
    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    if mu_arr.ndim != 1 or mu_arr.shape[0] != 6:
        raise ValueError("mu must be a 6-dimensional vector")
    if float(np.max(np.abs(mu_arr))) == 0.0:
        # μ=0 → must preserve the original engine order exactly
        return ranked

    ctx = ctx or {}
    scored = [
        (
            r,
            float(r.final_score)
            + float(np.dot(mu_arr, np.asarray(phi(r.shop, ctx), dtype=float))),
        )
        for r in ranked
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [r for r, _ in scored]


# ---------------------------------------------------------------
# (End of Phase A1 feature layer)
# ---------------------------------------------------------------


@dataclass
class UserPreference:
    preferred_tags: list[str] = field(default_factory=list)
    avoid_tags: list[str] = field(default_factory=list)
    dietary_preference: str = "regular"
    max_wait_minutes: int = 35
    health_budget_limit: float = 1.2
    prefers_driving: bool = False
    max_total_minutes: int = 120
    max_budget_impact: float = 0.75


@dataclass
class UserMinefield:
    blocked_shop_names: set[str] = field(default_factory=set)
    blocked_tags: set[str] = field(default_factory=set)


@dataclass
class WeightProfile:
    trust_bias: float = 0.8
    preference_bias: float = 0.1
    logistics_bias: float = 0.1

    @classmethod
    def trust_first(cls) -> "WeightProfile":
        return cls(trust_bias=0.8, preference_bias=0.1, logistics_bias=0.1)


class OptimizationMode(str, Enum):
    BALANCED = "BALANCED"
    TASTE_MAX = "TASTE_MAX"
    RIGHT_NOW = "RIGHT_NOW"


class SelectionStrategy(str, Enum):
    FOODIE_STRATEGY = "FOODIE_STRATEGY"
    TOURIST_STRATEGY = "TOURIST_STRATEGY"


@dataclass
class RankedShop:
    shop: ShopProfile
    final_score: float
    preference_match_score: float = 0.0
    rank_note: str = ""
    is_wildcard: bool = False
    filter_reason: str = ""
    top_3_reasons: list[str] = field(default_factory=list)
    # 平民美食／非獎項熱店加成（排行榜與 UI 【老饕私藏】）
    insider_pick: bool = False


@dataclass
class RejectedShop:
    shop_name: str
    estimated_score: float
    reason: str


@dataclass
class ScheduleNode:
    title: str
    start_at: datetime
    end_at: datetime
    note: str = ""


@dataclass
class SynthesisResult:
    nodes: list[ScheduleNode]
    backup_nodes: list[str]
    warnings: list[str] = field(default_factory=list)
    solver_audit_log: list[str] = field(default_factory=list)
    graph_debug_traces: list[str] = field(default_factory=list)
    rollback_triggered: bool = False


@dataclass
class GraphNode:
    node_id: str
    shop_name: str
    slot_index: int
    start_time: datetime
    end_time: datetime
    final_score: float
    tags: tuple[str, ...] = ()
    ideal_duration_minutes: int = 0
    actual_duration_minutes: int = 0


@dataclass
class GraphEdge:
    from_node_id: str
    to_node_id: str
    weight: float


@dataclass
class SpatioTemporalGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    debug_traces: list[str] = field(default_factory=list)


@dataclass
class HealthBudgetTracker:
    spent: float = 0.0

    def add(self, delta: float) -> None:
        self.spent += max(0.0, delta)

    def is_budget_exceeded(self, current_limit: float) -> bool:
        return self.spent > max(0.0, current_limit)


class MinefieldFilter:
    @staticmethod
    def hard_drop(shops: list[ShopProfile], minefield: UserMinefield) -> tuple[list[ShopProfile], list[tuple[str, str]]]:
        kept: list[ShopProfile] = []
        dropped: list[tuple[str, str]] = []
        for s in shops:
            if s.name in minefield.blocked_shop_names:
                dropped.append((s.name, "blocked_shop_name"))
                continue
            matched_tag = next((t for t in s.tags if t in minefield.blocked_tags), None)
            if matched_tag:
                dropped.append((s.name, f"blocked_tag:{matched_tag}"))
                continue
            kept.append(s)
        return kept, dropped


class SourceWeightManager:
    DEFAULT_WEIGHTS_BY_REGION: dict[str, dict[str, float]] = {
        "jp": {"tablelog": 0.9, "sns": 0.7, "google": 0.4},
        "tw": {"google": 0.9, "dcard": 0.7, "threads": 0.55, "sns": 0.45},
        "default": {"google": 0.7, "sns": 0.5},
    }

    @classmethod
    def weighted_trust_score(cls, source_scores: dict[str, float], region: str = "default") -> float:
        weights = cls.DEFAULT_WEIGHTS_BY_REGION.get(region, cls.DEFAULT_WEIGHTS_BY_REGION["default"])
        numerator = 0.0
        denominator = 0.0
        for source, weight in weights.items():
            score = source_scores.get(source, None)
            if score is None:
                continue
            numerator += score * weight
            denominator += weight
        if denominator <= 0:
            return 0.0
        return numerator / denominator


class DensityScanner:
    """
    Geographic scarcity scanner.
    alpha (Isolation_Factor) in [0,1]: higher means fewer alternatives nearby.
    """

    @staticmethod
    def scan_isolation_factor(nearby_alternative_count: int, scarcity_threshold: int = 5) -> float:
        count = max(0, nearby_alternative_count)
        alpha = 1.0 - min(1.0, count / max(1, scarcity_threshold))
        return round(alpha, 4)


class MinefieldManager:
    # keyword -> semantic mine category
    KEYWORD_MAP: dict[str, str] = {
        "臭臉": "SERVICE_MINE",
        "態度": "SERVICE_MINE",
        "煙味重": "ENVIRONMENT_MINE",
        "賄賂送禮": "BRIBE_MINE",
        "評論換": "BRIBE_MINE",
    }
    # Keep hard-drop conservative to avoid over-filtering potentially good shops.
    HARD_MINES: set[str] = {"BRIBE_MINE"}

    @classmethod
    def scan_for_mines(cls, reviews: list[str]) -> tuple[set[str], list[str]]:
        mines: set[str] = set()
        evidences: list[str] = []
        for r in reviews:
            for kw, mine in cls.KEYWORD_MAP.items():
                if kw in r:
                    mines.add(mine)
                    evidences.append(f"{kw}: {r}")
        return mines, evidences

    @classmethod
    def hard_mines_triggered(cls, reviews: list[str], min_hits: int = 2) -> set[str]:
        mine_hits: dict[str, int] = {}
        for r in reviews:
            for kw, mine in cls.KEYWORD_MAP.items():
                if kw in r:
                    mine_hits[mine] = mine_hits.get(mine, 0) + 1
        return {m for m, c in mine_hits.items() if m in cls.HARD_MINES and c >= min_hits}


class ScoringEngine:
    # Layer-2 risk budget coefficients:
    # - lambda_queue penalizes intrinsically long queue shops
    # - lambda_buffer penalizes low slack against expected transport buffer
    LAMBDA_QUEUE: float = 18.0
    LAMBDA_BUFFER: float = 22.0
    # Fame damping: Taste_Score - (Accolade_Bonus × marketing_noise_score)
    UNDERDOG_MODE: bool = False

    @staticmethod
    def accolade_bonus(shop: ShopProfile) -> float:
        """
        Aggregate marketing / award lift used by authority-style scoring (0..~45).
        UNDERDOG_MODE subtracts (this × marketing_noise_score) from the taste score.
        """
        auth = shop.authority_data
        medal = auth.tablelog_medal or ""
        bonus = 0.0
        if auth.michelin_star > 0:
            bonus += min(24.0, float(auth.michelin_star) * 8.0)
        if "百名店" in medal:
            bonus += 18.0
        elif "金" in medal:
            bonus += 8.0
        elif "銀" in medal:
            bonus += 6.0
        elif "銅" in medal:
            bonus += 4.0
        bonus += min(10.0, len(auth.chef_lineage) * 3.5)
        return min(45.0, bonus)

    @staticmethod
    def fame_damped_final_score(taste_score: float, shop: ShopProfile, *, underdog_mode: bool) -> float:
        """Final_Score = Taste_Score - (Accolade_Bonus × marketing_noise_score)."""
        if not underdog_mode:
            return float(taste_score)
        noise = float(getattr(shop, "marketing_noise_score", 0.45) or 0.45)
        noise = max(0.0, min(1.0, noise))
        damp = ScoringEngine.accolade_bonus(shop) * noise
        return max(0.0, float(taste_score) - damp)

    @staticmethod
    def normalize_weights(profile: WeightProfile) -> WeightProfile:
        total = profile.trust_bias + profile.preference_bias + profile.logistics_bias
        if total <= 0:
            return WeightProfile.trust_first()
        return WeightProfile(
            trust_bias=profile.trust_bias / total,
            preference_bias=profile.preference_bias / total,
            logistics_bias=profile.logistics_bias / total,
        )

    @staticmethod
    def preference_score(shop: ShopProfile, pref: UserPreference) -> float:
        score = 0.0
        tags = set(shop.tags)
        score += 0.20 * len(tags.intersection(pref.preferred_tags))
        score -= 0.25 * len(tags.intersection(pref.avoid_tags))
        if pref.dietary_preference and pref.dietary_preference.lower() in [t.lower() for t in shop.tags]:
            score += 0.10
        return max(0.0, min(1.0, 0.5 + score))

    @staticmethod
    def logistics_score(shop: ShopProfile, pref: UserPreference) -> float:
        # Lower queue expectation yields better logistics score.
        queue_ratio = min(1.0, shop.base_wait_minutes / max(1, pref.max_wait_minutes))
        return max(0.0, 1.0 - queue_ratio)

    @staticmethod
    def score(shop: ShopProfile, pref: UserPreference, profile: WeightProfile) -> tuple[float, float]:
        w = ScoringEngine.normalize_weights(profile)
        source_trust = SourceWeightManager.weighted_trust_score(shop.source_scores, region=shop.region)
        trust = max(0.0, min(1.0, source_trust if source_trust > 0 else shop.trust_score))
        pscore = ScoringEngine.preference_score(shop, pref)
        lscore = ScoringEngine.logistics_score(shop, pref)
        final = (trust * w.trust_bias + pscore * w.preference_bias + lscore * w.logistics_bias) * 100.0
        return round(final, 2), round(pscore, 4)

    @staticmethod
    def optimized_intensity_impact(shop: ShopProfile) -> float:
        base = max(0.0, min(1.0, shop.base_health_impact))
        custom = max(0.0, min(1.0, shop.customization_score))
        # Reframed as taste-intensity burden: high customization can reduce burden.
        optimized = base * (1.0 - 0.6 * custom)
        return round(max(0.0, min(1.0, optimized)), 4)

    @staticmethod
    def optimized_health_impact(shop: ShopProfile) -> float:
        # Backward-compatible alias. Prefer optimized_intensity_impact.
        return ScoringEngine.optimized_intensity_impact(shop)

    @staticmethod
    def risk_penalty(
        *,
        queue_risk: float,
        travel_buffer_gap: float,
        lambda_queue: float | None = None,
        lambda_buffer: float | None = None,
    ) -> float:
        lq = ScoringEngine.LAMBDA_QUEUE if lambda_queue is None else float(lambda_queue)
        lb = ScoringEngine.LAMBDA_BUFFER if lambda_buffer is None else float(lambda_buffer)
        qr = max(0.0, min(1.0, float(queue_risk)))
        tbg = max(0.0, min(1.0, float(travel_buffer_gap)))
        return (lq * qr) + (lb * tbg)

    @staticmethod
    def risk_adjusted_score(
        taste_score: float,
        *,
        queue_risk: float,
        travel_buffer_gap: float,
    ) -> float:
        penalty = ScoringEngine.risk_penalty(queue_risk=queue_risk, travel_buffer_gap=travel_buffer_gap)
        return max(0.0, float(taste_score) - penalty)

    @staticmethod
    def score_with_isolation(
        shop: ShopProfile,
        pref: UserPreference,
        profile: WeightProfile,
        isolation_factor: float,
        isolation_threshold: float = 0.7,
    ) -> tuple[float, float, WeightProfile]:
        # In sparse areas, trust-first policy dominates to minimize outage/sold-out risk.
        adaptive_profile = profile
        if isolation_factor >= isolation_threshold:
            adaptive_profile = WeightProfile(trust_bias=0.9, preference_bias=0.1, logistics_bias=0.0)
        score, pref_match = ScoringEngine.score(shop, pref, adaptive_profile)
        return score, pref_match, ScoringEngine.normalize_weights(adaptive_profile)


class UserPreferenceLearner:
    """
    Lightweight online linear learner (ElasticNet-style SGD) for 1-5 star feedback.
    Learns a personalized preference surface, then projects core coefficients back to WeightProfile.
    """

    def __init__(
        self,
        learning_rate: float = 0.03,
        l1_penalty: float = 0.001,
        l2_penalty: float = 0.003,
    ) -> None:
        self.feature_names = [
            "trust",
            "preference_match",
            "logistics",
            "health_friendly",
            "flavor_balance",
        ]
        self.weights: list[float] = [0.0 for _ in self.feature_names]
        self.bias: float = 3.0
        self.learning_rate = learning_rate
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

    def _feature_vector(self, shop: ShopProfile, pref: UserPreference) -> list[float]:
        source_trust = SourceWeightManager.weighted_trust_score(shop.source_scores, region=shop.region)
        trust = max(0.0, min(1.0, source_trust if source_trust > 0 else shop.trust_score))
        pscore = ScoringEngine.preference_score(shop, pref)
        lscore = ScoringEngine.logistics_score(shop, pref)
        health_friendly = 1.0 - ScoringEngine.optimized_health_impact(shop)
        flavor_balance = 1.0 - min(1.0, abs(shop.flavor_intensity - 0.65))
        return [trust, pscore, lscore, health_friendly, flavor_balance]

    def predict_stars(self, shop: ShopProfile, pref: UserPreference) -> float:
        x = self._feature_vector(shop, pref)
        raw = self.bias + sum(w * v for w, v in zip(self.weights, x))
        return max(1.0, min(5.0, raw))

    def personalized_taste_score(self, shop: ShopProfile, pref: UserPreference) -> float:
        stars = self.predict_stars(shop, pref)
        return round((stars - 1.0) / 4.0 * 100.0, 2)

    def partial_fit(self, samples: list[tuple[ShopProfile, int]], pref: UserPreference, epochs: int = 20) -> None:
        if not samples:
            return
        for _ in range(max(1, epochs)):
            for shop, stars in samples:
                y = float(max(1, min(5, stars)))
                x = self._feature_vector(shop, pref)
                pred = self.bias + sum(w * v for w, v in zip(self.weights, x))
                err = pred - y
                self.bias -= self.learning_rate * err
                for i, v in enumerate(x):
                    grad = err * v + (self.l2_penalty * self.weights[i]) + (self.l1_penalty * (1 if self.weights[i] >= 0 else -1))
                    self.weights[i] -= self.learning_rate * grad

    def to_weight_profile(self) -> WeightProfile:
        trust_w = abs(self.weights[0])
        pref_w = abs(self.weights[1])
        log_w = abs(self.weights[2])
        total = trust_w + pref_w + log_w
        if total <= 1e-9:
            return WeightProfile.trust_first()
        return WeightProfile(
            trust_bias=trust_w / total,
            preference_bias=pref_w / total,
            logistics_bias=log_w / total,
        )


class RankingEngine:
    @staticmethod
    def _has_authority_medal(shop: ShopProfile) -> bool:
        """米其林、Tabelog 百名店／獎牌等權威標記。"""
        auth = shop.authority_data
        if auth.michelin_star > 0:
            return True
        medal = (auth.tablelog_medal or "").strip()
        if not medal:
            return False
        if "百名店" in medal:
            return True
        if "米其林" in medal or "michelin" in medal.lower():
            return True
        for token in ("金賞", "銀賞", "銅賞", "金", "銀", "銅"):
            if token in medal:
                return True
        return False

    @staticmethod
    def _effective_star_rating(shop: ShopProfile) -> float:
        """0–5 星語意；優先 google_rating，否則由 source_scores 推估。"""
        gr = float(shop.google_rating or 0.0)
        if gr > 0.0:
            return gr
        g = float(shop.source_scores.get("google", 0.0)) * 5.0
        t = float(shop.source_scores.get("tablelog", 0.0)) * 5.0
        return max(g, t, 0.0)

    @staticmethod
    def _qualifies_low_key_bonus(shop: ShopProfile) -> bool:
        """
        評論量中等（非爆紅觀光店）、無權威獎牌、評分穩定 — 平民美食紅利。
        """
        rc = int(shop.authority_data.review_count or 0)
        if not (100 <= rc <= 300):
            return False
        if RankingEngine._has_authority_medal(shop):
            return False
        return RankingEngine._effective_star_rating(shop) > 4.2

    @staticmethod
    def _build_reasons(
        shop: ShopProfile,
        score: float,
        pref_match: float,
        alpha: float,
        strategy: SelectionStrategy,
    ) -> list[str]:
        source_trust = SourceWeightManager.weighted_trust_score(shop.source_scores, region=shop.region)
        trust = source_trust if source_trust > 0 else shop.trust_score
        logistics = max(0.0, 1.0 - min(1.0, shop.base_wait_minutes / 35.0))
        reasons = [
            f"綜合分數 {score:.1f}",
            f"信賴度 {trust:.2f}",
            f"偏好匹配 {pref_match:.2f}",
        ]
        if strategy == SelectionStrategy.FOODIE_STRATEGY:
            reasons[0] = f"個人化口味分 {score:.1f}"
        if logistics >= 0.6:
            reasons[-1] = f"排隊風險較低 ({shop.base_wait_minutes}m)"
        if alpha >= 0.7:
            reasons.append(f"高孤立區補償策略 alpha={alpha:.2f}")
        return reasons[:3]

    @staticmethod
    def generate_top_picks(
        shops: list[ShopProfile],
        preference: UserPreference,
        minefield: UserMinefield,
        weight_profile: WeightProfile | None = None,
        nearby_counts: dict[str, int] | None = None,
        isolation_threshold: float = 0.7,
        mode: OptimizationMode = OptimizationMode.BALANCED,
        health_tracker: HealthBudgetTracker | None = None,
        selection_strategy: SelectionStrategy = SelectionStrategy.TOURIST_STRATEGY,
        learner: UserPreferenceLearner | None = None,
        skip_semantic_mines: bool = False,
        shop_penalties: dict[str, float] | None = None,
        must_have_tags: list[str] | None = None,
        taste_max_blacklist: set[str] | None = None,
        appetite_light_mode: bool = False,
        underdog_mode: bool | None = None,
    ) -> tuple[list[RankedShop], list[RejectedShop]]:
        profile = weight_profile or WeightProfile.trust_first()
        nearby_counts = nearby_counts or {}
        shop_penalties = shop_penalties or {}
        effective_underdog = (
            bool(underdog_mode) if underdog_mode is not None else bool(ScoringEngine.UNDERDOG_MODE)
        )
        must_have = {t.strip().lower() for t in (must_have_tags or []) if t.strip()}
        taste_max_blacklist = {x.strip() for x in (taste_max_blacklist or set()) if x.strip()}
        filtered_by_tags: list[ShopProfile] = list(shops)
        # Score first so rejected list can include high-score shops.
        pre_scored: dict[str, tuple[float, float]] = {}
        for s in filtered_by_tags:
            alpha = DensityScanner.scan_isolation_factor(nearby_counts.get(s.name, 3))
            ok_diet, reason_diet, _warn_diet = s.metadata_verification(preference.dietary_preference)
            if not ok_diet:
                pre_scored[s.name] = (0.0, 0.0)
                setattr(s, "_hard_constraint_reason", f"DIETARY_CONSTRAINT:{reason_diet}")
                setattr(s, "_isolation_factor", alpha)
                setattr(s, "_adaptive_weights", WeightProfile(trust_bias=0.0, preference_bias=1.0, logistics_bias=0.0))
                continue
            if must_have:
                tag_set = {str(t).lower() for t in s.tags}
                # Any key tag match is enough for intent satisfaction.
                if not any(t in tag_set for t in must_have):
                    pre_scored[s.name] = (0.0, 0.0)
                    setattr(s, "_hard_constraint_reason", "TAG_MISMATCH")
                    setattr(s, "_isolation_factor", alpha)
                    setattr(s, "_adaptive_weights", WeightProfile(trust_bias=0.0, preference_bias=1.0, logistics_bias=0.0))
                    continue
            if mode == OptimizationMode.TASTE_MAX and s.name in taste_max_blacklist:
                pre_scored[s.name] = (0.0, 0.0)
                setattr(s, "_hard_constraint_reason", "TASTE_MAX_BLACKLIST")
                setattr(s, "_isolation_factor", alpha)
                setattr(s, "_adaptive_weights", WeightProfile(trust_bias=0.0, preference_bias=1.0, logistics_bias=0.0))
                continue
            if appetite_light_mode:
                ps = float(getattr(s, "portion_strictness", 0.5))
                if ps > 0.9 and not bool(getattr(s, "has_small_portion", False)):
                    pre_scored[s.name] = (0.0, 0.0)
                    setattr(s, "_hard_constraint_reason", "APPETITE_LIGHT_STRICT_PORTION")
                    setattr(s, "_isolation_factor", alpha)
                    setattr(s, "_adaptive_weights", WeightProfile(trust_bias=0.0, preference_bias=1.0, logistics_bias=0.0))
                    continue
            # FOODIE_STRATEGY: maximize personalized taste with hard constraints.
            if selection_strategy == SelectionStrategy.FOODIE_STRATEGY:
                service_minutes = int(s.base_wait_minutes) + int(s.avg_eat_minutes)
                budget_impact = ScoringEngine.optimized_intensity_impact(s)
                if service_minutes > preference.max_total_minutes:
                    pre_scored[s.name] = (0.0, 0.0)
                    setattr(s, "_hard_constraint_reason", f"TIME_HARD_CONSTRAINT({service_minutes}>{preference.max_total_minutes})")
                    setattr(s, "_isolation_factor", alpha)
                    setattr(s, "_adaptive_weights", WeightProfile(trust_bias=0.0, preference_bias=1.0, logistics_bias=0.0))
                    continue
                if budget_impact > preference.max_budget_impact:
                    pre_scored[s.name] = (0.0, 0.0)
                    setattr(s, "_hard_constraint_reason", f"BUDGET_HARD_CONSTRAINT({budget_impact:.2f}>{preference.max_budget_impact:.2f})")
                    setattr(s, "_isolation_factor", alpha)
                    setattr(s, "_adaptive_weights", WeightProfile(trust_bias=0.0, preference_bias=1.0, logistics_bias=0.0))
                    continue
                personalized = learner.personalized_taste_score(s, preference) if learner else 50.0
                if appetite_light_mode and getattr(s, "has_small_portion", False):
                    personalized *= 1.15
                personalized = ScoringEngine.fame_damped_final_score(
                    personalized, s, underdog_mode=effective_underdog
                )
                if RankingEngine._qualifies_low_key_bonus(s):
                    personalized *= 1.2
                    setattr(s, "_low_key_bonus", True)
                else:
                    setattr(s, "_low_key_bonus", False)
                pre_scored[s.name] = (max(0.0, round(personalized, 2)), ScoringEngine.preference_score(s, preference))
                setattr(s, "_isolation_factor", alpha)
                setattr(s, "_adaptive_weights", WeightProfile(trust_bias=0.0, preference_bias=1.0, logistics_bias=0.0))
                continue

            if mode == OptimizationMode.TASTE_MAX:
                taste = TasteAuthorityEngine.calculate_taste_ground_truth(s)
                score = float(taste["final_taste_ground_truth"])
                pref_match = ScoringEngine.preference_score(s, preference)
                adaptive_w = WeightProfile(trust_bias=0.0, preference_bias=0.0, logistics_bias=0.0)
            else:
                score, pref_match, adaptive_w = ScoringEngine.score_with_isolation(
                    s, preference, profile, isolation_factor=alpha, isolation_threshold=isolation_threshold
                )
                # TOURIST_STRATEGY: weighted loss balancing all factors.
                if selection_strategy == SelectionStrategy.TOURIST_STRATEGY:
                    source_trust = SourceWeightManager.weighted_trust_score(s.source_scores, region=s.region)
                    trust = max(0.0, min(1.0, source_trust if source_trust > 0 else s.trust_score))
                    logistics = ScoringEngine.logistics_score(s, preference)
                    intensity_friendly = 1.0 - ScoringEngine.optimized_intensity_impact(s)
                    composite_loss = (
                        (1.0 - trust) * 0.35
                        + (1.0 - pref_match) * 0.35
                        + (1.0 - logistics) * 0.20
                        + (1.0 - intensity_friendly) * 0.10
                    )
                    score = max(0.0, 100.0 * (1.0 - composite_loss))
                # Intensity-aware penalty when palate budget is tight.
                if health_tracker is not None and health_tracker.is_budget_exceeded(preference.health_budget_limit * 0.6):
                    impact = ScoringEngine.optimized_intensity_impact(s)
                    # Bigger intensity burden => larger penalty. Keep customizable shops viable.
                    score -= impact * 22.0
            # Negative feedback loop: penalize historically disliked shops.
            penalty = max(0.0, float(shop_penalties.get(s.name, 0.0)))
            score = max(0.0, score - penalty)
            # Hard feedback loop requirement:
            # if learner state contains "一蘭=1", force multiplicative penalty.
            if "一蘭" in s.name and penalty >= 18.0:
                score = score * 0.2
            if appetite_light_mode and getattr(s, "has_small_portion", False):
                score *= 1.15
            score = ScoringEngine.fame_damped_final_score(score, s, underdog_mode=effective_underdog)
            if RankingEngine._qualifies_low_key_bonus(s):
                score *= 1.2
                setattr(s, "_low_key_bonus", True)
            else:
                setattr(s, "_low_key_bonus", False)
            pre_scored[s.name] = (max(0.0, round(score, 2)), pref_match)
            setattr(s, "_isolation_factor", alpha)
            setattr(s, "_adaptive_weights", adaptive_w)
        filtered, dropped = MinefieldFilter.hard_drop(filtered_by_tags, minefield)
        # Semantic minefield scan (can hard-drop unusable shops).
        filtered_after_mines: list[ShopProfile]
        if skip_semantic_mines:
            filtered_after_mines = filtered
        else:
            semantic_dropped: list[tuple[str, str]] = []
            filtered_after_mines = []
            for s in filtered:
                review_texts = [str(x) for x in getattr(s, "_review_texts", [])]
                mines, evidences = MinefieldManager.scan_for_mines(review_texts)
                hard = MinefieldManager.hard_mines_triggered(review_texts, min_hits=2)
                if hard:
                    reason = f"UNUSABLE:{','.join(sorted(hard))}"
                    if evidences:
                        reason += f" ({evidences[0]})"
                    semantic_dropped.append((s.name, reason))
                    continue
                filtered_after_mines.append(s)
            dropped.extend(semantic_dropped)
        scored: list[RankedShop] = []
        for s in filtered_after_mines:
            final, pref_match = pre_scored[s.name]
            constraint_reason = getattr(s, "_hard_constraint_reason", "")
            if constraint_reason:
                dropped.append((s.name, constraint_reason))
                continue
            alpha = float(getattr(s, "_isolation_factor", 0.0))
            reasons = RankingEngine._build_reasons(s, final, pref_match, alpha, selection_strategy)
            insider = bool(getattr(s, "_low_key_bonus", False))
            insider_note = ""
            if insider:
                insider_note = "【老饕私藏】此店名氣較低，但味覺信號純粹，避開了權威獎項的行銷噪音"
            scored.append(
                RankedShop(
                    shop=s,
                    final_score=final,
                    preference_match_score=pref_match,
                    rank_note=insider_note,
                    insider_pick=insider,
                    top_3_reasons=reasons,
                )
            )

        scored.sort(key=lambda x: x.final_score, reverse=True)
        top = scored[:5]

        # Ensure 1 wildcard exists in top results.
        if top:
            wildcard_source = None
            if len(scored) > len(top):
                wildcard_source = scored[len(top)]
            elif len(top) >= 2:
                wildcard_source = top[-1]
            if wildcard_source:
                wildcard = RankedShop(
                    shop=wildcard_source.shop,
                    final_score=wildcard_source.final_score,
                    is_wildcard=True,
                    rank_note=getattr(wildcard_source, "rank_note", "") or "",
                    insider_pick=getattr(wildcard_source, "insider_pick", False),
                    top_3_reasons=list(getattr(wildcard_source, "top_3_reasons", []) or []),
                )
                if wildcard.shop.name not in {x.shop.name for x in top}:
                    if len(top) >= 5:
                        top[-1] = wildcard
                    else:
                        top.append(wildcard)
                else:
                    top[-1].is_wildcard = True
        # Add explicit trust-first note for top-1 low preference match.
        if top and top[0].preference_match_score < 0.45:
            trust_note = "此店被選中是因為其信賴分數極高，能大幅降低您的踩雷風險。"
            if top[0].rank_note:
                top[0].rank_note = f"{top[0].rank_note}；{trust_note}"
            else:
                top[0].rank_note = trust_note

        rejected: list[RejectedShop] = []
        for name, reason in dropped:
            score, _ = pre_scored.get(name, (0.0, 0.0))
            if "賄賂送禮" in reason:
                reason_text = "偵測到關鍵字『賄賂送禮』"
            else:
                reason_text = reason
            rejected.append(RejectedShop(shop_name=name, estimated_score=score, reason=reason_text))
        rejected.sort(key=lambda x: x.estimated_score, reverse=True)
        return top, rejected

    @staticmethod
    def rank(
        shops: list[ShopProfile],
        preference: UserPreference,
        minefield: UserMinefield,
    ) -> tuple[list[RankedShop], list[tuple[str, str]]]:
        # Backward-compatible wrapper.
        ranked, rejected = RankingEngine.generate_top_picks(shops, preference, minefield)
        return ranked, [(r.shop_name, r.reason) for r in rejected]


class ItinerarySynthesizer:
    DAYTIME_WINDOW: tuple[tuple[int, int], tuple[int, int]] = ((7, 0), (23, 0))
    SLOT_TAG_MATRIX: dict[str, dict[str, float]] = {
        "breakfast": {"breakfast": 3.0, "brunch": 2.2, "morning": 2.0, "soy_milk": 1.8, "ramen": 1.2},
        "lunch": {"lunch": 2.5, "main_meal": 1.6, "quick_meal": 1.4, "beef_noodle": 1.4, "ramen": 1.3},
        "tea": {"tea": 2.8, "afternoon_tea": 2.6, "dessert": 2.0, "cafe": 1.8, "refresh": 1.2},
        "dinner": {"dinner": 2.5, "main_meal": 1.7, "course": 1.6, "kaiseki": 1.5, "social": 1.2, "izakaya": 1.2},
        "late_night": {"late_night": 3.0, "night_food": 2.4, "izakaya": 2.8, "snack": 1.4, "ramen": 1.6},
    }
    SLOT_ORDER: dict[str, int] = {
        "breakfast": 0,
        "lunch": 1,
        "tea": 2,
        "dinner": 3,
        "late_night": 4,
    }
    #: Soft DP objective multiplier when node.tags intersects these (×1.3 vs slot).
    SLOT_PREFERRED_TAGS: dict[str, frozenset[str]] = {
        "breakfast": frozenset({"breakfast", "morning", "coffee", "morning_set"}),
        "lunch": frozenset({"lunch", "main_meal", "quick_meal"}),
        "tea": frozenset({"afternoon_tea", "cafe", "dessert", "coffee", "refresh"}),
        "dinner": frozenset({"dinner", "main_meal", "course"}),
        "late_night": frozenset({"late_night", "izakaya", "ramen", "night_food"}),
    }
    #: Soft DP objective multiplier when node.tags hit "wrong slot" cues (×0.5).
    SLOT_NEGATIVE_TAGS: dict[str, frozenset[str]] = {
        "lunch": frozenset({"afternoon_tea", "dessert", "cake", "patisserie"}),
        "dinner": frozenset({"morning", "breakfast", "morning_set"}),
        "breakfast": frozenset({"late_night", "dinner", "izakaya"}),
    }
    #: Hard filter: shop tag-bag (tags ∪ occasion_tags) ∩ excluded ≠ ∅ → ineligible.
    DIETARY_EXCLUDED_TAGS: dict[str, frozenset[str]] = {
        "no_beef": frozenset({"beef", "yakiniku", "bbq_yakiniku", "wagyu", "beef_cutlet", "katsu"}),
        "no_ramen": frozenset({"ramen", "kotteri", "tonkotsu", "豚骨"}),
        "no_pork": frozenset({"pork", "tonkatsu", "tonkotsu"}),
        "vegetarian": frozenset(
            {"beef", "pork", "chicken", "seafood", "beef_cutlet", "katsu", "yakiniku", "bbq_yakiniku", "wagyu"}
        ),
        "vegan": frozenset(
            {
                "beef",
                "pork",
                "chicken",
                "seafood",
                "dairy",
                "beef_cutlet",
                "katsu",
                "yakiniku",
                "bbq_yakiniku",
                "wagyu",
            }
        ),
        # Legacy / intent: fish allowed; land-animal meats excluded
        "pescatarian": frozenset(
            {
                "beef",
                "pork",
                "chicken",
                "meat",
                "tonkatsu",
                "tonkotsu",
                "yakiniku",
                "bbq_yakiniku",
                "wagyu",
                "beef_cutlet",
                "katsu",
            }
        ),
    }

    @staticmethod
    def _canonical_dietary_constraint_key(raw: str | None) -> str | None:
        """Normalize API / profile strings to DIETARY_EXCLUDED_TAGS keys."""
        if not raw:
            return None
        k = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        if k in {"", "none", "omnivore", "unspecified", "regular"}:
            return None
        if k in {"nob_beef", "nobeef"}:
            k = "no_beef"
        if k in {"noramen", "no_ramyen"}:
            k = "no_ramen"
        if k in {"nopork"}:
            k = "no_pork"
        return k if k in ItinerarySynthesizer.DIETARY_EXCLUDED_TAGS else None

    @staticmethod
    def excluded_tags_for_dietary_key(key: str | None) -> frozenset[str]:
        canon = ItinerarySynthesizer._canonical_dietary_constraint_key(key)
        if not canon:
            return frozenset()
        return ItinerarySynthesizer.DIETARY_EXCLUDED_TAGS.get(canon, frozenset())

    @staticmethod
    def union_excluded_tags_from_dietary_keys(keys: Iterable[str | None]) -> frozenset[str]:
        bag: set[str] = set()
        for k in keys:
            if not k:
                continue
            bag |= set(ItinerarySynthesizer.excluded_tags_for_dietary_key(str(k).strip()))
        return frozenset(bag)

    @staticmethod
    def shop_has_excluded_tag(shop: ShopProfile, excluded: frozenset[str]) -> bool:
        if not excluded:
            return False
        tag_bag = {str(t).lower() for t in shop.tags} | {
            str(t).lower() for t in (getattr(shop, "occasion_tags", None) or ())
        }
        return bool(tag_bag & excluded)

    @staticmethod
    def node_tags_intersect_excluded(tags: tuple[str, ...] | Iterable[str], excluded: frozenset[str]) -> bool:
        if not excluded:
            return False
        tag_bag = {str(t).lower() for t in tags}
        return bool(tag_bag & excluded)

    @staticmethod
    def _slot_to_time_bucket(slot: str) -> str:
        if slot == "lunch":
            return "lunch"
        if slot in {"dinner", "late_night"}:
            return "dinner"
        return "offpeak"

    @staticmethod
    def _slot_semantic_match_score(slot: str, ranked_shop: RankedShop) -> int:
        expected = ItinerarySynthesizer.SLOT_TAG_MATRIX.get(slot, {})
        if not expected:
            return 0
        occasion_tags = {t.lower() for t in ranked_shop.shop.occasion_tags}
        generic_tags = {t.lower() for t in ranked_shop.shop.tags}
        score = 0.0
        for tag, w in expected.items():
            if tag in occasion_tags:
                score += w * 2.0
            elif tag in generic_tags:
                score += w
        return int(round(score * 10.0))

    @staticmethod
    def _slot_semantic_bonus_multiplier(slot: str, ranked_shop: RankedShop) -> float:
        # Keep semantic matching as a ranking preference, not a hard feasibility gate.
        semantic_score = ItinerarySynthesizer._slot_semantic_match_score(slot, ranked_shop)
        if semantic_score <= 0:
            return 1.0
        # Cap bonus so semantic preference doesn't overwhelm hard operational constraints.
        return min(1.35, 1.0 + (semantic_score / 1000.0))

    @staticmethod
    def _slot_must_have_priority_tags(slot: str) -> set[str]:
        """
        Dynamic must-have priority by slot.
        Tags with large weights are treated as high-priority anchors.
        """
        matrix = ItinerarySynthesizer.SLOT_TAG_MATRIX.get(slot, {})
        return {tag for tag, weight in matrix.items() if weight >= 2.6}

    @staticmethod
    def _normalize_slot_sequence(meal_slots: list[str] | None) -> list[str]:
        if not meal_slots:
            return []
        valid = [s for s in meal_slots if s in ItinerarySynthesizer.SLOT_ORDER]
        return sorted(valid, key=lambda s: ItinerarySynthesizer.SLOT_ORDER[s])

    @staticmethod
    def _normalize_slot_required_tags(
        slot_required_tags: dict[str, set[str]] | None,
    ) -> dict[str, set[str]]:
        if not slot_required_tags:
            return {}
        return {
            str(k).lower(): {str(x).lower() for x in v}
            for k, v in slot_required_tags.items()
        }

    @staticmethod
    def _shop_matches_slot_required_tags(
        shop: ShopProfile,
        slot_name: str | None,
        slot_required_tags: dict[str, set[str]],
    ) -> bool:
        """Each slot maps to an OR-group; shop tags ∪ occasion_tags must intersect."""
        if not slot_name or not slot_required_tags:
            return True
        need = slot_required_tags.get(str(slot_name).lower())
        if not need:
            return True
        bag = {str(t).lower() for t in shop.tags} | {str(t).lower() for t in shop.occasion_tags}
        return bool(bag & need)

    @staticmethod
    def _calculate_cooldown(
        prev_shop: ShopProfile | None,
        *,
        mode: OptimizationMode = OptimizationMode.BALANCED,
        requested_meal_count: int | None = None,
        appetite_light_mode: bool = False,
    ) -> int:
        """Dynamic digestion gap after the previous meal (minutes)."""
        from feasibility_utils import calculate_cooldown as _fc_cd
        return _fc_cd(prev_shop, mode=mode, requested_meal_count=requested_meal_count, appetite_light_mode=appetite_light_mode)

    @staticmethod
    def _shop_last_call_at(base: datetime, shop: ShopProfile) -> datetime:
        hh, mm = [int(x) for x in shop.close_time.split(":", 1)]
        close_at = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
        # Last-call is strictly anchored to the planning day; never auto-roll to next day.
        return close_at - timedelta(minutes=max(0, int(shop.last_call_offset)))

    @staticmethod
    def _early_bird_priority(base: datetime, shop: ShopProfile) -> int:
        """
        Early-bird protection:
        - shop whose last-call is before 12:00 should be prioritized
        - only if still physically feasible at current planning moment
        """
        last_call_at = ItinerarySynthesizer._shop_last_call_at(base, shop)
        is_early_bird = (last_call_at.hour, last_call_at.minute) < (12, 0)
        if not is_early_bird:
            return 0
        return 1 if base < last_call_at else 0

    @staticmethod
    def _shop_open_at(base: datetime, shop: ShopProfile) -> datetime:
        from feasibility_utils import shop_open_at as _fe_oa
        return _fe_oa(base, shop)

    @staticmethod
    def _shop_open_close_window(base: datetime, shop: ShopProfile) -> tuple[datetime, datetime]:
        from feasibility_utils import shop_open_close_window as _fe_ocw
        return _fe_ocw(base, shop)

    @staticmethod
    def _scarcity_bonus(base: datetime, shop: ShopProfile) -> int:
        """
        Scarcity bonus for fragile operating windows.
        Higher score when:
        - operating window is short
        - current scheduling time is inside that window
        """
        open_at, close_at = ItinerarySynthesizer._shop_open_close_window(base, shop)
        if not (open_at <= base < close_at):
            return 0
        window_minutes = max(1, int((close_at - open_at).total_seconds() // 60))
        if window_minutes <= 240:  # <= 4 hours
            return 3
        if window_minutes <= 360:  # <= 6 hours
            return 2
        if window_minutes <= 480:  # <= 8 hours
            return 1
        return 0

    @staticmethod
    def _align_to_daytime(current: datetime) -> tuple[datetime, bool]:
        start_hm, end_hm = ItinerarySynthesizer.DAYTIME_WINDOW
        day_start = current.replace(hour=start_hm[0], minute=start_hm[1], second=0, microsecond=0)
        day_end = current.replace(hour=end_hm[0], minute=end_hm[1], second=0, microsecond=0)
        if current < day_start:
            return day_start, True
        if current <= day_end:
            return current, False
        next_day_start = (current + timedelta(days=1)).replace(
            hour=start_hm[0], minute=start_hm[1], second=0, microsecond=0
        )
        return next_day_start, True

    @staticmethod
    def find_optimal_path(
        graph: SpatioTemporalGraph,
        required_length: int,
        must_have_tags: set[str] | None = None,
        banned_node_ids: set[str] | None = None,
        solver_audit_log: list[str] | None = None,
        meal_slots: list[str] | None = None,
        excluded_shop_tags: frozenset[str] | None = None,
        allow_global_repeat: bool = False,
    ) -> list[GraphNode]:
        """
        DAG longest-path with DP under fixed path length and tag-coverage constraints.
        Each shop_name appears at most once per path when allow_global_repeat=False (default).
        With allow_global_repeat=True, the same shop can appear at non‑adjacent positions,
        but **immediately consecutive identical shop_name is always forbidden**.
        Optional meal_slots activates soft slot–tag affinity (×1.3 when node.tags hits
        SLOT_PREFERRED_TAGS for that slot).
        """
        if solver_audit_log is not None:
            solver_audit_log.append(
                _dj(
                    "dp_start",
                    required_length=required_length,
                    nodes=len(graph.nodes),
                    edges=len(graph.edges),
                )
            )
        if required_length <= 0 or not graph.nodes:
            if solver_audit_log is not None:
                solver_audit_log.append(
                    _dj("dp_early_exit", reason="invalid_required_length_or_empty_graph")
                )
            return []
        must_have_tags = {t.lower() for t in (must_have_tags or set())}
        required_tags = sorted(must_have_tags)
        tag_idx = {t: i for i, t in enumerate(required_tags)}
        full_mask = (1 << len(required_tags)) - 1

        slot_sequence_norm = ItinerarySynthesizer._normalize_slot_sequence(meal_slots) if meal_slots else []

        banned_node_ids = banned_node_ids or set()
        usable_nodes = [n for n in graph.nodes if n.node_id not in banned_node_ids]
        _excluded = excluded_shop_tags or frozenset()
        if _excluded:
            before_dn = len(usable_nodes)
            usable_nodes = [
                n for n in usable_nodes if not ItinerarySynthesizer.node_tags_intersect_excluded(n.tags, _excluded)
            ]
            if solver_audit_log is not None and before_dn != len(usable_nodes):
                solver_audit_log.append(
                    _dj(
                        "dp_excluded_tag_node_filter",
                        excluded_count=len(_excluded),
                        dropped=before_dn - len(usable_nodes),
                        retained=len(usable_nodes),
                    )
                )
        if not usable_nodes:
            if solver_audit_log is not None:
                solver_audit_log.append(
                    _dj("dp_early_exit", reason="all_nodes_filtered_by_banned_node_ids")
                )
            return []
        node_by_id = {n.node_id: n for n in usable_nodes}
        indeg: dict[str, int] = {nid: 0 for nid in node_by_id}
        out_edges: dict[str, list[GraphEdge]] = {nid: [] for nid in node_by_id}
        for e in graph.edges:
            if (
                e.from_node_id not in node_by_id
                or e.to_node_id not in node_by_id
                or e.from_node_id in banned_node_ids
                or e.to_node_id in banned_node_ids
            ):
                continue
            indeg[e.to_node_id] += 1
            out_edges[e.from_node_id].append(e)

        # Kahn topological sort
        queue = [nid for nid, d in indeg.items() if d == 0]
        topo: list[str] = []
        while queue:
            queue.sort(key=lambda nid: (node_by_id[nid].slot_index, node_by_id[nid].start_time))
            cur = queue.pop(0)
            topo.append(cur)
            for e in out_edges[cur]:
                indeg[e.to_node_id] -= 1
                if indeg[e.to_node_id] == 0:
                    queue.append(e.to_node_id)
        if len(topo) < len(node_by_id):
            # Fallback stable order if graph isn't fully sortable (shouldn't happen in DAG).
            topo = sorted(node_by_id.keys(), key=lambda nid: (node_by_id[nid].slot_index, node_by_id[nid].start_time))

        def tag_mask(node: GraphNode) -> int:
            m = 0
            node_tags_lp = {t.lower() for t in node.tags}
            for t, idx in tag_idx.items():
                if t in node_tags_lp:
                    m |= 1 << idx
            return m

        def _preferred_tag_bonus(node: GraphNode) -> float:
            if not slot_sequence_norm or node.slot_index < 0 or node.slot_index >= len(slot_sequence_norm):
                return 1.0
            slot_nm = str(slot_sequence_norm[node.slot_index]).lower()
            preferred = ItinerarySynthesizer.SLOT_PREFERRED_TAGS.get(slot_nm)
            if not preferred:
                return 1.0
            node_tags_lp = {t.lower() for t in node.tags}
            return 1.3 if node_tags_lp & preferred else 1.0

        def _negative_tag_penalty(node: GraphNode) -> float:
            if not slot_sequence_norm or node.slot_index < 0 or node.slot_index >= len(slot_sequence_norm):
                return 1.0
            slot_nm = str(slot_sequence_norm[node.slot_index]).lower()
            avoid = ItinerarySynthesizer.SLOT_NEGATIVE_TAGS.get(slot_nm)
            if not avoid:
                return 1.0
            node_tags_lp = {t.lower() for t in node.tags}
            return 0.5 if node_tags_lp & avoid else 1.0

        def node_objective_score(node: GraphNode) -> float:
            ideal = max(1, int(node.ideal_duration_minutes))
            actual = max(1, int(node.actual_duration_minutes))
            fidelity = max(0.0, min(1.0, float(actual) / float(ideal)))
            return (
                float(node.final_score)
                * fidelity
                * _preferred_tag_bonus(node)
                * _negative_tag_penalty(node)
            )

        # (node_id, length, mask, seen_shop_names, previous_shop_name) -> score
        KeyT = tuple[str, int, int, frozenset[str], str | None]
        best: dict[KeyT, float] = {}
        prev: dict[KeyT, KeyT | None] = {}

        for nid in topo:
            node = node_by_id[nid]
            m = tag_mask(node)
            seen0 = frozenset({node.shop_name})
            key: KeyT = (nid, 1, m, seen0, node.shop_name)
            best[key] = node_objective_score(node)
            prev[key] = None

        for nid in topo:
            outgoing = out_edges.get(nid, [])
            # enumerate existing states ending at nid
            cur_states = [(k, v) for k, v in best.items() if k[0] == nid]
            if not cur_states:
                continue
            for (cur_key, cur_score) in cur_states:
                _, cur_len, cur_mask, cur_seen, cur_prev_shop = cur_key
                if cur_len >= required_length:
                    continue
                for e in outgoing:
                    to_node = node_by_id[e.to_node_id]
                    # hard rule: never allow immediately consecutive same shop
                    if to_node.shop_name == cur_prev_shop:
                        continue
                    if not allow_global_repeat and to_node.shop_name in cur_seen:
                        continue
                    next_mask = cur_mask | tag_mask(to_node)
                    next_seen = frozenset(cur_seen | {to_node.shop_name}) if not allow_global_repeat else cur_seen
                    nxt: KeyT = (e.to_node_id, cur_len + 1, next_mask, next_seen, to_node.shop_name)
                    cand = cur_score + node_objective_score(to_node)
                    if cand > best.get(nxt, float("-inf")):
                        best[nxt] = cand
                        prev[nxt] = cur_key

        # Prefer exact required length; otherwise degrade gracefully to the
        # longest feasible length instead of returning empty path.
        max_len = max((k[1] for k in best.keys()), default=0)
        target_len = required_length if any(k[1] == required_length for k in best.keys()) else max_len
        if target_len <= 0:
            if solver_audit_log is not None:
                solver_audit_log.append(_dj("dp_early_exit", reason="no_feasible_terminal_state"))
            return []
        terminal_keys = [k for k in best.keys() if k[1] == target_len]
        if required_tags:
            covered = [k for k in terminal_keys if k[2] == full_mask]
            if covered:
                terminal_keys = covered
            elif target_len == required_length:
                # If exact length cannot satisfy full tag coverage, degrade to the
                # longest feasible length that satisfies coverage.
                feasible_lens = sorted({k[1] for k in best.keys()}, reverse=True)
                for ln in feasible_lens:
                    covered_ln = [k for k in best.keys() if k[1] == ln and k[2] == full_mask]
                    if covered_ln:
                        terminal_keys = covered_ln
                        target_len = ln
                        break
        if target_len < required_length and solver_audit_log is not None and terminal_keys:
            best_terminal = max(terminal_keys, key=lambda k: best[k])
            terminal_id = best_terminal[0]
            terminal = node_by_id[terminal_id]
            out_degree = len(out_edges.get(terminal_id, []))
            solver_audit_log.append(
                _dj(
                    "dp_downgrade",
                    required=required_length,
                    reachable=target_len,
                    terminal_shop=terminal.shop_name,
                    terminal_slot=terminal.slot_index,
                )
            )
            if out_degree == 0:
                solver_audit_log.append(
                    _dj(
                        "dp_extension_blocked",
                        variant="terminal_no_outgoing_compatible_transitions",
                    )
                )
            else:
                solver_audit_log.append(
                    _dj(
                        "dp_extension_blocked",
                        variant="outgoing_exist_but_cannot_extend_under_constraints",
                    )
                )
        if solver_audit_log is not None:
            solver_audit_log.append(
                _dj(
                    "dp_terminal",
                    target_len=target_len,
                    terminal_candidates=len(terminal_keys),
                )
            )
        end_key = max(terminal_keys, key=lambda k: best[k])

        # reconstruct
        path_keys: list[KeyT] = []
        cur_k: KeyT | None = end_key
        while cur_k is not None:
            path_keys.append(cur_k)
            cur_k = prev.get(cur_k)
        path_keys.reverse()
        resolved_path = [node_by_id[k[0]] for k in path_keys]
        if solver_audit_log is not None:
            solver_audit_log.append(
                _dj(
                    "dp_path_selected",
                    path=[
                        {"shop": n.shop_name, "slot": n.slot_index} for n in resolved_path
                    ],
                )
            )
        return resolved_path

    @staticmethod
    def find_k_optimal_paths(
        graph: SpatioTemporalGraph,
        required_length: int,
        must_have_tags: set[str] | None = None,
        k: int = 3,
        meal_slots: list[str] | None = None,
        excluded_shop_tags: frozenset[str] | None = None,
    ) -> list[list[GraphNode]]:
        """
        Lightweight K-best paths: iteratively ban one chosen node from previous path
        and rerun optimal solver. This approximates K-shortest alternatives for fallback.
        """
        out: list[list[GraphNode]] = []
        banned: set[str] = set()
        for _ in range(max(1, k)):
            path = ItinerarySynthesizer.find_optimal_path(
                graph=graph,
                required_length=required_length,
                must_have_tags=must_have_tags,
                banned_node_ids=banned,
                meal_slots=meal_slots,
                excluded_shop_tags=excluded_shop_tags,
            )
            if not path:
                break
            out.append(path)
            # ban the last node to force alternative continuation on next iteration.
            banned.add(path[-1].node_id)
        return out

    @staticmethod
    def synthesize(
        ranked: list[RankedShop],
        traffic: TrafficAdapter,
        preference: UserPreference,
        start_time: datetime,
        sns_adapter: SnsAdapter | None = None,
        isolation_threshold: float = 0.7,
        mode: OptimizationMode = OptimizationMode.BALANCED,
        meal_slots: list[str] | None = None,
        global_end_time: datetime | None = None,
        requested_meal_count: int | None = None,
        explicit_required_tags: set[str] | None = None,
        slot_required_tags: dict[str, set[str]] | None = None,
        appetite_light_mode: bool = False,
        respect_slot_order: bool = False,
        excluded_shop_tags: frozenset[str] | None = None,
    ) -> SynthesisResult:
        nodes: list[ScheduleNode] = []
        backups: list[str] = []
        warnings: list[str] = []
        solver_audit_log: list[str] = []
        graph_debug_traces: list[str] = []
        rollback_triggered = False
        previous_flavor_intensity = 0.0
        fatigue_detected = False
        current = start_time
        last_meal_end: datetime | None = None
        prev_meal_shop: ShopProfile | None = None
        prev_shop_name = "Kyoto Station"
        used: set[str] = set()
        insider_by_shop: dict[str, bool] = {
            r.shop.name: bool(getattr(r, "insider_pick", False)) for r in ranked
        }
        normalized_slots = ItinerarySynthesizer._normalize_slot_sequence(meal_slots)
        global_req = {t.lower() for t in (explicit_required_tags or set())}
        slot_req_norm = ItinerarySynthesizer._normalize_slot_required_tags(slot_required_tags)
        desired_len = max(1, requested_meal_count or len(normalized_slots or ranked[:3]))
        preferred_shop_by_slot: dict[int, str] = {}
        graph = GraphBuilder.build_graph(
            ranked=ranked,
            traffic=traffic,
            start_time=start_time,
            meal_slots=normalized_slots if normalized_slots else None,
            mode=mode,
            requested_meal_count=requested_meal_count,
            slot_required_tags=slot_req_norm if slot_req_norm else None,
            excluded_shop_tags=excluded_shop_tags,
        )
        graph_debug_traces.extend(graph.debug_traces)
        if not graph.debug_traces:
            graph_debug_traces.append(
                _dj(
                    "graph_no_rejected_edges",
                    nodes=len(graph.nodes),
                    edges=len(graph.edges),
                )
            )
        optimal_nodes = ItinerarySynthesizer.find_optimal_path(
            graph=graph,
            required_length=min(desired_len, max(1, len(normalized_slots or ranked[:3]))),
            must_have_tags=global_req,
            solver_audit_log=solver_audit_log,
            meal_slots=normalized_slots if normalized_slots else None,
            excluded_shop_tags=excluded_shop_tags,
        )
        if optimal_nodes and len(optimal_nodes) < desired_len:
            warnings.append(
                f"無法滿足 {desired_len} 餐需求，已根據物理約束提供最佳 {len(optimal_nodes)} 餐方案"
            )
        for gn in optimal_nodes:
            preferred_shop_by_slot[gn.slot_index] = gn.shop_name
        if preferred_shop_by_slot:
            warnings.append(f"GRAPH_OPTIMAL_PATH slots={preferred_shop_by_slot}")
        if meal_slots and normalized_slots != meal_slots:
            warnings.append(f"MEAL_ORDER_NORMALIZED from={meal_slots} to={normalized_slots}")
        slot_requests: list[str | None] = normalized_slots if normalized_slots else [None for _ in ranked[:3]]

        for i, slot_name in enumerate(slot_requests):
            if respect_slot_order:
                if i >= len(ranked):
                    warnings.append(f"{slot_name or 'meal'} 無可用店家：DP 綁定清單長度不足")
                    break
                slot_rs = ranked[i]
                _excl = excluded_shop_tags or frozenset()
                if _excl and ItinerarySynthesizer.shop_has_excluded_tag(slot_rs.shop, _excl):
                    warnings.append(
                        f"EXCLUDED_TAG_SKIP {slot_rs.shop.name}: matches excluded_shop_tags"
                    )
                    break
                if slot_rs.shop.name in used:
                    warnings.append(f"DP_SLOT_ORDER_CONFLICT {slot_rs.shop.name} at slot_index={i}")
                    break
                candidates = [slot_rs]
            else:
                candidates = [r for r in ranked if r.shop.name not in used]
                if not candidates:
                    break
                if slot_name is not None:
                    slot_anchor_tags = ItinerarySynthesizer._slot_must_have_priority_tags(slot_name)
                    preferred_shop = preferred_shop_by_slot.get(i)
                    candidates.sort(
                        key=lambda r: (
                            1 if preferred_shop and r.shop.name == preferred_shop else 0,
                            ItinerarySynthesizer._scarcity_bonus(current, r.shop),
                            1 if any(t in {str(x).lower() for x in r.shop.tags} for t in slot_anchor_tags) else 0,
                            ItinerarySynthesizer._early_bird_priority(current, r.shop),
                            r.final_score * ItinerarySynthesizer._slot_semantic_bonus_multiplier(slot_name, r),
                            ItinerarySynthesizer._slot_semantic_match_score(slot_name, r),
                        ),
                        reverse=True,
                    )
                else:
                    candidates.sort(
                        key=lambda r: (
                            ItinerarySynthesizer._scarcity_bonus(current, r.shop),
                            ItinerarySynthesizer._early_bird_priority(current, r.shop),
                            r.final_score,
                        ),
                        reverse=True,
                    )
            if not candidates:
                break
            # Breakfast visibility: full-candidate scan only (strict DP order uses one shop per slot).
            if slot_name == "breakfast" and not respect_slot_order:
                for _rs in candidates:
                    _s = _rs.shop
                    _generic = {str(t).lower() for t in _s.tags}
                    if global_req and not any(t in _generic for t in global_req):
                        continue
                    if not ItinerarySynthesizer._shop_matches_slot_required_tags(_s, slot_name, slot_req_norm):
                        continue
                    _occ = {t.lower() for t in _s.occasion_tags}
                    _has_bf = ("breakfast" in _occ) or ("breakfast" in _generic)
                    _ramen_bypass = bool(
                        global_req and "ramen" in global_req and "ramen" in _generic
                    )
                    if not _has_bf and not _ramen_bypass:
                        warnings.append(f"BREAKFAST_TAG_REQUIRED_SKIP {_s.name}")
            scheduled = False
            for ranked_shop in candidates:
                s = ranked_shop.shop
                _ex_loop = excluded_shop_tags or frozenset()
                if _ex_loop and ItinerarySynthesizer.shop_has_excluded_tag(s, _ex_loop):
                    continue
                candidate_current = current
                candidate_last_meal_end = last_meal_end
                alpha = float(getattr(s, "_isolation_factor", 0.0))
                shop_tag_set = {str(t).lower() for t in s.tags}
                if global_req and not any(t in shop_tag_set for t in global_req):
                    warnings.append(
                        f"HARD_FILTER_TAG_SKIP {s.name}: required={sorted(global_req)} shop_tags={sorted(shop_tag_set)}"
                    )
                    continue
                if not ItinerarySynthesizer._shop_matches_slot_required_tags(s, slot_name, slot_req_norm):
                    need = sorted(slot_req_norm.get(str(slot_name).lower(), set()))
                    occ = sorted({str(t).lower() for t in s.occasion_tags})
                    warnings.append(
                        f"TAG_MISMATCH slot={slot_name or 'n/a'} shop={s.name} "
                        f"need_any_of={need} have_tags={sorted(shop_tag_set)} have_occasion={occ}"
                    )
                    continue
                if slot_name == "breakfast":
                    occasion_tags = {t.lower() for t in s.occasion_tags}
                    generic_tags = {t.lower() for t in s.tags}
                    has_breakfast_tag = ("breakfast" in occasion_tags) or ("breakfast" in generic_tags)
                    has_ramen_intent_match = bool(
                        global_req and "ramen" in global_req and "ramen" in shop_tag_set
                    )
                    if not has_breakfast_tag and not has_ramen_intent_match:
                        continue
                    if not has_breakfast_tag and has_ramen_intent_match:
                        warnings.append(
                            f"BREAKFAST_TAG_BYPASS_EXPLICIT_INTENT {s.name}: allowed_by=ramen_intent"
                        )
                cooldown_end: datetime | None = None
                cooldown_m = 0
                if candidate_last_meal_end is not None:
                    cooldown_m = ItinerarySynthesizer._calculate_cooldown(
                        prev_meal_shop,
                        mode=mode,
                        requested_meal_count=requested_meal_count,
                        appetite_light_mode=appetite_light_mode,
                    )
                    cooldown_end = candidate_last_meal_end + timedelta(minutes=cooldown_m)
                if slot_name not in {None, "late_night"}:
                    aligned_daytime, moved = ItinerarySynthesizer._align_to_daytime(candidate_current)
                    if moved:
                        candidate_current = aligned_daytime
                        warnings.append(
                            f"{slot_name} 已套用日間可行時間對齊：{candidate_current.strftime('%Y-%m-%d %H:%M')}"
                        )
                # Meal slot names affect catalog affinity only; no fixed clock windows per slot.
                from_loc = prev_shop_name if i > 0 else "Kyoto Station"
                to_loc = s.name
                traffic_status = traffic.get_route_status(from_loc, to_loc)
                travel_m = 18 + (8 * i)
                slot_bucket = ItinerarySynthesizer._slot_to_time_bucket(slot_name or "lunch")
                wait_m = predict_wait_time(s, start_time.strftime("%a"), slot_bucket, visit_time=candidate_current)
                buffer_m = traffic_status.transport_buffer_minutes

                if mode == OptimizationMode.TASTE_MAX:
                    travel_m = 0
                    buffer_m = 0
                    from_loc = "ANYWHERE"

                if alpha >= isolation_threshold:
                    if preference.prefers_driving:
                        driving_status = traffic.get_driving_status(from_loc, to_loc)
                        buffer_m = driving_status.traffic_jitter_minutes
                    else:
                        buffer_m += 30
                    if sns_adapter is not None:
                        signal = sns_adapter.check_store_status(s.sns_handle).lower()
                        risk_keywords = ("火山", "臨休", "休業", "完売", "sold out")
                        if any(k in signal for k in risk_keywords):
                            warnings.append(f"OPERATING_BOUNDARY_SKIP {s.name}: live-risk={signal[:30]}")
                            continue

                t_start = candidate_current
                t_end = candidate_current + timedelta(minutes=travel_m + buffer_m)
                arrival_time = t_end
                arrival_last_call_at = ItinerarySynthesizer._shop_last_call_at(arrival_time, s)
                if arrival_time >= arrival_last_call_at:
                    warnings.append(
                        f"CONSTRAINT_LAST_CALL_EXCEEDED {s.name}: arrival={arrival_time.strftime('%H:%M')} >= last_call={arrival_last_call_at.strftime('%H:%M')}"
                    )
                    continue
                raw_seat_time = arrival_time + timedelta(minutes=wait_m)
                if cooldown_end is not None and candidate_last_meal_end is not None:
                    eat_start = max(raw_seat_time, cooldown_end)
                    # Digestion window vs. outbound transit + queue (overlap reduces idle waiting).
                    overlap_start = max(candidate_last_meal_end, candidate_current)
                    overlap_end = min(cooldown_end, raw_seat_time)
                    overlap_minutes = max(0, int((overlap_end - overlap_start).total_seconds() // 60))
                    if eat_start > raw_seat_time:
                        residual = int((eat_start - raw_seat_time).total_seconds() // 60)
                        warnings.append(
                            f"消化冷卻與交通併行：抵達與排隊後仍需等候 {residual} 分鐘（上一餐 flavor={prev_meal_shop.flavor_category.value if prev_meal_shop else 'n/a'}，冷卻總長 {cooldown_m} 分鐘）"
                        )
                    elif overlap_minutes > 0:
                        warnings.append(
                            f"交通與消化時間重疊約 {overlap_minutes} 分鐘（冷卻需求 {cooldown_m} 分鐘已由路程／等待路程期間抵銷）"
                        )
                else:
                    eat_start = raw_seat_time
                if slot_name not in {None, "late_night"}:
                    eat_start, moved = ItinerarySynthesizer._align_to_daytime(eat_start)
                    if moved:
                        warnings.append(f"{slot_name} 用餐起點已對齊日間可行時間：{eat_start.strftime('%Y-%m-%d %H:%M')}")
                open_at_shop = ItinerarySynthesizer._shop_open_at(eat_start, s)
                if eat_start < open_at_shop:
                    warnings.append(
                        f"WAIT_UNTIL_OPEN {s.name}: deferred meal start to {open_at_shop.strftime('%H:%M')}"
                    )
                    eat_start = open_at_shop
                selected_eat_minutes = int(s.avg_eat_minutes)
                compression_note = ""
                eat_end = eat_start + timedelta(minutes=selected_eat_minutes)
                if global_end_time is not None and eat_end > global_end_time:
                    min_eat = max(1, int(s.min_eat_minutes or s.avg_eat_minutes))
                    if min_eat < selected_eat_minutes:
                        compressed_end = eat_start + timedelta(minutes=min_eat)
                        if compressed_end <= global_end_time:
                            selected_eat_minutes = min_eat
                            eat_end = compressed_end
                            compression_note = (
                                f"【行程壓縮】為了趕上下一站，建議縮短用餐時間至 {selected_eat_minutes} 分鐘。"
                            )
                        else:
                            warnings.append(
                                f"WINDOW_EXCEEDED {s.name}: eat_end={eat_end.strftime('%H:%M')} > window_end={global_end_time.strftime('%H:%M')}"
                            )
                            return SynthesisResult(
                                nodes=nodes,
                                backup_nodes=backups,
                                warnings=warnings,
                                solver_audit_log=solver_audit_log,
                                graph_debug_traces=graph_debug_traces,
                                rollback_triggered=rollback_triggered,
                            )
                    else:
                        warnings.append(
                            f"WINDOW_EXCEEDED {s.name}: eat_end={eat_end.strftime('%H:%M')} > window_end={global_end_time.strftime('%H:%M')}"
                        )
                        return SynthesisResult(
                            nodes=nodes,
                            backup_nodes=backups,
                            warnings=warnings,
                            solver_audit_log=solver_audit_log,
                            graph_debug_traces=graph_debug_traces,
                            rollback_triggered=rollback_triggered,
                        )

                last_call_at = ItinerarySynthesizer._shop_last_call_at(eat_start, s)
                if eat_start >= last_call_at:
                    warnings.append(
                        f"CONSTRAINT_LAST_CALL_EXCEEDED {s.name}: eat_start={eat_start.strftime('%H:%M')} >= last_call={last_call_at.strftime('%H:%M')}"
                    )
                    continue
                if eat_end > last_call_at:
                    min_eat = max(1, int(s.min_eat_minutes or s.avg_eat_minutes))
                    if min_eat < selected_eat_minutes:
                        compressed_end = eat_start + timedelta(minutes=min_eat)
                        if compressed_end <= last_call_at:
                            selected_eat_minutes = min_eat
                            eat_end = compressed_end
                            compression_note = (
                                f"【行程壓縮】為了趕上下一站，建議縮短用餐時間至 {selected_eat_minutes} 分鐘。"
                            )
                        else:
                            warnings.append(
                                f"OPERATING_BOUNDARY_SKIP {s.name}: eat_end={eat_end.strftime('%H:%M')} > last_call={last_call_at.strftime('%H:%M')}"
                            )
                            continue
                    else:
                        warnings.append(
                            f"OPERATING_BOUNDARY_SKIP {s.name}: eat_end={eat_end.strftime('%H:%M')} > last_call={last_call_at.strftime('%H:%M')}"
                        )
                        continue
                if "TIME_UNKNOWN" in s.tags:
                    warnings.append(
                        f"YELLOW_WARNING TIME_UNKNOWN {s.name}: source missing open_time, schedule confidence reduced"
                    )

                # Commit scheduled candidate.
                used.add(s.name)
                nodes.append(
                    ScheduleNode(
                        title=f"Transit to {s.name}",
                        start_at=t_start,
                        end_at=t_end,
                        note=f"traffic={traffic_status.semantic_status}, buffer={buffer_m}m",
                    )
                )

                flavor_sum = previous_flavor_intensity + max(0.0, min(1.0, s.flavor_intensity))
                if flavor_sum > 1.6:
                    fatigue_detected = True
                    warnings.append(
                        f"Palate Fatigue Warning: consecutive flavor intensity {previous_flavor_intensity:.2f}+{s.flavor_intensity:.2f}={flavor_sum:.2f} > 1.6"
                    )

                if mode != OptimizationMode.TASTE_MAX and wait_m > preference.max_wait_minutes:
                    backup_candidates = [x.shop.name for x in ranked[:2] if x.shop.name != s.name]
                    if backup_candidates:
                        backups.append(backup_candidates[0])
                        nodes.append(
                            ScheduleNode(
                                title=f"BackupNode -> {backup_candidates[0]}",
                                start_at=t_end,
                                end_at=t_end + timedelta(minutes=5),
                                note=f"wait={wait_m}m > threshold={preference.max_wait_minutes}m",
                            )
                        )
                        current = t_end + timedelta(minutes=5)
                        prev_shop_name = s.name
                        scheduled = True
                        break

                meal_title = f"{slot_name or 'meal'} · {s.name}"
                if insider_by_shop.get(s.name):
                    meal_title = f"{slot_name or 'meal'} · 【老饕私藏】· {s.name}"
                nodes.append(
                    ScheduleNode(
                        title=meal_title,
                        start_at=eat_start,
                        end_at=eat_end,
                        note=(
                            f"wait={wait_m}m, eat={selected_eat_minutes}m, last_call={last_call_at.strftime('%H:%M')}"
                            + (f" {compression_note}" if compression_note else "")
                        ),
                    )
                )
                current = eat_end + timedelta(minutes=15)
                last_meal_end = eat_end
                prev_meal_shop = s
                prev_shop_name = s.name
                previous_flavor_intensity = max(0.0, min(1.0, s.flavor_intensity))
                scheduled = True
                break

            if not scheduled:
                warnings.append(f"{slot_name or 'meal'} 無可用店家：所有候選皆觸發營業硬邊界或風險限制")
        return SynthesisResult(
            nodes=nodes,
            backup_nodes=backups,
            warnings=warnings,
            solver_audit_log=solver_audit_log,
            graph_debug_traces=graph_debug_traces,
            rollback_triggered=rollback_triggered,
        )


class GraphBuilder:
    """
    Transform ranked shops into a spatio-temporal compatibility DAG.
    Node = (shop_name, slot_index), edge exists when A can physically reach B.
    Slot indices encode meal-order / affinity layers only, not fixed clock windows per slot label.
    """


    @staticmethod
    def _slot_anchor_times(start_time: datetime, meal_slots: list[str] | None, node_count: int) -> list[datetime]:
        normalized = ItinerarySynthesizer._normalize_slot_sequence(meal_slots)
        if normalized:
            anchors: list[datetime] = []
            occurrence_count: dict[str, int] = {}
            for slot_name in normalized:
                hh, mm = SLOT_CLOCK_HOUR_MINUTE.get(slot_name, (start_time.hour, start_time.minute))
                occurrence = occurrence_count.get(slot_name, 0)
                occurrence_count[slot_name] = occurrence + 1
                anchor = start_time.replace(hour=hh, minute=mm, second=0, microsecond=0)
                anchor = anchor + timedelta(minutes=70 * occurrence)
                if anchor < start_time:
                    anchor = anchor + timedelta(days=1)
                anchors.append(anchor)
            return anchors
        return [start_time + timedelta(minutes=90 * i) for i in range(max(1, node_count))]

    @staticmethod
    def build_graph(
        ranked: list[RankedShop],
        traffic: TrafficAdapter,
        start_time: datetime,
        *,
        meal_slots: list[str] | None = None,
        mode: OptimizationMode = OptimizationMode.BALANCED,
        requested_meal_count: int | None = None,
        slot_required_tags: dict[str, set[str]] | None = None,
        excluded_shop_tags: frozenset[str] | None = None,
    ) -> SpatioTemporalGraph:
        if not ranked:
            return SpatioTemporalGraph(nodes=[], edges=[], debug_traces=[])

        normalized_meal_slots = ItinerarySynthesizer._normalize_slot_sequence(meal_slots) if meal_slots else []
        slot_req_norm = ItinerarySynthesizer._normalize_slot_required_tags(slot_required_tags)
        anchors = GraphBuilder._slot_anchor_times(start_time, meal_slots, len(meal_slots or ranked[:3]))
        nodes: list[GraphNode] = []
        by_slot: dict[int, list[GraphNode]] = {}
        slot_count = len(anchors)

        for slot_index in range(slot_count):
            slot_name = (
                normalized_meal_slots[slot_index]
                if slot_index < len(normalized_meal_slots)
                else None
            )
            base_anchor = anchors[slot_index]
            for r in ranked:
                s = r.shop
                if not ItinerarySynthesizer._shop_matches_slot_required_tags(s, slot_name, slot_req_norm):
                    continue
                _gx = excluded_shop_tags or frozenset()
                if _gx and ItinerarySynthesizer.shop_has_excluded_tag(s, _gx):
                    continue
                candidate_start = base_anchor
                open_at = ItinerarySynthesizer._shop_open_at(candidate_start, s)
                if candidate_start < open_at:
                    candidate_start = open_at
                # Layer-1 physical interval: base queue + pure eating duration.
                ideal_duration = int(s.base_wait_minutes) + int(s.avg_eat_minutes)
                actual_duration = int(s.base_wait_minutes) + int(s.min_eat_minutes or s.avg_eat_minutes)
                end_time = candidate_start + timedelta(minutes=ideal_duration)
                last_call = ItinerarySynthesizer._shop_last_call_at(candidate_start, s)
                if end_time > last_call:
                    continue
                node_id = f"{s.name}__{slot_index}"
                node = GraphNode(
                    node_id=node_id,
                    shop_name=s.name,
                    slot_index=slot_index,
                    start_time=candidate_start,
                    end_time=end_time,
                    final_score=r.final_score,
                    tags=tuple(str(t).lower() for t in s.tags),
                    ideal_duration_minutes=ideal_duration,
                    actual_duration_minutes=actual_duration,
                )
                nodes.append(node)
                by_slot.setdefault(slot_index, []).append(node)

        ranked_index = {r.shop.name: r for r in ranked}
        shop_index = {r.shop.name: r.shop for r in ranked}
        edges: list[GraphEdge] = []
        debug_traces: list[str] = []
        for i in range(slot_count - 1):
            from_nodes = by_slot.get(i, [])
            if not from_nodes:
                continue
            for j in range(i + 1, slot_count):
                to_nodes = by_slot.get(j, [])
                if not to_nodes:
                    continue
                for a in from_nodes:
                    shop_a = shop_index[a.shop_name]
                    for b in to_nodes:
                        if a.shop_name == b.shop_name:
                            continue
                        # Layer-1 hard constraint uses only physical travel reachability.
                        # Buffer/jitter/cooldown are treated later as risk penalties (soft constraints).
                        travel_m = 18 + (8 * i)
                        fastest_finish_at = a.start_time + timedelta(
                            minutes=int(shop_a.base_wait_minutes) + int(shop_a.min_eat_minutes or shop_a.avg_eat_minutes)
                        )
                        ready_at = fastest_finish_at + timedelta(minutes=travel_m)
                        shop_b = shop_index[b.shop_name]
                        open_b = ItinerarySynthesizer._shop_open_at(b.start_time, shop_b)
                        if ready_at <= b.start_time and b.start_time >= open_b:
                            # Layer-2: risk-budget penalty on edge weight
                            # Queue risk: longer baseline queue => higher risk.
                            queue_risk = min(1.0, max(0.0, float(shop_b.base_wait_minutes) / 45.0))
                            # Buffer gap: if physical slack cannot cover expected buffer, penalize.
                            status = traffic.get_route_status(a.shop_name, b.shop_name)
                            slack_m = max(0.0, (b.start_time - ready_at).total_seconds() / 60.0)
                            expected_buffer_m = max(0.0, float(status.transport_buffer_minutes))
                            buffer_gap_m = max(0.0, expected_buffer_m - slack_m)
                            travel_buffer_gap = min(1.0, buffer_gap_m / 45.0)
                            base_score = float(ranked_index[b.shop_name].final_score)
                            weight = ScoringEngine.risk_adjusted_score(
                                base_score,
                                queue_risk=queue_risk,
                                travel_buffer_gap=travel_buffer_gap,
                            )
                            edges.append(GraphEdge(from_node_id=a.node_id, to_node_id=b.node_id, weight=weight))
                        else:
                            reasons: list[str] = []
                            if ready_at > b.start_time:
                                reasons.append(
                                    f"準備時間 (ready_at) {ready_at.strftime('%H:%M')} > 開始時間 (B.start) {b.start_time.strftime('%H:%M')}"
                                )
                            if b.start_time < open_b:
                                reasons.append(
                                    f"B.start {b.start_time.strftime('%H:%M')} < B.open_time {open_b.strftime('%H:%M')}"
                                )
                            reason_text = "；".join(reasons) if reasons else "未知原因"
                            debug_traces.append(
                                _dj(
                                    "graph_rejected_edge",
                                    from_shop=a.shop_name,
                                    to_shop=b.shop_name,
                                    detail=reason_text,
                                )
                            )

        return SpatioTemporalGraph(nodes=nodes, edges=edges, debug_traces=debug_traces)


def choose_health_backup(shops: list[ShopProfile], current: ShopProfile) -> ShopProfile | None:
    candidates = [s for s in shops if s.name != current.name]
    if not candidates:
        return None
    # Prefer lowest optimized impact, then higher customization.
    candidates.sort(
        key=lambda s: (ScoringEngine.optimized_health_impact(s), -max(0.0, min(1.0, s.customization_score)))
    )
    return candidates[0]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _fake_shop(name: str, **attrs: object) -> SimpleNamespace:
        """Minimal mock of a ShopProfile for feature verification."""
        defaults = {
            "name": name,
            "tags": [],
            "flavor_intensity": 0.5,
            "portion_strictness": 0.5,
            "review_count": 0,
            "authority_data": SimpleNamespace(tablelog_medal=""),
            "base_wait_minutes": 0,
            "price_level": 2.5,
        }
        merged = defaults.copy()
        merged.update(attrs)
        return SimpleNamespace(**merged)

    pool = [
        _fake_shop("店A", tags=["ramen"], review_count=150, base_wait_minutes=15, flavor_intensity=0.8),
        _fake_shop("店B", tags=["beef"], review_count=80, base_wait_minutes=30, flavor_intensity=0.6),
        _fake_shop("店C", tags=["cafe"], review_count=200, base_wait_minutes=10, flavor_intensity=0.4),
        _fake_shop("店D", tags=["ramen"], review_count=250, base_wait_minutes=25, flavor_intensity=0.7),
        _fake_shop("店E", tags=["sushi"], review_count=40, base_wait_minutes=5, flavor_intensity=0.5),
    ]
    ctx = {"preferred_tags": ["ramen"], "travel_minutes": 25}

    means, stds = compute_trip_frozen_scaling(pool, ctx)
    print("FROZEN_MEANS", means)
    print("FROZEN_STDS", stds)

    matrix = np.array([phi(c, ctx) for c in pool])
    norm = (matrix - means) / stds
    print("NORM_MEAN", np.round(np.mean(norm, axis=0), 6))
    print("NORM_STD", np.round(np.std(norm, axis=0), 6))

    new_shop = _fake_shop(
        "店X",
        tags=["udon"],
        review_count=400,
        base_wait_minutes=45,
        flavor_intensity=0.9,
    )
    raw_new = phi(new_shop, ctx)
    z_new = z_score(new_shop, ctx, (means, stds))
    print("NEW_RAW", raw_new)
    print("NEW_Z", np.round(z_new, 4))
    print("FROZEN_MEANS_AFTER_NEW", means)
    print("FROZEN_STDS_AFTER_NEW", stds)

