"""Core recommendation engine for DecodeLabs Project 3.

The engine uses transparent, content-based similarity:
1. User preferences and item attributes are converted into weighted tokens.
2. Cosine similarity measures preference alignment.
3. A small normalized quality signal (rating) is used only as a tie-breaker.
4. Results include an explanation so recommendations are interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import math


@dataclass(frozen=True)
class Resource:
    id: str
    title: str
    topics: tuple[str, ...]
    difficulty: str
    format: str
    style: str
    rating: float


@dataclass(frozen=True)
class UserProfile:
    interests: tuple[str, ...]
    difficulty: str | None = None
    format: str | None = None
    style: str | None = None


class RecommendationEngine:
    """Simple explainable content-based recommender."""

    TOPIC_WEIGHT = 3.0
    DIFFICULTY_WEIGHT = 1.5
    FORMAT_WEIGHT = 1.0
    STYLE_WEIGHT = 1.0

    def __init__(self, resources: Iterable[Resource]):
        self.resources = list(resources)
        if not self.resources:
            raise ValueError("At least one resource is required.")

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.strip().lower().split())

    @classmethod
    def _profile_vector(cls, profile: UserProfile) -> dict[str, float]:
        vector: dict[str, float] = {}
        for interest in profile.interests:
            token = cls._normalize(interest)
            if token:
                vector[f"topic:{token}"] = cls.TOPIC_WEIGHT

        if profile.difficulty:
            vector[f"difficulty:{cls._normalize(profile.difficulty)}"] = cls.DIFFICULTY_WEIGHT
        if profile.format:
            vector[f"format:{cls._normalize(profile.format)}"] = cls.FORMAT_WEIGHT
        if profile.style:
            vector[f"style:{cls._normalize(profile.style)}"] = cls.STYLE_WEIGHT
        return vector

    @classmethod
    def _resource_vector(cls, resource: Resource) -> dict[str, float]:
        vector: dict[str, float] = {}
        for topic in resource.topics:
            vector[f"topic:{cls._normalize(topic)}"] = cls.TOPIC_WEIGHT
        vector[f"difficulty:{cls._normalize(resource.difficulty)}"] = cls.DIFFICULTY_WEIGHT
        vector[f"format:{cls._normalize(resource.format)}"] = cls.FORMAT_WEIGHT
        vector[f"style:{cls._normalize(resource.style)}"] = cls.STYLE_WEIGHT
        return vector

    @staticmethod
    def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0

        common = set(a).intersection(b)
        dot = sum(a[k] * b[k] for k in common)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @classmethod
    def _explain(cls, profile: UserProfile, resource: Resource) -> list[str]:
        reasons: list[str] = []
        interests = {cls._normalize(x) for x in profile.interests}
        matched_topics = [t for t in resource.topics if cls._normalize(t) in interests]
        if matched_topics:
            reasons.append("Matched interests: " + ", ".join(matched_topics))
        if profile.difficulty and cls._normalize(profile.difficulty) == cls._normalize(resource.difficulty):
            reasons.append(f"Preferred difficulty: {resource.difficulty}")
        if profile.format and cls._normalize(profile.format) == cls._normalize(resource.format):
            reasons.append(f"Preferred format: {resource.format}")
        if profile.style and cls._normalize(profile.style) == cls._normalize(resource.style):
            reasons.append(f"Preferred learning style: {resource.style}")
        if not reasons:
            reasons.append("Closest available attribute combination")
        return reasons

    def recommend(self, profile: UserProfile, top_k: int = 5) -> list[dict]:
        """Return top-k ranked recommendations with transparent scores."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        user_vec = self._profile_vector(profile)
        if not user_vec:
            raise ValueError("Please provide at least one preference.")

        scored = []
        for resource in self.resources:
            similarity = self._cosine_similarity(user_vec, self._resource_vector(resource))
            # Rating is intentionally a very small tie-breaker; preference similarity dominates.
            quality_bonus = (resource.rating / 5.0) * 0.02
            final_score = min(1.0, similarity + quality_bonus)

            scored.append(
                {
                    "id": resource.id,
                    "title": resource.title,
                    "topics": list(resource.topics),
                    "difficulty": resource.difficulty,
                    "format": resource.format,
                    "style": resource.style,
                    "rating": resource.rating,
                    "similarity": round(similarity, 4),
                    "score": round(final_score, 4),
                    "match_percent": round(final_score * 100, 1),
                    "reasons": self._explain(profile, resource),
                }
            )

        scored.sort(key=lambda x: (x["score"], x["rating"]), reverse=True)
        return scored[: min(top_k, len(scored))]


def load_resources(csv_path: str | Path) -> list[Resource]:
    csv_path = Path(csv_path)
    resources: list[Resource] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            resources.append(
                Resource(
                    id=row["id"],
                    title=row["title"],
                    topics=tuple(x.strip() for x in row["topics"].split(";") if x.strip()),
                    difficulty=row["difficulty"],
                    format=row["format"],
                    style=row["style"],
                    rating=float(row["rating"]),
                )
            )
    return resources


def default_engine() -> RecommendationEngine:
    data_path = Path(__file__).resolve().parent / "data" / "resources.csv"
    return RecommendationEngine(load_resources(data_path))
