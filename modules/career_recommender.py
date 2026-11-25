"""Explainable, deterministic career recommendation engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from modules.skill_extractor import (
    SkillExtractionError,
    _load_taxonomy,
    extract_skills,
    normalize_text,
)


CORE_SKILL_WEIGHT = 0.70
SUPPORTING_SKILL_WEIGHT = 0.30
_CAREERS_PATH = Path(__file__).resolve().parent.parent / "data" / "careers.json"


class CareerCatalogError(ValueError):
    """Raised when the curated career catalog cannot be loaded or validated."""


class InvalidCareerRecommendationInputError(ValueError):
    """Raised when inputs passed to the career recommender are invalid."""


@dataclass(frozen=True)
class CareerProfile:
    """Validated career definition loaded from the curated catalog."""

    title: str
    description: str
    core_skills: tuple[str, ...]
    supporting_skills: tuple[str, ...]


@dataclass(frozen=True)
class CareerRecommendation:
    """Explainable career recommendation; scores are percentages from 0 to 100."""

    title: str
    description: str
    suitability_score: float
    core_coverage_score: float
    supporting_coverage_score: float
    matching_core_skills: list[str]
    matching_supporting_skills: list[str]
    missing_core_skills: list[str]
    missing_supporting_skills: list[str]
    next_skills_to_learn: list[str]
    explanation: str


@lru_cache(maxsize=1)
def _build_skill_alias_map() -> dict[str, str]:
    """Map normalized canonical names and aliases to canonical taxonomy names."""
    taxonomy = _load_taxonomy()
    alias_map: dict[str, str] = {}
    for category in taxonomy:
        for skill in category["skills"]:
            canonical_name = skill["name"]
            alias_map[normalize_text(canonical_name)] = canonical_name
            for alias in skill["aliases"]:
                alias_map[normalize_text(alias)] = canonical_name
    return alias_map


def _validate_skill_list(
    raw_skills: object,
    *,
    field_name: str,
    career_title: str,
    alias_map: dict[str, str],
    require_non_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(raw_skills, list):
        raise CareerCatalogError(
            f"Career '{career_title}' field '{field_name}' must be a list."
        )
    if require_non_empty and not raw_skills:
        raise CareerCatalogError(
            f"Career '{career_title}' must define at least one skill in '{field_name}'."
        )

    canonical_skills: list[str] = []
    seen: set[str] = set()
    for item in raw_skills:
        if not isinstance(item, str) or not item.strip():
            raise CareerCatalogError(
                f"Career '{career_title}' has an invalid skill entry in '{field_name}'."
            )
        normalized = normalize_text(item)
        canonical = alias_map.get(normalized)
        if canonical is None:
            raise CareerCatalogError(
                f"Career '{career_title}' references unknown taxonomy skill: {item}"
            )
        if canonical in seen:
            raise CareerCatalogError(
                f"Career '{career_title}' lists duplicate skill '{canonical}' in '{field_name}'."
            )
        seen.add(canonical)
        canonical_skills.append(canonical)
    return tuple(canonical_skills)


def load_career_catalog(catalog_path: Path = _CAREERS_PATH) -> tuple[CareerProfile, ...]:
    """Load and validate career profiles from JSON against the skill taxonomy."""
    if catalog_path == _CAREERS_PATH:
        return _load_default_career_catalog()
    return _read_and_validate_catalog(catalog_path)


@lru_cache(maxsize=1)
def _load_default_career_catalog() -> tuple[CareerProfile, ...]:
    return _read_and_validate_catalog(_CAREERS_PATH)


def _read_and_validate_catalog(catalog_path: Path) -> tuple[CareerProfile, ...]:
    try:
        with catalog_path.open(encoding="utf-8") as catalog_file:
            data = json.load(catalog_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CareerCatalogError("The career catalog could not be loaded.") from error

    careers = data.get("careers") if isinstance(data, dict) else None
    if not isinstance(careers, list) or not careers:
        raise CareerCatalogError("The career catalog must contain a non-empty 'careers' list.")

    try:
        alias_map = _build_skill_alias_map()
    except SkillExtractionError as error:
        raise CareerCatalogError("The skill taxonomy required by the career catalog is unavailable.") from error

    profiles: list[CareerProfile] = []
    seen_titles: set[str] = set()
    for entry in careers:
        if not isinstance(entry, dict):
            raise CareerCatalogError("Each career entry must be an object.")
        title = entry.get("title")
        description = entry.get("description")
        if not isinstance(title, str) or not title.strip():
            raise CareerCatalogError("Each career entry must have a non-empty 'title'.")
        if not isinstance(description, str) or not description.strip():
            raise CareerCatalogError(f"Career '{title}' must have a non-empty 'description'.")

        normalized_title = normalize_text(title)
        if normalized_title in seen_titles:
            raise CareerCatalogError(f"Duplicate career title in catalog: {title}")
        seen_titles.add(normalized_title)

        core_skills = _validate_skill_list(
            entry.get("core_skills"),
            field_name="core_skills",
            career_title=title.strip(),
            alias_map=alias_map,
            require_non_empty=True,
        )
        supporting_skills = _validate_skill_list(
            entry.get("supporting_skills", []),
            field_name="supporting_skills",
            career_title=title.strip(),
            alias_map=alias_map,
            require_non_empty=False,
        )
        overlap = set(core_skills) & set(supporting_skills)
        if overlap:
            raise CareerCatalogError(
                f"Career '{title.strip()}' lists skills in both core and supporting: "
                + ", ".join(sorted(overlap))
            )

        profiles.append(
            CareerProfile(
                title=title.strip(),
                description=description.strip(),
                core_skills=core_skills,
                supporting_skills=supporting_skills,
            )
        )

    return tuple(profiles)


def _validate_top_n(top_n: int | None) -> int | None:
    if top_n is None:
        return None
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise InvalidCareerRecommendationInputError(
            "top_n must be None or a positive integer."
        )
    return top_n


def normalize_candidate_skills(skills: Iterable[str]) -> list[str]:
    """Normalize and deduplicate candidate skills, ignoring unknown skills safely."""
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Iterable):
        raise InvalidCareerRecommendationInputError(
            "Candidate skills must be an iterable of skill strings (not a single string)."
        )

    alias_map = _build_skill_alias_map()
    canonical_skills: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        if not isinstance(skill, str):
            raise InvalidCareerRecommendationInputError(
                "Every candidate skill must be a string."
            )
        normalized = normalize_text(skill)
        if not normalized:
            continue
        canonical = alias_map.get(normalized)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        canonical_skills.append(canonical)
    return canonical_skills


def _build_explanation(
    profile: CareerProfile,
    matching_core: list[str],
    matching_supporting: list[str],
    missing_core: list[str],
    suitability_score: float,
) -> str:
    if not matching_core and not matching_supporting:
        return (
            f"No matching taxonomy skills were detected for {profile.title} "
            f"(heuristic alignment: {suitability_score:.1f}%). "
            f"Start by building core skills: {', '.join(missing_core[:3])}."
        )

    parts: list[str] = []
    if matching_core:
        parts.append(
            f"Matches {len(matching_core)} of {len(profile.core_skills)} core skills "
            f"({', '.join(matching_core)})"
        )
    else:
        parts.append(f"Matches 0 of {len(profile.core_skills)} core skills")

    if matching_supporting:
        parts.append(
            f"and {len(matching_supporting)} supporting skill(s) "
            f"({', '.join(matching_supporting)})"
        )

    summary = " ".join(parts) + "."
    if missing_core:
        summary += f" Key core skill gaps: {', '.join(missing_core)}."
    else:
        summary += " All core skills for this role are present."
    return summary


def recommend_careers_from_skills(
    skills: Iterable[str],
    top_n: int | None = None,
    *,
    catalog_path: Path = _CAREERS_PATH,
) -> list[CareerRecommendation]:
    """Rank career paths using weighted core (70%) and supporting (30%) skill coverage."""
    validated_top_n = _validate_top_n(top_n)
    candidate_skills = set(normalize_candidate_skills(skills))
    profiles = load_career_catalog(catalog_path)

    recommendations: list[CareerRecommendation] = []
    for profile in profiles:
        matching_core = [s for s in profile.core_skills if s in candidate_skills]
        missing_core = [s for s in profile.core_skills if s not in candidate_skills]
        matching_supporting = [
            s for s in profile.supporting_skills if s in candidate_skills
        ]
        missing_supporting = [
            s for s in profile.supporting_skills if s not in candidate_skills
        ]

        core_coverage = round(
            len(matching_core) / len(profile.core_skills) * 100, 1
        )
        if profile.supporting_skills:
            supporting_coverage = round(
                len(matching_supporting) / len(profile.supporting_skills) * 100, 1
            )
            suitability_score = round(
                (
                    (len(matching_core) / len(profile.core_skills)) * CORE_SKILL_WEIGHT
                    + (len(matching_supporting) / len(profile.supporting_skills))
                    * SUPPORTING_SKILL_WEIGHT
                )
                * 100,
                1,
            )
        else:
            supporting_coverage = 0.0
            suitability_score = core_coverage

        next_skills = missing_core + missing_supporting
        explanation = _build_explanation(
            profile,
            matching_core,
            matching_supporting,
            missing_core,
            suitability_score,
        )

        recommendations.append(
            CareerRecommendation(
                title=profile.title,
                description=profile.description,
                suitability_score=suitability_score,
                core_coverage_score=core_coverage,
                supporting_coverage_score=supporting_coverage,
                matching_core_skills=matching_core,
                matching_supporting_skills=matching_supporting,
                missing_core_skills=missing_core,
                missing_supporting_skills=missing_supporting,
                next_skills_to_learn=next_skills,
                explanation=explanation,
            )
        )

    recommendations.sort(
        key=lambda rec: (
            -rec.suitability_score,
            -len(rec.matching_core_skills),
            rec.title,
        )
    )
    if validated_top_n is not None:
        return recommendations[:validated_top_n]
    return recommendations


def recommend_careers_from_text(
    resume_text: str,
    top_n: int | None = None,
    *,
    catalog_path: Path = _CAREERS_PATH,
) -> list[CareerRecommendation]:
    """Extract skills from resume text and return ranked career recommendations."""
    if not isinstance(resume_text, str):
        raise InvalidCareerRecommendationInputError("Resume text must be a string.")
    _validate_top_n(top_n)
    extracted_skills = extract_skills(resume_text)
    return recommend_careers_from_skills(
        extracted_skills,
        top_n=top_n,
        catalog_path=catalog_path,
    )


def recommend_careers(
    skills: Iterable[str],
    top_n: int | None = None,
) -> list[dict[str, object]]:
    """Return ranked career recommendations as plain dictionaries."""
    return [
        asdict(recommendation)
        for recommendation in recommend_careers_from_skills(skills, top_n=top_n)
    ]
