"""Dependency-aware learning roadmaps for curated career profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from modules.career_recommender import (
    CareerCatalogError,
    CareerProfile,
    InvalidCareerRecommendationInputError,
    load_career_catalog,
    normalize_candidate_skills,
)
from modules.skill_extractor import (
    SkillExtractionError,
    _load_taxonomy,
    normalize_text,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LEARNING_CATALOG_PATH = _PROJECT_ROOT / "data" / "learning_paths.json"
_CAREERS_CATALOG_PATH = _PROJECT_ROOT / "data" / "careers.json"
_SKILLS_CATALOG_PATH = _PROJECT_ROOT / "data" / "skills.json"
_LEVEL_ORDER = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}


class LearningCatalogError(ValueError):
    """Raised when the learning catalog or its dependencies are invalid."""


class InvalidRoadmapInputError(ValueError):
    """Raised when a career or skill input cannot produce a roadmap."""


@dataclass(frozen=True)
class SkillLearningMetadata:
    """Validated learning guidance for one canonical taxonomy skill."""

    skill: str
    level: str
    prerequisites: tuple[str, ...]
    mini_project: str
    milestone: str


@dataclass(frozen=True)
class RoadmapStep:
    """A missing career skill and its learning guidance."""

    skill: str
    level: str
    prerequisites: tuple[str, ...]
    mini_project: str
    milestone: str
    is_core_skill: bool
    is_completed: bool


@dataclass(frozen=True)
class LearningRoadmap:
    """Immutable, ordered roadmap with stage groups and progress totals."""

    career_title: str
    steps: tuple[RoadmapStep, ...]
    beginner_steps: tuple[RoadmapStep, ...]
    intermediate_steps: tuple[RoadmapStep, ...]
    advanced_steps: tuple[RoadmapStep, ...]
    completed_skills: tuple[str, ...]
    completed_count: int
    total_count: int
    remaining_skills: tuple[str, ...]
    progress_percentage: float


def _read_json(path: Path, description: str) -> object:
    try:
        with path.open(encoding="utf-8") as json_file:
            return json.load(json_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LearningCatalogError(f"The {description} could not be loaded.") from error


def _validate_learning_entries(
    data: object,
    canonical_skills: set[str],
) -> tuple[SkillLearningMetadata, ...]:
    entries = data.get("learning_paths") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise LearningCatalogError(
            "The learning catalog must contain a non-empty 'learning_paths' list."
        )

    metadata: list[SkillLearningMetadata] = []
    seen_skills: set[str] = set()
    required_fields = {
        "skill",
        "level",
        "prerequisites",
        "mini_project",
        "milestone",
    }

    for entry in entries:
        if not isinstance(entry, dict):
            raise LearningCatalogError("Each learning path must be an object.")
        if not required_fields.issubset(entry):
            raise LearningCatalogError(
                "Each learning path must define skill, level, prerequisites, "
                "mini_project, and milestone."
            )

        skill = entry["skill"]
        if not isinstance(skill, str) or not skill.strip():
            raise LearningCatalogError("Each learning path needs a non-empty skill name.")
        if skill not in canonical_skills:
            raise LearningCatalogError(
                f"Learning path references non-canonical taxonomy skill: {skill}"
            )
        if skill in seen_skills:
            raise LearningCatalogError(f"Duplicate learning path for skill: {skill}")
        seen_skills.add(skill)

        level = entry["level"]
        if not isinstance(level, str) or level not in _LEVEL_ORDER:
            raise LearningCatalogError(
                f"Learning path for '{skill}' has an invalid level."
            )

        raw_prerequisites = entry["prerequisites"]
        if not isinstance(raw_prerequisites, list):
            raise LearningCatalogError(
                f"Prerequisites for '{skill}' must be a list."
            )
        if any(
            not isinstance(prerequisite, str) or not prerequisite.strip()
            for prerequisite in raw_prerequisites
        ):
            raise LearningCatalogError(
                f"Learning path for '{skill}' has an invalid prerequisite."
            )
        if len(raw_prerequisites) != len(set(raw_prerequisites)):
            raise LearningCatalogError(
                f"Learning path for '{skill}' contains duplicate prerequisites."
            )
        if any(prerequisite not in canonical_skills for prerequisite in raw_prerequisites):
            invalid_prerequisite = next(
                prerequisite
                for prerequisite in raw_prerequisites
                if prerequisite not in canonical_skills
            )
            raise LearningCatalogError(
                f"Learning path for '{skill}' references non-canonical prerequisite: "
                f"{invalid_prerequisite}"
            )

        mini_project = entry["mini_project"]
        milestone = entry["milestone"]
        if not isinstance(mini_project, str) or not mini_project.strip():
            raise LearningCatalogError(
                f"Learning path for '{skill}' needs a non-empty mini_project."
            )
        if not isinstance(milestone, str) or not milestone.strip():
            raise LearningCatalogError(
                f"Learning path for '{skill}' needs a non-empty milestone."
            )

        metadata.append(
            SkillLearningMetadata(
                skill=skill,
                level=level,
                prerequisites=tuple(raw_prerequisites),
                mini_project=mini_project.strip(),
                milestone=milestone.strip(),
            )
        )

    return tuple(metadata)


def _validate_prerequisite_graph(
    metadata: tuple[SkillLearningMetadata, ...],
) -> None:
    by_skill = {item.skill: item for item in metadata}
    for item in metadata:
        for prerequisite in item.prerequisites:
            prerequisite_metadata = by_skill.get(prerequisite)
            if prerequisite_metadata is None:
                raise LearningCatalogError(
                    f"Prerequisite '{prerequisite}' for '{item.skill}' "
                    "has no learning metadata."
                )
            if _LEVEL_ORDER[prerequisite_metadata.level] > _LEVEL_ORDER[item.level]:
                raise LearningCatalogError(
                    f"Prerequisite '{prerequisite}' for '{item.skill}' "
                    "is assigned a later learning level."
                )

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(skill: str) -> None:
        if skill in visiting:
            cycle_start = visiting.index(skill)
            cycle = visiting[cycle_start:] + [skill]
            raise LearningCatalogError(
                "Circular learning prerequisites detected: " + " -> ".join(cycle)
            )
        if skill in visited:
            return

        visiting.append(skill)
        for prerequisite in by_skill[skill].prerequisites:
            visit(prerequisite)
        visiting.pop()
        visited.add(skill)

    for item in metadata:
        visit(item.skill)


def load_learning_catalog(
    catalog_path: str | Path = _LEARNING_CATALOG_PATH,
    *,
    careers_path: str | Path = _CAREERS_CATALOG_PATH,
) -> tuple[SkillLearningMetadata, ...]:
    """Load learning metadata and validate it against taxonomy and career data."""
    try:
        resolved_catalog_path = Path(catalog_path)
        resolved_careers_path = Path(careers_path)
    except TypeError as error:
        raise LearningCatalogError("Catalog paths must be filesystem paths.") from error

    if not _SKILLS_CATALOG_PATH.is_file():
        raise LearningCatalogError("The skills taxonomy could not be loaded.")
    if not resolved_careers_path.is_file():
        raise LearningCatalogError("The career catalog could not be loaded.")

    catalog_data = _read_json(resolved_catalog_path, "learning catalog")
    try:
        taxonomy = _load_taxonomy()
    except SkillExtractionError as error:
        raise LearningCatalogError("The skills taxonomy is unavailable.") from error
    canonical_skills = {
        skill["name"]
        for category in taxonomy
        for skill in category["skills"]
    }

    try:
        careers = load_career_catalog(resolved_careers_path)
    except (CareerCatalogError, SkillExtractionError) as error:
        raise LearningCatalogError("The career catalog is invalid or unavailable.") from error

    metadata = _validate_learning_entries(catalog_data, canonical_skills)
    catalog_skills = {item.skill for item in metadata}
    required_career_skills = {
        skill
        for career in careers
        for skill in career.core_skills + career.supporting_skills
    }
    missing_skills = sorted(required_career_skills - catalog_skills)
    if missing_skills:
        raise LearningCatalogError(
            "Learning metadata is missing career skills: " + ", ".join(missing_skills)
        )

    _validate_prerequisite_graph(metadata)
    return metadata


def _normalize_skill_list(
    skills: Iterable[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Iterable):
        raise InvalidRoadmapInputError(
            f"{field_name} must be an iterable of skill strings, not a single string."
        )

    canonical_skills: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        if not isinstance(skill, str):
            raise InvalidRoadmapInputError(
                f"Every entry in {field_name} must be a string."
            )
        if not normalize_text(skill):
            continue
        try:
            normalized = normalize_candidate_skills([skill])
        except InvalidCareerRecommendationInputError as error:
            raise InvalidRoadmapInputError(str(error)) from error
        if not normalized:
            raise InvalidRoadmapInputError(
                f"Unknown skill in {field_name}: {skill.strip()}"
            )
        canonical = normalized[0]
        if canonical not in seen:
            seen.add(canonical)
            canonical_skills.append(canonical)
    return tuple(canonical_skills)


def _find_career_profile(
    career_title: str,
    careers: tuple[CareerProfile, ...],
) -> CareerProfile:
    if not isinstance(career_title, str) or not career_title.strip():
        raise InvalidRoadmapInputError("Career title must be a non-empty string.")

    normalized_title = normalize_text(career_title)
    for career in careers:
        if normalize_text(career.title) == normalized_title:
            return career
    raise InvalidRoadmapInputError(f"Unknown career: {career_title.strip()}")


def _order_missing_skills(
    profile: CareerProfile,
    metadata_by_skill: dict[str, SkillLearningMetadata],
    candidate_skills: set[str],
) -> tuple[str, ...]:
    career_skills = profile.core_skills + profile.supporting_skills
    remaining = set(career_skills) - candidate_skills
    core_skills = set(profile.core_skills)
    catalog_order = {skill: index for index, skill in enumerate(career_skills)}
    ordered: list[str] = []

    while remaining:
        available = [
            skill
            for skill in remaining
            if not (set(metadata_by_skill[skill].prerequisites) & remaining)
        ]
        if not available:
            raise LearningCatalogError(
                f"Career '{profile.title}' has unresolved learning dependencies."
            )

        next_skill = min(
            available,
            key=lambda skill: (
                _LEVEL_ORDER[metadata_by_skill[skill].level],
                skill not in core_skills,
                catalog_order[skill],
            ),
        )
        ordered.append(next_skill)
        remaining.remove(next_skill)

    return tuple(ordered)


def generate_learning_roadmap(
    career_title: str,
    candidate_skills: Iterable[str],
    completed_skills: Iterable[str] = (),
    *,
    catalog_path: str | Path = _LEARNING_CATALOG_PATH,
    careers_path: str | Path = _CAREERS_CATALOG_PATH,
) -> LearningRoadmap:
    """Generate an ordered roadmap for skills missing from one career profile."""
    catalog = load_learning_catalog(catalog_path, careers_path=careers_path)
    careers = load_career_catalog(Path(careers_path))
    profile = _find_career_profile(career_title, careers)

    candidate_skill_tuple = _normalize_skill_list(
        candidate_skills,
        field_name="candidate_skills",
    )
    completed_skill_tuple = _normalize_skill_list(
        completed_skills,
        field_name="completed_skills",
    )

    metadata_by_skill = {item.skill: item for item in catalog}
    candidate_skill_set = set(candidate_skill_tuple)
    ordered_skills = _order_missing_skills(
        profile,
        metadata_by_skill,
        candidate_skill_set,
    )
    completed_skill_set = set(completed_skill_tuple)
    invalid_completed = sorted(completed_skill_set - set(ordered_skills))
    if invalid_completed:
        raise InvalidRoadmapInputError(
            "Completed skills must be missing skills in the selected career roadmap: "
            + ", ".join(invalid_completed)
        )

    core_skills = set(profile.core_skills)
    steps = tuple(
        RoadmapStep(
            skill=skill,
            level=metadata_by_skill[skill].level,
            prerequisites=metadata_by_skill[skill].prerequisites,
            mini_project=metadata_by_skill[skill].mini_project,
            milestone=metadata_by_skill[skill].milestone,
            is_core_skill=skill in core_skills,
            is_completed=skill in completed_skill_set,
        )
        for skill in ordered_skills
    )
    completed_in_order = tuple(
        step.skill for step in steps if step.is_completed
    )
    remaining_skills = tuple(
        step.skill for step in steps if not step.is_completed
    )
    completed_count = len(completed_in_order)
    total_count = len(steps)
    progress_percentage = (
        round(completed_count / total_count * 100, 1)
        if total_count
        else 100.0
    )

    return LearningRoadmap(
        career_title=profile.title,
        steps=steps,
        beginner_steps=tuple(step for step in steps if step.level == "Beginner"),
        intermediate_steps=tuple(
            step for step in steps if step.level == "Intermediate"
        ),
        advanced_steps=tuple(step for step in steps if step.level == "Advanced"),
        completed_skills=completed_in_order,
        completed_count=completed_count,
        total_count=total_count,
        remaining_skills=remaining_skills,
        progress_percentage=progress_percentage,
    )
