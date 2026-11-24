"""Taxonomy-based skill extraction and resume/job skill comparison."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypedDict


_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.json"


class SkillExtractionError(ValueError):
    """Raised when the curated skill taxonomy cannot be loaded or validated."""


class SkillEntry(TypedDict):
    name: str
    aliases: list[str]


class CategoryEntry(TypedDict):
    name: str
    skills: list[SkillEntry]


@dataclass(frozen=True)
class SkillComparison:
    """Skills found in each input and their exact taxonomy-name overlap."""

    candidate_skills: list[str]
    required_skills: list[str]
    matching_skills: list[str]
    missing_skills: list[str]
    candidate_by_category: dict[str, list[str]]
    required_by_category: dict[str, list[str]]
    matching_by_category: dict[str, list[str]]
    missing_by_category: dict[str, list[str]]


def normalize_text(text: str) -> str:
    """Apply Unicode and case normalization before keyword matching."""
    if not isinstance(text, str):
        raise TypeError("Skill extraction input must be text.")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


@lru_cache(maxsize=1)
def _load_taxonomy() -> tuple[CategoryEntry, ...]:
    try:
        with _TAXONOMY_PATH.open(encoding="utf-8") as taxonomy_file:
            data = json.load(taxonomy_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SkillExtractionError("The skills taxonomy could not be loaded.") from error

    categories = data.get("categories") if isinstance(data, dict) else None
    if not isinstance(categories, list) or not categories:
        raise SkillExtractionError("The skills taxonomy must contain categories.")

    validated_categories: list[CategoryEntry] = []
    category_names: set[str] = set()
    skill_names: set[str] = set()
    aliases: set[str] = set()
    for category in categories:
        if not isinstance(category, dict):
            raise SkillExtractionError("Each taxonomy category must be an object.")
        name = category.get("name")
        skills = category.get("skills")
        if not isinstance(name, str) or not name.strip() or not isinstance(skills, list):
            raise SkillExtractionError("Each category needs a name and a skills list.")
        if name in category_names:
            raise SkillExtractionError(f"Duplicate taxonomy category: {name}")
        category_names.add(name)

        validated_skills: list[SkillEntry] = []
        for skill in skills:
            if not isinstance(skill, dict):
                raise SkillExtractionError("Each taxonomy skill must be an object.")
            skill_name = skill.get("name")
            skill_aliases = skill.get("aliases", [])
            if (
                not isinstance(skill_name, str)
                or not skill_name.strip()
                or not isinstance(skill_aliases, list)
                or any(not isinstance(alias, str) or not alias.strip() for alias in skill_aliases)
            ):
                raise SkillExtractionError("Each skill needs a name and valid aliases.")

            normalized_name = normalize_text(skill_name)
            if normalized_name in skill_names:
                raise SkillExtractionError(f"Duplicate taxonomy skill: {skill_name}")
            skill_names.add(normalized_name)

            unique_aliases = list(dict.fromkeys(alias.strip() for alias in skill_aliases))
            for term in [skill_name.strip(), *unique_aliases]:
                normalized_term = normalize_text(term)
                if normalized_term in aliases:
                    raise SkillExtractionError(f"Ambiguous taxonomy alias: {term}")
                aliases.add(normalized_term)
            validated_skills.append({"name": skill_name.strip(), "aliases": unique_aliases})

        validated_categories.append({"name": name.strip(), "skills": validated_skills})

    return tuple(validated_categories)


def _matches_term(text: str, term: str) -> bool:
    escaped_term = re.escape(normalize_text(term))
    start_boundary = r"(?<!\w)"
    if term[-1].isalnum() or term[-1] == "_":
        end_boundary = r"(?!\w)"
    else:
        end_boundary = r"(?![^\W\d])"
    return re.search(start_boundary + escaped_term + end_boundary, text) is not None


def extract_skills_by_category(text: str) -> dict[str, list[str]]:
    """Return each taxonomy category with skills detected in the supplied text."""
    normalized_text = normalize_text(text)
    taxonomy = _load_taxonomy()
    found_by_category: dict[str, list[str]] = {}
    for category in taxonomy:
        found: list[str] = []
        for skill in category["skills"]:
            if len(skill["name"]) == 1:
                # Single-letter language names need explicit context to avoid noise.
                terms = skill["aliases"]
            else:
                terms = [skill["name"], *skill["aliases"]]
            if any(_matches_term(normalized_text, term) for term in terms):
                found.append(skill["name"])
        found_by_category[category["name"]] = found
    return found_by_category


def extract_skills(text: str) -> list[str]:
    """Extract unique canonical skill names from resume or job-description text."""
    return [
        skill
        for category_skills in extract_skills_by_category(text).values()
        for skill in category_skills
    ]


def compare_skills(resume_text: str, job_description: str) -> SkillComparison:
    """Compare candidate and job skills using exact canonical taxonomy names."""
    candidate_by_category = extract_skills_by_category(resume_text)
    required_by_category = extract_skills_by_category(job_description)
    candidate_skills = [
        skill for skills in candidate_by_category.values() for skill in skills
    ]
    required_skills = [
        skill for skills in required_by_category.values() for skill in skills
    ]
    candidate_set = set(candidate_skills)
    matching_skills = [skill for skill in required_skills if skill in candidate_set]
    missing_skills = [skill for skill in required_skills if skill not in candidate_set]
    matching_set = set(matching_skills)
    missing_set = set(missing_skills)

    return SkillComparison(
        candidate_skills=candidate_skills,
        required_skills=required_skills,
        matching_skills=matching_skills,
        missing_skills=missing_skills,
        candidate_by_category=candidate_by_category,
        required_by_category=required_by_category,
        matching_by_category={
            category: [skill for skill in skills if skill in matching_set]
            for category, skills in required_by_category.items()
        },
        missing_by_category={
            category: [skill for skill in skills if skill in missing_set]
            for category, skills in required_by_category.items()
        },
    )
