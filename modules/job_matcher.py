"""Explainable resume-to-job matching using taxonomy skills and TF-IDF."""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from modules.skill_extractor import SkillComparison, compare_skills, normalize_text


SKILL_COVERAGE_WEIGHT = 0.70
TEXT_SIMILARITY_WEIGHT = 0.30
TFIDF_BACKEND_ENV_VAR = "AI_CAREER_NAVIGATOR_TFIDF_BACKEND"


class InvalidJobMatchingInputError(ValueError):
    """Raised when a resume or job description is not text."""


@dataclass(frozen=True)
class JobMatchAnalysis:
    """Explainable matching results; scores are percentages from 0 to 100."""

    overall_match_score: float | None
    skill_coverage_score: float | None
    text_similarity_score: float | None
    candidate_skills: list[str]
    required_skills: list[str]
    matching_skills: list[str]
    missing_skills: list[str]
    additional_candidate_skills: list[str]
    candidate_skills_by_category: dict[str, list[str]]
    required_skills_by_category: dict[str, list[str]]
    matching_skills_by_category: dict[str, list[str]]
    missing_skills_by_category: dict[str, list[str]]
    additional_skills_by_category: dict[str, list[str]]
    text_similarity_method: str
    strengths: list[str]
    improvement_suggestions: list[str]
    score_unavailable_reason: str | None


def _python_tfidf_similarity(resume_text: str, job_description: str) -> float | None:
    """Calculate unigram/bigram TF-IDF cosine similarity without binary extensions."""
    documents = [resume_text, job_description]
    analyzed_documents: list[list[str]] = []
    for document in documents:
        decomposed = unicodedata.normalize("NFKD", document)
        unaccented = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        tokens = re.findall(r"(?u)\b\w\w+\b", unaccented.casefold())
        analyzed_documents.append(
            tokens
            + [
                f"{tokens[index]} {tokens[index + 1]}"
                for index in range(len(tokens) - 1)
            ]
        )
    if any(not terms for terms in analyzed_documents):
        return None

    document_frequencies = Counter(
        term
        for document_terms in analyzed_documents
        for term in set(document_terms)
    )
    document_count = len(analyzed_documents)
    inverse_document_frequencies = {
        term: math.log((1 + document_count) / (1 + frequency)) + 1
        for term, frequency in document_frequencies.items()
    }

    vectors: list[dict[str, float]] = []
    for document_terms in analyzed_documents:
        term_counts = Counter(document_terms)
        vector = {
            term: count * inverse_document_frequencies[term]
            for term, count in term_counts.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        vectors.append({term: weight / norm for term, weight in vector.items()})

    similarity = sum(
        weight * vectors[1].get(term, 0.0)
        for term, weight in vectors[0].items()
    )
    return round(similarity * 100, 1)


def _calculate_text_similarity(
    resume_text: str,
    job_description: str,
) -> tuple[float | None, str]:
    """Return lexical TF-IDF cosine similarity and identify its implementation."""
    if not resume_text or not job_description:
        return None, "Not calculated (insufficient text)"

    backend = os.environ.get(TFIDF_BACKEND_ENV_VAR, "standard-library").strip().casefold()
    if backend == "standard-library":
        return (
            _python_tfidf_similarity(resume_text, job_description),
            "Standard-library TF-IDF",
        )
    if backend != "scikit-learn":
        raise ValueError(
            f"{TFIDF_BACKEND_ENV_VAR} must be 'standard-library' or 'scikit-learn'."
        )

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as error:
        raise ImportError(
            "Scikit-learn could not be loaded for TF-IDF. "
            "Use the standard-library backend or repair the Python environment."
        ) from error

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
    analyzer = vectorizer.build_analyzer()
    if not analyzer(resume_text) or not analyzer(job_description):
        return None, "Scikit-learn TF-IDF (no usable terms)"
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0])
    return round(similarity * 100, 1), "Scikit-learn TF-IDF"


def _category_difference(
    candidate_by_category: dict[str, list[str]],
    required_skills: set[str],
) -> dict[str, list[str]]:
    """Keep candidate skills that were not detected as job requirements."""
    return {
        category: [skill for skill in skills if skill not in required_skills]
        for category, skills in candidate_by_category.items()
    }


def _build_strengths_and_suggestions(
    comparison: SkillComparison,
    additional_skills: list[str],
) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    suggestions: list[str] = []
    if comparison.matching_skills:
        strengths.append(
            "The resume explicitly mentions these detected job skills: "
            + ", ".join(comparison.matching_skills)
            + "."
        )
    if additional_skills:
        strengths.append(
            "The resume also mentions skills not detected in the job description: "
            + ", ".join(additional_skills)
            + "."
        )
    if comparison.missing_skills:
        suggestions.append(
            "If you have experience with these requirements, make it explicit in "
            "the resume: "
            + ", ".join(comparison.missing_skills)
            + "."
        )
    elif not comparison.required_skills:
        suggestions.append(
            "No skills from the curated taxonomy were detected in the job description; "
            "include specific requirements for a more informative comparison."
        )
    if not comparison.candidate_skills:
        suggestions.append(
            "No skills from the curated taxonomy were detected in the resume text; "
            "check the extracted text or expand the taxonomy."
        )
    return strengths, suggestions


def match_job(resume_text: str, job_description: str) -> JobMatchAnalysis:
    """Compare resume text with a job description using skills and TF-IDF.

    The overall heuristic score is 70% skill coverage and 30% lexical TF-IDF
    cosine similarity. It is not a probability of being hired.
    """
    if not isinstance(resume_text, str) or not isinstance(job_description, str):
        raise InvalidJobMatchingInputError(
            "Resume text and job description must both be strings."
        )

    normalized_resume = normalize_text(resume_text)
    normalized_job = normalize_text(job_description)
    comparison = compare_skills(normalized_resume, normalized_job)

    required_skills = set(comparison.required_skills)
    additional_skills = [
        skill
        for skill in comparison.candidate_skills
        if skill not in required_skills
    ]
    additional_by_category = _category_difference(
        comparison.candidate_by_category,
        required_skills,
    )

    text_similarity_score, text_similarity_method = _calculate_text_similarity(
        normalized_resume,
        normalized_job,
    )
    skill_coverage_score: float | None = None
    if normalized_resume and comparison.required_skills:
        skill_coverage_score = round(
            len(comparison.matching_skills) / len(comparison.required_skills) * 100,
            1,
        )

    overall_match_score: float | None = None
    if skill_coverage_score is not None and text_similarity_score is not None:
        overall_match_score = round(
            skill_coverage_score * SKILL_COVERAGE_WEIGHT
            + text_similarity_score * TEXT_SIMILARITY_WEIGHT,
            1,
        )

    score_unavailable_reason: str | None = None
    if not normalized_resume or not normalized_job:
        score_unavailable_reason = (
            "Provide non-empty resume text and a job description to calculate a score."
        )
    elif not comparison.required_skills:
        score_unavailable_reason = (
            "No required skills from the curated taxonomy were detected; "
            "skill coverage and the overall score are unavailable."
        )
    elif skill_coverage_score is None or text_similarity_score is None:
        score_unavailable_reason = (
            "The inputs did not contain enough usable text to calculate both score components."
        )

    strengths, improvement_suggestions = _build_strengths_and_suggestions(
        comparison,
        additional_skills,
    )

    return JobMatchAnalysis(
        overall_match_score=overall_match_score,
        skill_coverage_score=skill_coverage_score,
        text_similarity_score=text_similarity_score,
        candidate_skills=comparison.candidate_skills,
        required_skills=comparison.required_skills,
        matching_skills=comparison.matching_skills,
        missing_skills=comparison.missing_skills,
        additional_candidate_skills=additional_skills,
        candidate_skills_by_category=comparison.candidate_by_category,
        required_skills_by_category=comparison.required_by_category,
        matching_skills_by_category=comparison.matching_by_category,
        missing_skills_by_category=comparison.missing_by_category,
        additional_skills_by_category=additional_by_category,
        text_similarity_method=text_similarity_method,
        strengths=strengths,
        improvement_suggestions=improvement_suggestions,
        score_unavailable_reason=score_unavailable_reason,
    )
