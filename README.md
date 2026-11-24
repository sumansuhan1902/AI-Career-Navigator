# AI Career Navigator

AI Career Navigator is an early Streamlit prototype for a career-guidance platform. It extracts resume text, identifies taxonomy skills, and provides an explainable heuristic comparison with a target job description. Career recommendations are not implemented.

## Problem Statement

Job seekers often need to translate varied experience into role-specific skills and compare that experience with changing job requirements. The intended platform will make that comparison easier to inspect and act on, while keeping recommendations grounded in evidence rather than presenting unsupported scores.

## Current Features

- Streamlit dashboard with sidebar navigation and workspace status indicators.
- In-memory PDF resume text extraction with PyMuPDF, with a 10 MB upload limit, a 20-page limit, PDF signature validation, and clear messages for empty, scanned, password-protected, oversized, and invalid PDFs.
- Extracted resume text displayed in a read-only text area. Uploaded resume bytes are not written to disk or persisted.
- Curated, JSON-backed skill taxonomy across programming, AI/ML, data science, databases, web development, frameworks/libraries, developer tools, and soft skills.
- Case-insensitive skill detection using canonical names, aliases, and word-boundary-aware regular expressions.
- Categorized candidate and job skills, plus exact-name matching and missing-skill lists, on the Skill Analysis page.
- On-demand job matching combining taxonomy skill coverage with TF-IDF cosine similarity, with categorized matches, missing requirements, additional candidate skills, strengths, and improvement suggestions.
- Resume text and job-description input held only in the active Streamlit session.
- Clearly marked placeholder for career recommendations.
- SQLite connection helper for future persistence; it is not currently used by the app.

Skill extraction is deterministic taxonomy-based keyword matching, not an AI model or semantic NLP system. TF-IDF similarity is lexical and does not infer meaning. The overall match score is a heuristic compatibility estimate, not a probability of being hired. The engine does not evaluate proficiency or account for context and negation. The remaining career recommendation module raises `NotImplementedError` instead of returning sample or fabricated results.

## Technology Stack

- Python 3.10 or later
- Streamlit for the dashboard
- Pandas and NumPy remain available for future data preparation; the default matcher uses built-in Python data structures.
- SQLite through Python's standard-library `sqlite3` module
- PyMuPDF (`pymupdf`) for PDF text extraction
- Python regular expressions and a curated JSON taxonomy for deterministic skill extraction
- Optional scikit-learn TF-IDF vectorization and cosine similarity for lexical resume/job comparison

## Job Matching Engine

The Skill Analysis page runs the comparison only when **Analyze Job Match** is clicked. The engine reuses the skill extractor to identify canonical candidate and required skills, then reports:

- Matching skills and missing detected requirements
- Additional candidate skills not detected in the job description
- Strengths and improvement suggestions grounded in those results
- Skill coverage, TF-IDF text similarity, and a weighted overall estimate

### Matching algorithm

Skill coverage is computed only when resume text and at least one required taxonomy skill are available:

```text
Skill coverage (%) = matching required skills / total detected required skills * 100
```

Text similarity is the cosine similarity of L2-normalized TF-IDF vectors (unigrams and bigrams), expressed as a percentage. The default standard-library implementation avoids binary dependency issues in the current Windows environment. Set `AI_CAREER_NAVIGATOR_TFIDF_BACKEND=scikit-learn` to use scikit-learn's vectorizer and cosine similarity. The result measures shared vocabulary, not semantic similarity.

```text
Overall estimate (%) = 0.70 * skill coverage + 0.30 * text similarity
```

The overall estimate is unavailable if inputs are empty, required taxonomy skills are not detected, or TF-IDF cannot form usable vectors. Scores are heuristic comparisons of the supplied text, not hiring probabilities.

### Example

```python
from modules.job_matcher import match_job

analysis = match_job(
    "Python developer with experience building REST APIs.",
    "Looking for a Python developer with REST API experience and SQL.",
)

print(analysis.overall_match_score)
print(analysis.matching_skills)
print(analysis.missing_skills)
```

The taxonomy is intentionally curated and finite; unlisted skills and aliases will not be detected. Keyword extraction can miss paraphrases or match contextually irrelevant uses of a term. TF-IDF is sensitive to wording and cannot assess skill depth, proficiency, or experience quality. Pandas remains in the project dependency set but is not required for the current matching result structures.

## Installation

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
```

`requirements.lock.txt` pins the resolved dependency versions and hashes. Update it intentionally when changing `requirements.txt`; the lock file is the reproducible installation source.

Run the application:

```powershell
python -m streamlit run app.py
```

Run the focused tests:

```powershell
python -m unittest discover -s tests -v
```

## Project Structure

```text
AI-Career-Navigator/
├── app.py                       # Streamlit dashboard and input workflow
├── requirements.txt             # Runtime and planned data/model dependencies
├── README.md                    # Project overview, setup, and implementation status
├── .gitignore                   # Python, local environment, secrets, and database exclusions
├── modules/
│   ├── __init__.py              # Analysis package marker
│   ├── resume_parser.py         # In-memory PDF text extraction
│   ├── skill_extractor.py       # Taxonomy-based skill extraction and comparison
│   ├── job_matcher.py           # Skill coverage, TF-IDF similarity, and explainable results
│   └── career_recommender.py    # Future career recommendation interface
├── database/
│   └── db.py                    # SQLite connection helper; persistence not yet wired into the app
├── data/
│   └── skills.json              # Curated canonical skills and aliases by category
└── tests/
    └── test_app.py              # Tests PDF parsing, skill extraction, and job matching
```

## Security and runtime notes

- PDF uploads are limited to 10 MB in Streamlit and the parser, and documents over 20 pages are rejected before text extraction.
- The parser validates a PDF header in the first 1,024 bytes and processes the upload in memory without writing the original file to disk.
- Windows Application Control blocked the installed scikit-learn/SciPy binary import in the development environment. The default standard-library TF-IDF implementation remains the verified backend; the optional scikit-learn backend has not been verified on a supported environment.

## Future Enhancements

- Expand and validate taxonomy coverage against representative resumes and job descriptions.
- Improve context-aware disambiguation and evaluate extraction precision and recall.
- Provide career-path suggestions with transparent evidence and user controls.
- Add consent-based persistence, retention controls, and database migrations.
- Expand automated tests for matching quality, accessibility, and UI workflows.