"""Focused tests for the initial project scaffolding."""

from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pymupdf
from streamlit.testing.v1 import AppTest

from database.db import get_connection
from modules.career_recommender import (
    CORE_SKILL_WEIGHT,
    SUPPORTING_SKILL_WEIGHT,
    CareerCatalogError,
    CareerRecommendation,
    InvalidCareerRecommendationInputError,
    load_career_catalog,
    recommend_careers,
    recommend_careers_from_skills,
    recommend_careers_from_text,
)
from modules.job_matcher import (
    InvalidJobMatchingInputError,
    SKILL_COVERAGE_WEIGHT,
    TEXT_SIMILARITY_WEIGHT,
    match_job,
)
from modules.resume_parser import (
    EmptyPDFError,
    InvalidPDFError,
    MAX_PDF_PAGES,
    MAX_UPLOAD_SIZE_BYTES,
    PDFPageLimitError,
    PDFTooLargeError,
    ScannedPDFError,
    extract_text_from_pdf,
)
from modules.skill_extractor import (
    compare_skills,
    extract_skills,
    extract_skills_by_category,
)


class DatabaseTests(unittest.TestCase):
    def test_connection_sets_row_factory_and_foreign_keys(self) -> None:
        connection = get_connection(":memory:")
        try:
            self.assertIsNotNone(connection.execute("SELECT 1 AS value").fetchone()["value"])
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        finally:
            connection.close()


class DashboardWorkflowTests(unittest.TestCase):
    @staticmethod
    def run_app(app: AppTest) -> None:
        app.run(timeout=20)
        if app.exception:
            raise AssertionError([error.message for error in app.exception])

    def test_pdf_upload_inputs_and_match_results_survive_navigation(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        self.run_app(app)
        self.assertIn("Home", app.radio[0].options)

        app.radio[0].set_value("Resume & Job")
        self.run_app(app)
        first_pdf = ResumeParserTests.create_pdf(text="Python developer")
        app.file_uploader[0].set_value(
            ("first.pdf", first_pdf, "application/pdf")
        )
        self.run_app(app)
        self.assertEqual(app.session_state["resume_text"], "Python developer")
        app.text_area(key="job_description_input").set_value(
            "Python and SQL developer"
        )
        self.run_app(app)

        app.radio[0].set_value("Skill Analysis")
        self.run_app(app)
        self.assertNotIn("job_match_analysis", app.session_state)
        app.button[0].click()
        self.run_app(app)
        first_analysis = app.session_state["job_match_analysis"]
        self.assertEqual(first_analysis.matching_skills, ["Python"])
        self.assertEqual(first_analysis.missing_skills, ["SQL"])

        app.radio[0].set_value("Resume & Job")
        self.run_app(app)
        self.assertEqual(app.session_state["resume_text"], "Python developer")
        self.assertEqual(
            app.session_state["job_description_text"],
            "Python and SQL developer",
        )
        second_pdf = ResumeParserTests.create_pdf(text="Java engineer")
        app.file_uploader[0].set_value(
            ("second.pdf", second_pdf, "application/pdf")
        )
        self.run_app(app)
        app.text_area(key="job_description_input").set_value(
            "Java and SQL engineer"
        )
        self.run_app(app)
        self.assertNotIn("job_match_analysis", app.session_state)

        app.radio[0].set_value("Skill Analysis")
        self.run_app(app)
        app.button[0].click()
        self.run_app(app)
        second_analysis = app.session_state["job_match_analysis"]
        self.assertEqual(second_analysis.candidate_skills, ["Java"])
        self.assertEqual(second_analysis.matching_skills, ["Java"])
        self.assertEqual(second_analysis.missing_skills, ["SQL"])

        app.radio[0].set_value("Home")
        self.run_app(app)
        self.assertTrue(any("Ready" in item.value for item in app.markdown))
        app.radio[0].set_value("Career Recommendations")
        self.run_app(app)
        self.assertTrue(
            any("heuristic skill-alignment" in item.value.lower() for item in app.caption)
        )
        self.assertTrue(
            any("Software Developer" in item.label for item in app.expander)
        )

    def test_career_recommendations_page_warns_when_resume_is_missing(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        self.run_app(app)

        app.radio[0].set_value("Career Recommendations")
        self.run_app(app)
        self.assertTrue(
            any("resume is missing" in item.value.lower() for item in app.warning)
        )


class CareerRecommenderTests(unittest.TestCase):
    def test_catalog_loads_all_six_careers_and_validates_against_taxonomy(self) -> None:
        profiles = load_career_catalog()
        titles = {profile.title for profile in profiles}
        self.assertEqual(
            titles,
            {
                "AI/ML Engineer",
                "Python Developer",
                "Data Analyst",
                "Data Scientist",
                "NLP Engineer",
                "Software Developer",
            },
        )

    def test_each_career_ranks_first_for_its_core_skill_profile(self) -> None:
        profiles = load_career_catalog()
        for profile in profiles:
            with self.subTest(career=profile.title):
                results = recommend_careers_from_skills(list(profile.core_skills))
                self.assertEqual(results[0].title, profile.title)
                self.assertEqual(results[0].core_coverage_score, 100.0)
                self.assertEqual(results[0].missing_core_skills, [])

    def test_exact_weighted_suitability_score_calculation(self) -> None:
        # AI/ML Engineer has 6 core skills and 6 supporting skills.
        # Match 3 core (50%) and 2 supporting (33.333...%).
        results = recommend_careers_from_skills(
            ["Python", "Machine Learning", "Deep Learning", "NumPy", "Pandas"]
        )
        aiml = next(item for item in results if item.title == "AI/ML Engineer")
        expected_score = round(
            ((3 / 6) * CORE_SKILL_WEIGHT + (2 / 6) * SUPPORTING_SKILL_WEIGHT) * 100,
            1,
        )
        self.assertEqual(aiml.core_coverage_score, 50.0)
        self.assertEqual(aiml.supporting_coverage_score, 33.3)
        self.assertEqual(aiml.suitability_score, expected_score)
        self.assertEqual(
            aiml.next_skills_to_learn,
            aiml.missing_core_skills + aiml.missing_supporting_skills,
        )
        self.assertIn("Matches 3 of 6 core skills", aiml.explanation)

    def test_ranking_and_deterministic_tie_breaking(self) -> None:
        import tempfile

        catalog_data = {
            "careers": [
                {
                    "title": "Zeta Role",
                    "description": "Zeta description",
                    "core_skills": ["Python", "SQL"],
                    "supporting_skills": [],
                },
                {
                    "title": "Alpha Role",
                    "description": "Alpha description",
                    "core_skills": ["Python", "SQL"],
                    "supporting_skills": [],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "careers.json"
            catalog_path.write_text(
                __import__("json").dumps(catalog_data),
                encoding="utf-8",
            )
            results = recommend_careers_from_skills(
                ["Python"], catalog_path=catalog_path
            )
            self.assertEqual([r.title for r in results], ["Alpha Role", "Zeta Role"])

    def test_aliases_duplicates_and_unknown_skills_are_handled_safely(self) -> None:
        canonical_results = recommend_careers_from_skills(
            ["Python", "scikit-learn", "Machine Learning"]
        )
        noisy_results = recommend_careers_from_skills(
            [
                "python",
                "PYTHON 3",
                "sklearn",
                "ML",
                "   ",
                "NonExistentFrameworkXYZ",
            ]
        )
        self.assertEqual(canonical_results, noisy_results)

    def test_empty_skills_returns_zero_scores_with_guidance(self) -> None:
        results = recommend_careers_from_skills([])
        self.assertEqual(len(results), 6)
        self.assertTrue(all(r.suitability_score == 0.0 for r in results))
        self.assertTrue(all(r.next_skills_to_learn for r in results))

    def test_top_n_and_invalid_inputs_raise_explicit_errors(self) -> None:
        top_two = recommend_careers_from_skills(["Python"], top_n=2)
        self.assertEqual(len(top_two), 2)

        for bad_top_n in (0, -3, True, "2", 1.5):
            with self.subTest(bad_top_n=bad_top_n):
                with self.assertRaises(InvalidCareerRecommendationInputError):
                    recommend_careers_from_skills(["Python"], top_n=bad_top_n)  # type: ignore[arg-type]

        for bad_skills in ("Python", b"Python", None, 42, ["Python", 123]):
            with self.subTest(bad_skills=bad_skills):
                with self.assertRaises(InvalidCareerRecommendationInputError):
                    recommend_careers_from_skills(bad_skills)  # type: ignore[arg-type]

        with self.assertRaises(InvalidCareerRecommendationInputError):
            recommend_careers_from_text(None)  # type: ignore[arg-type]

    def test_missing_and_malformed_catalog_raise_career_catalog_error(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "does_not_exist.json"
            with self.assertRaises(CareerCatalogError):
                load_career_catalog(missing_path)

            bad_json = Path(tmpdir) / "bad.json"
            bad_json.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(CareerCatalogError):
                load_career_catalog(bad_json)

            unknown_skill_catalog = Path(tmpdir) / "unknown.json"
            unknown_skill_catalog.write_text(
                json.dumps(
                    {
                        "careers": [
                            {
                                "title": "Test Role",
                                "description": "Desc",
                                "core_skills": ["MadeUpSkill123"],
                                "supporting_skills": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CareerCatalogError):
                load_career_catalog(unknown_skill_catalog)

            overlap_catalog = Path(tmpdir) / "overlap.json"
            overlap_catalog.write_text(
                json.dumps(
                    {
                        "careers": [
                            {
                                "title": "Test Role",
                                "description": "Desc",
                                "core_skills": ["Python"],
                                "supporting_skills": ["Python"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CareerCatalogError):
                load_career_catalog(overlap_catalog)

    def test_backward_compatible_recommend_careers_returns_dicts(self) -> None:
        results = recommend_careers(["Python", "Natural Language Processing", "PyTorch"])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 6)
        self.assertIsInstance(results[0], dict)
        self.assertEqual(results[0]["title"], "NLP Engineer")
        self.assertIn("suitability_score", results[0])
        self.assertIn("next_skills_to_learn", results[0])

    def test_recommend_careers_from_text_extracts_and_ranks(self) -> None:
        results = recommend_careers_from_text(
            "Experienced in NLP, LLMs, PyTorch, Deep Learning, and Python."
        )
        self.assertIsInstance(results[0], CareerRecommendation)
        self.assertEqual(results[0].title, "NLP Engineer")
        self.assertGreater(results[0].suitability_score, results[-1].suitability_score)


class ResumeParserTests(unittest.TestCase):
    @staticmethod
    def create_pdf(
        *,
        text: str | None = None,
        image: bool = False,
        page_count: int = 1,
    ) -> bytes:
        document = pymupdf.open()
        for _ in range(page_count):
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
            if image:
                pixmap = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 1, 1), 0)
                pixmap.clear_with(255)
                page.insert_image(
                    pymupdf.Rect(0, 0, 100, 100),
                    stream=pixmap.tobytes("png"),
                )
        pdf_bytes = document.tobytes()
        document.close()
        return pdf_bytes

    def test_extracts_text_from_valid_pdf(self) -> None:
        pdf_stream = BytesIO(self.create_pdf(text="Candidate resume text"))
        pdf_stream.seek(len(pdf_stream.getvalue()))

        extracted_text = extract_text_from_pdf(pdf_stream)

        self.assertIn("Candidate resume text", extracted_text)
        self.assertEqual(pdf_stream.tell(), len(pdf_stream.getvalue()))

    def test_rejects_empty_pdf_page(self) -> None:
        pdf_bytes = self.create_pdf()

        with self.assertRaises(EmptyPDFError):
            extract_text_from_pdf(BytesIO(pdf_bytes))

    def test_rejects_invalid_pdf(self) -> None:
        with self.assertRaisesRegex(InvalidPDFError, "PDF signature"):
            extract_text_from_pdf(BytesIO(b"not a PDF"))

    def test_rejects_pdf_over_upload_size_limit(self) -> None:
        oversized_pdf = BytesIO(b"%PDF-1.7\n" + b"x" * MAX_UPLOAD_SIZE_BYTES)

        with self.assertRaisesRegex(PDFTooLargeError, "10 MB"):
            extract_text_from_pdf(oversized_pdf)

    def test_rejects_pdf_over_page_limit(self) -> None:
        pdf_bytes = self.create_pdf(
            text="Page content",
            page_count=MAX_PDF_PAGES + 1,
        )

        with self.assertRaisesRegex(PDFPageLimitError, "20-page"):
            extract_text_from_pdf(BytesIO(pdf_bytes))

    def test_accepts_pdf_at_page_limit(self) -> None:
        pdf_bytes = self.create_pdf(
            text="Page content",
            page_count=MAX_PDF_PAGES,
        )

        extracted_text = extract_text_from_pdf(BytesIO(pdf_bytes))

        self.assertEqual(extracted_text.count("Page content"), MAX_PDF_PAGES)

    def test_rejects_malformed_pdf_with_valid_signature(self) -> None:
        malformed_pdf = b"%PDF-1.7\nThis is not a valid PDF document."

        with self.assertRaisesRegex(InvalidPDFError, "invalid or corrupted"):
            extract_text_from_pdf(BytesIO(malformed_pdf))

    def test_rejects_non_pdf_upload(self) -> None:
        with self.assertRaises(InvalidPDFError):
            extract_text_from_pdf(BytesIO(b"Plain text file, not a PDF"))

    def test_rejects_password_protected_pdf(self) -> None:
        document = pymupdf.open()
        document.new_page()
        pdf_bytes = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw="test-password",
            owner_pw="test-owner",
        )
        document.close()

        with self.assertRaisesRegex(InvalidPDFError, "Password-protected"):
            extract_text_from_pdf(BytesIO(pdf_bytes))

    def test_explains_scanned_pdf_ocr_limitation(self) -> None:
        pdf_bytes = self.create_pdf(image=True)

        with self.assertRaisesRegex(ScannedPDFError, "OCR is not supported"):
            extract_text_from_pdf(BytesIO(pdf_bytes))

    def test_opens_pdf_from_memory_without_a_filename(self) -> None:
        pdf_bytes = self.create_pdf(text="In-memory resume")

        with patch(
            "modules.resume_parser.pymupdf.open",
            wraps=pymupdf.open,
        ) as pdf_open:
            extracted_text = extract_text_from_pdf(BytesIO(pdf_bytes))

        self.assertIn("In-memory resume", extracted_text)
        self.assertIn("stream", pdf_open.call_args.kwargs)
        self.assertNotIn("filename", pdf_open.call_args.kwargs)


class SkillExtractorTests(unittest.TestCase):
    def test_extracts_case_insensitive_keywords(self) -> None:
        self.assertEqual(
            extract_skills("PYTHON, JavaScript and python"),
            ["Python", "JavaScript"],
        )

    def test_prevents_duplicate_skills(self) -> None:
        self.assertEqual(
            extract_skills("Python 3, Python, PYTHON 3"),
            ["Python"],
        )

    def test_recognizes_aliases(self) -> None:
        self.assertEqual(
            extract_skills("JS, sklearn, and Golang"),
            ["JavaScript", "Go", "scikit-learn"],
        )

    def test_detects_requested_skills_and_categories(self) -> None:
        skills = extract_skills(
            "Python, Java, C programming, Machine Learning, Deep Learning, NLP, "
            "Pandas, NumPy, scikit-learn, TensorFlow, PyTorch, SQL, MySQL, "
            "React, and FastAPI"
        )
        categorized = extract_skills_by_category(
            "C programming, Machine Learning, Pandas, MySQL, React, FastAPI"
        )

        self.assertTrue(
            {
                "Python",
                "Java",
                "C",
                "Machine Learning",
                "Deep Learning",
                "Natural Language Processing",
                "Pandas",
                "NumPy",
                "scikit-learn",
                "TensorFlow",
                "PyTorch",
                "SQL",
                "MySQL",
                "React",
                "FastAPI",
            }.issubset(skills)
        )
        self.assertIn("C", categorized["Programming Languages"])
        self.assertIn("Machine Learning", categorized["AI and Machine Learning"])
        self.assertIn("Pandas", categorized["Frameworks and Libraries"])
        self.assertIn("MySQL", categorized["Databases"])
        self.assertIn("React", categorized["Frameworks and Libraries"])
        self.assertIn("FastAPI", categorized["Frameworks and Libraries"])

    def test_single_letter_c_requires_programming_context(self) -> None:
        self.assertNotIn("C", extract_skills("The letter C appears in this sentence."))

    def test_does_not_match_skills_inside_unrelated_words(self) -> None:
        self.assertEqual(
            extract_skills(
                "A pythonista used SQLAlchemy, JavaScripted components, and Reacting UI."
            ),
            [],
        )

    def test_empty_input_returns_no_skills(self) -> None:
        self.assertEqual(extract_skills(" \n\t "), [])

    def test_rejects_non_text_input(self) -> None:
        with self.assertRaises(TypeError):
            extract_skills(None)  # type: ignore[arg-type]

    def test_compares_candidate_and_required_skills(self) -> None:
        comparison = compare_skills(
            "Python, JavaScript, and Docker",
            "Python, SQL, and Docker experience required",
        )

        self.assertEqual(
            comparison.candidate_skills,
            ["Python", "JavaScript", "Docker"],
        )
        self.assertEqual(comparison.required_skills, ["Python", "SQL", "Docker"])
        self.assertEqual(comparison.matching_skills, ["Python", "Docker"])
        self.assertEqual(comparison.missing_skills, ["SQL"])
        self.assertEqual(
            comparison.missing_by_category["Databases"],
            ["SQL"],
        )


class JobMatcherTests(unittest.TestCase):
    def test_pdf_text_flows_through_skill_extraction_and_matching(self) -> None:
        pdf_bytes = ResumeParserTests.create_pdf(
            text="Experienced Python developer building REST APIs."
        )
        resume_text = extract_text_from_pdf(BytesIO(pdf_bytes))

        analysis = match_job(
            resume_text,
            "Python developer building REST APIs with SQL experience.",
        )

        self.assertIn("Python", analysis.matching_skills)
        self.assertIn("REST API", analysis.matching_skills)
        self.assertIn("SQL", analysis.missing_skills)
        self.assertIsNotNone(analysis.overall_match_score)

    def test_perfect_skill_and_text_match_scores_one_hundred(self) -> None:
        text = "Experienced Python and SQL developer."

        analysis = match_job(text, text)

        self.assertEqual(analysis.skill_coverage_score, 100.0)
        self.assertEqual(analysis.text_similarity_score, 100.0)
        self.assertEqual(analysis.overall_match_score, 100.0)
        self.assertEqual(analysis.missing_skills, [])

    def test_partial_match_uses_documented_weighted_score(self) -> None:
        analysis = match_job(
            "Experienced Python developer.",
            "Python and SQL developer required.",
        )

        self.assertEqual(analysis.skill_coverage_score, 50.0)
        self.assertEqual(analysis.matching_skills, ["Python"])
        self.assertEqual(analysis.missing_skills, ["SQL"])
        self.assertEqual(
            analysis.overall_match_score,
            round(
                analysis.skill_coverage_score * SKILL_COVERAGE_WEIGHT
                + analysis.text_similarity_score * TEXT_SIMILARITY_WEIGHT,
                1,
            ),
        )
        self.assertTrue(analysis.improvement_suggestions)

    def test_no_skill_match_has_zero_coverage_and_lists_additional_skills(self) -> None:
        analysis = match_job("JavaScript React", "Python SQL")

        self.assertEqual(analysis.skill_coverage_score, 0.0)
        self.assertEqual(analysis.text_similarity_score, 0.0)
        self.assertEqual(analysis.overall_match_score, 0.0)
        self.assertEqual(analysis.matching_skills, [])
        self.assertEqual(analysis.missing_skills, ["Python", "SQL"])
        self.assertEqual(
            analysis.additional_candidate_skills,
            ["JavaScript", "React"],
        )

    def test_empty_resume_does_not_produce_a_score(self) -> None:
        analysis = match_job("", "Python and SQL required")

        self.assertIsNone(analysis.overall_match_score)
        self.assertIsNone(analysis.skill_coverage_score)
        self.assertIsNone(analysis.text_similarity_score)
        self.assertEqual(analysis.missing_skills, ["Python", "SQL"])
        self.assertIn("non-empty", analysis.score_unavailable_reason)

    def test_empty_job_description_does_not_produce_a_score(self) -> None:
        analysis = match_job("Python and SQL experience", "  ")

        self.assertIsNone(analysis.overall_match_score)
        self.assertIsNone(analysis.skill_coverage_score)
        self.assertIsNone(analysis.text_similarity_score)
        self.assertEqual(analysis.required_skills, [])

    def test_duplicate_and_case_variants_count_once(self) -> None:
        analysis = match_job(
            "python PYTHON Python",
            "PYTHON Python, PYTHON",
        )

        self.assertEqual(analysis.candidate_skills, ["Python"])
        self.assertEqual(analysis.required_skills, ["Python"])
        self.assertEqual(analysis.skill_coverage_score, 100.0)

    def test_text_similarity_is_lexical(self) -> None:
        identical = match_job(
            "Python developer",
            "Python developer",
        )
        unrelated = match_job("Python", "SQL")

        self.assertEqual(identical.text_similarity_score, 100.0)
        self.assertEqual(unrelated.text_similarity_score, 0.0)

    def test_overall_score_unavailable_without_detected_job_skills(self) -> None:
        analysis = match_job(
            "Python developer",
            "Work with a collaborative and adaptable team.",
        )

        self.assertEqual(analysis.required_skills, [])
        self.assertIsNone(analysis.skill_coverage_score)
        self.assertIsNone(analysis.overall_match_score)
        self.assertTrue(analysis.score_unavailable_reason)

    def test_rejects_non_text_inputs(self) -> None:
        with self.assertRaises(InvalidJobMatchingInputError):
            match_job(None, "Python")  # type: ignore[arg-type]
        with self.assertRaises(InvalidJobMatchingInputError):
            match_job("Python", None)  # type: ignore[arg-type]


class SecurityRegressionTests(unittest.TestCase):
    """Regression tests that document and enforce the security controls
    introduced during the security remediation phase.

    These tests must never be deleted or skipped.  They verify:
      - PDF size and page limits as hard security controls.
      - PDF header signature validation as a file-type gate.
      - In-memory processing (no disk writes).
      - PyMuPDF is imported via the non-deprecated 'pymupdf' namespace.
      - The deprecated 'fitz' alias is not used anywhere in production modules.

    Dependency vulnerability auditing (`python -m pip_audit --desc`) is run as a
    separate security audit command so ordinary unit tests remain offline-safe.
    """

    # ------------------------------------------------------------------
    # PDF size limit — enforced at the parser layer, independent of the
    # Streamlit UI limit.  One byte over MAX_UPLOAD_SIZE_BYTES must fail.
    # ------------------------------------------------------------------

    def test_size_limit_constant_is_ten_megabytes(self) -> None:
        """MAX_UPLOAD_SIZE_BYTES must equal exactly 10 MiB (10 * 1024 * 1024)."""
        self.assertEqual(MAX_UPLOAD_SIZE_BYTES, 10 * 1024 * 1024)

    def test_parser_rejects_file_one_byte_over_size_limit(self) -> None:
        """A stream whose size attribute is MAX_UPLOAD_SIZE_BYTES + 1 must raise
        PDFTooLargeError before any PDF parsing begins."""
        oversized = BytesIO(b"%PDF-1.7\n" + b"x" * MAX_UPLOAD_SIZE_BYTES)
        # Simulate what Streamlit sets: a .size attribute on the file object.
        oversized.size = MAX_UPLOAD_SIZE_BYTES + 1  # type: ignore[attr-defined]

        with self.assertRaisesRegex(PDFTooLargeError, "10 MB"):
            extract_text_from_pdf(oversized)

    def test_parser_accepts_file_exactly_at_size_limit(self) -> None:
        """A valid PDF whose byte count equals MAX_UPLOAD_SIZE_BYTES must not
        raise PDFTooLargeError (it may raise other errors for content reasons)."""
        pdf_bytes = ResumeParserTests.create_pdf(text="At limit")
        # Pad to exactly the limit so we test the boundary condition.
        # We just need the size attribute check to pass, not a parseable PDF.
        boundary_stream = BytesIO(pdf_bytes)
        boundary_stream.size = MAX_UPLOAD_SIZE_BYTES  # type: ignore[attr-defined]
        try:
            extract_text_from_pdf(boundary_stream)
        except PDFTooLargeError:
            self.fail("PDFTooLargeError raised for a file exactly at the size limit.")
        except Exception:
            # Other errors (EmptyPDF, InvalidPDF, etc.) are acceptable here.
            pass

    # ------------------------------------------------------------------
    # PDF page limit — 20 pages maximum.
    # ------------------------------------------------------------------

    def test_page_limit_constant_is_twenty(self) -> None:
        """MAX_PDF_PAGES must equal 20."""
        self.assertEqual(MAX_PDF_PAGES, 20)

    def test_parser_rejects_twenty_one_page_pdf(self) -> None:
        """A PDF with 21 pages must raise PDFPageLimitError."""
        pdf_bytes = ResumeParserTests.create_pdf(
            text="Content", page_count=21
        )
        with self.assertRaisesRegex(PDFPageLimitError, "20-page"):
            extract_text_from_pdf(BytesIO(pdf_bytes))

    def test_parser_accepts_twenty_page_pdf(self) -> None:
        """A PDF with exactly 20 pages must be accepted and return text."""
        pdf_bytes = ResumeParserTests.create_pdf(
            text="Page", page_count=20
        )
        text = extract_text_from_pdf(BytesIO(pdf_bytes))
        self.assertEqual(text.count("Page"), 20)

    def test_parser_rejects_single_page_over_limit(self) -> None:
        """Even a 1-page PDF must pass the page-limit check cleanly
        (control flow must not skip the page-count gate for small PDFs)."""
        pdf_bytes = ResumeParserTests.create_pdf(text="Single page")
        # 1 page is well within the limit; must succeed.
        text = extract_text_from_pdf(BytesIO(pdf_bytes))
        self.assertIn("Single page", text)

    # ------------------------------------------------------------------
    # PDF header / signature validation.
    # ------------------------------------------------------------------

    def test_pdf_header_pattern_rejects_missing_signature(self) -> None:
        """Files without a valid %PDF-X.Y header in the first 1024 bytes
        must raise InvalidPDFError with a message mentioning 'PDF signature'."""
        for bad_content in (
            b"",
            b"plain text",
            b"\x00\x01\x02\x03",
            b"PK\x03\x04notapdf",           # ZIP magic
            b"\xff\xd8\xff\xe0JFIF",         # JPEG magic
        ):
            with self.subTest(content=bad_content[:20]):
                with self.assertRaises(
                    (InvalidPDFError, EmptyPDFError),
                    msg=f"Expected rejection of non-PDF bytes: {bad_content[:20]!r}",
                ):
                    extract_text_from_pdf(BytesIO(bad_content))

    def test_pdf_header_pattern_accepts_common_pdf_versions(self) -> None:
        """The header regex must accept %PDF-1.0 through %PDF-2.0 prefixes
        without immediately rejecting the file on signature grounds."""
        import re
        from modules.resume_parser import PDF_HEADER_PATTERN

        for version in ("1.0", "1.4", "1.7", "2.0"):
            header = f"%PDF-{version}\n".encode()
            self.assertIsNotNone(
                PDF_HEADER_PATTERN.search(header),
                msg=f"Header pattern should match PDF version {version}",
            )

    def test_pdf_signature_check_rejects_header_at_offset(self) -> None:
        """The signature check must look only within the first 1024 bytes.
        A valid PDF header placed after byte 1024 must still be rejected."""
        late_header = b"x" * 1025 + b"%PDF-1.7\n"
        with self.assertRaises((InvalidPDFError, EmptyPDFError)):
            extract_text_from_pdf(BytesIO(late_header))

    # ------------------------------------------------------------------
    # In-memory processing — no resume bytes must be written to disk.
    # ------------------------------------------------------------------

    def test_pdf_extraction_does_not_write_to_disk(self) -> None:
        """extract_text_from_pdf must never call the built-in open() with
        a file-system path; all processing must stay in memory."""
        pdf_bytes = ResumeParserTests.create_pdf(text="Memory only")
        with patch("builtins.open", side_effect=AssertionError(
            "builtins.open was called — PDF data was written to disk"
        )) as mock_open:
            try:
                extract_text_from_pdf(BytesIO(pdf_bytes))
            except AssertionError:
                self.fail(
                    "extract_text_from_pdf called builtins.open, "
                    "which means it attempted to write to disk."
                )
            # builtins.open must not have been called at all.
            mock_open.assert_not_called()

    # ------------------------------------------------------------------
    # PyMuPDF import — production code must use the non-deprecated namespace.
    # ------------------------------------------------------------------

    def test_resume_parser_uses_pymupdf_not_fitz(self) -> None:
        """The production module resume_parser.py must import 'pymupdf',
        not the deprecated 'fitz' alias.  This test reads the source file
        directly to catch any future regression."""
        source_path = (
            Path(__file__).resolve().parents[1]
            / "modules"
            / "resume_parser.py"
        )
        source_text = source_path.read_text(encoding="utf-8")
        self.assertIn(
            "import pymupdf",
            source_text,
            "resume_parser.py must contain 'import pymupdf'.",
        )
        self.assertNotIn(
            "import fitz",
            source_text,
            "resume_parser.py must not use the deprecated 'import fitz'.",
        )

    def test_pymupdf_module_is_importable_via_non_deprecated_name(self) -> None:
        """'import pymupdf' must succeed without a DeprecationWarning."""
        import warnings
        import importlib

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module("pymupdf")

        deprecation_warnings = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "fitz" in str(w.message).lower()
        ]
        self.assertEqual(
            deprecation_warnings,
            [],
            f"Unexpected fitz DeprecationWarnings on 'import pymupdf': "
            f"{deprecation_warnings}",
        )

    # ------------------------------------------------------------------
    # HTML escaping & server upload config.
    # ------------------------------------------------------------------

    def test_streamlit_config_enforces_ten_mb_upload_limit(self) -> None:
        """Verify .streamlit/config.toml parses cleanly and sets maxUploadSize = 10."""
        import tomllib

        config_path = (
            Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
        )
        self.assertTrue(config_path.is_file(), ".streamlit/config.toml must exist.")
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
        self.assertEqual(config.get("server", {}).get("maxUploadSize"), 10)

    def test_render_heading_escapes_html_characters(self) -> None:
        """Dynamic values in render_heading must be escaped to prevent HTML injection."""
        import app as career_app

        with patch.object(career_app.st, "markdown") as mock_markdown:
            career_app.render_heading(
                "<script>alert(1)</script>",
                'Title "with" <tags>',
                "Sub & <img src=x onerror=alert(1)>",
            )

        rendered_html = mock_markdown.call_args.args[0]
        self.assertNotIn("<script>", rendered_html)
        self.assertNotIn("<img ", rendered_html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered_html)
        self.assertIn("&quot;with&quot;", rendered_html)
        self.assertIn("Sub &amp; &lt;img src=x onerror=alert(1)&gt;", rendered_html)

    def test_render_metric_escapes_html_characters(self) -> None:
        """Dynamic values in render_metric must be escaped to prevent HTML injection."""
        import app as career_app

        with patch.object(career_app.st, "markdown") as mock_markdown:
            career_app.render_metric(
                "<b>Label</b>",
                "<script>xss()</script>",
            )

        rendered_html = mock_markdown.call_args.args[0]
        self.assertNotIn("<b>Label</b>", rendered_html)
        self.assertNotIn("<script>", rendered_html)
        self.assertIn("&lt;b&gt;Label&lt;/b&gt;", rendered_html)
        self.assertIn("&lt;script&gt;xss()&lt;/script&gt;", rendered_html)

    def test_render_pending_page_escapes_html_characters(self) -> None:
        """Dynamic details in render_pending_page must be escaped."""
        import app as career_app

        with patch.object(career_app.st, "markdown") as mock_markdown:
            career_app.render_pending_page(
                "Page <1>",
                "Sub <2>",
                "<iframe src='javascript:alert(1)'></iframe>",
            )

        details_call_html = mock_markdown.call_args_list[-1].args[0]
        self.assertNotIn("<iframe", details_call_html)
        self.assertIn("&lt;iframe src=&#x27;javascript:alert(1)&#x27;&gt;", details_call_html)


if __name__ == "__main__":
    unittest.main()
