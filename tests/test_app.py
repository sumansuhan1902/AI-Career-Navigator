"""Focused tests for the initial project scaffolding."""

from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pymupdf as fitz
from streamlit.testing.v1 import AppTest

from database.db import get_connection
from modules.career_recommender import recommend_careers
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
            any("not implemented" in item.value.lower() for item in app.markdown)
        )


class PendingFeatureTests(unittest.TestCase):
    def test_analysis_modules_fail_explicitly_until_implemented(self) -> None:
        pending_calls = (
            lambda: recommend_careers(["Python"]),
        )
        for call in pending_calls:
            with self.subTest(call=call):
                with self.assertRaises(NotImplementedError):
                    call()


class ResumeParserTests(unittest.TestCase):
    @staticmethod
    def create_pdf(
        *,
        text: str | None = None,
        image: bool = False,
        page_count: int = 1,
    ) -> bytes:
        document = fitz.open()
        for _ in range(page_count):
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
            if image:
                pixmap = fitz.Pixmap(fitz.csRGB, (0, 0, 1, 1), 0)
                pixmap.clear_with(255)
                page.insert_image(
                    fitz.Rect(0, 0, 100, 100),
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
        document = fitz.open()
        document.new_page()
        pdf_bytes = document.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
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
            wraps=fitz.open,
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


if __name__ == "__main__":
    unittest.main()
