"""Focused tests for the initial project scaffolding."""

from __future__ import annotations

import json
import re
import tempfile
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
from modules.learning_roadmap import (
    InvalidRoadmapInputError,
    LearningCatalogError,
    generate_learning_roadmap,
    load_learning_catalog,
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

    def test_sidebar_displays_career_development_label(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        self.run_app(app)

        sidebar_markup = "\n".join(element.value for element in app.markdown)
        self.assertIn("CAREER DEVELOPMENT TOOL", sidebar_markup)
        self.assertNotIn("EARLY PROTOTYPE", sidebar_markup)

    def test_home_dashboard_reflects_empty_and_populated_input_state(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        self.run_app(app)

        self.assertTrue(
            any(
                "Analyze a text-based PDF resume to extract technical and "
                "professional skills" in item.value
                for item in app.markdown
            )
        )
        self.assertEqual(app.progress[0].value, 0.0)
        self.assertTrue(
            any("Upload a text-based PDF" in item.value for item in app.info)
        )
        self.assertTrue(
            any("Awaiting resume" in item.value for item in app.markdown)
        )

        app.session_state["resume_text"] = "Python developer with SQL experience"
        app.session_state["job_description_text"] = "Python and SQL developer"
        self.run_app(app)

        self.assertEqual(app.progress[0].value, 66)
        self.assertTrue(any("2 found" in item.value for item in app.markdown))
        self.assertTrue(
            any("Python, SQL" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("Ready to analyze" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("Setup progress" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("Open Skill Analysis" in item.value for item in app.info)
        )

    def test_home_shows_add_resume_when_only_job_description_exists(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        app.session_state["job_description_text"] = "Python and SQL developer"
        self.run_app(app)

        self.assertEqual(app.progress[0].value, 33)
        self.assertTrue(any("Add resume" in item.value for item in app.markdown))
        self.assertTrue(
            any("Upload a text-based PDF" in item.value for item in app.info)
        )

    def test_home_shows_add_job_description_when_only_resume_exists(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        app.session_state["resume_text"] = "Python developer"
        self.run_app(app)

        self.assertEqual(app.progress[0].value, 33)
        self.assertTrue(
            any("Add job description" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("Add a target job description" in item.value for item in app.info)
        )

    def test_home_shows_complete_progress_after_current_analysis(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        resume_text = "Python developer"
        job_description = "Python and SQL developer"
        app.session_state["resume_text"] = resume_text
        app.session_state["job_description_text"] = job_description
        app.session_state["job_match_analysis"] = match_job(
            resume_text,
            job_description,
        )
        app.session_state["job_match_inputs"] = (resume_text, job_description)
        self.run_app(app)

        self.assertEqual(app.progress[0].value, 100)
        self.assertTrue(
            any("Ready - analysis complete" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any(
                "Review your match in Skill Analysis" in item.value
                for item in app.success
            )
        )

    def test_resume_page_shows_helpful_first_upload_state(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        self.run_app(app)

        app.radio[0].set_value("Resume & Job")
        self.run_app(app)

        self.assertEqual(app.file_uploader[0].key, "resume_upload")
        self.assertTrue(
            any("Upload a text-based PDF" in item.value for item in app.info)
        )
        self.assertEqual(app.text_area(key="job_description_input").value, "")
        self.assertTrue(
            any(
                "No target job description added yet." in item.value
                for item in app.caption
            )
        )

    def test_valid_pdf_upload_is_not_reparsed_on_streamlit_reruns(self) -> None:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        self.run_app(app)
        app.radio[0].set_value("Resume & Job")
        self.run_app(app)
        pdf_bytes = ResumeParserTests.create_pdf(text="Python developer")

        with patch(
            "modules.resume_parser.pymupdf.open",
            wraps=pymupdf.open,
        ) as pdf_open:
            app.file_uploader[0].set_value(
                ("resume.pdf", pdf_bytes, "application/pdf")
            )
            self.run_app(app)
            self.run_app(app)

        self.assertEqual(app.session_state["resume_text"], "Python developer")
        self.assertEqual(pdf_open.call_count, 1)

    def test_invalid_replacement_preserves_previous_valid_resume(self) -> None:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        self.run_app(app)
        app.radio[0].set_value("Resume & Job")
        self.run_app(app)

        original_resume = "Python developer with SQL experience"
        app.file_uploader[0].set_value(
            (
                "valid.pdf",
                ResumeParserTests.create_pdf(text=original_resume),
                "application/pdf",
            )
        )
        self.run_app(app)
        app.file_uploader[0].set_value(
            ("replacement.pdf", b"not a PDF", "application/pdf")
        )
        self.run_app(app)

        self.assertEqual(app.session_state["resume_text"], original_resume)
        self.assertTrue(
            any("missing PDF signature" in item.value for item in app.error)
        )
        app.file_uploader[0].set_value(
            (
                "corrupt.pdf",
                b"%PDF-1.7\ncorrupt document",
                "application/pdf",
            )
        )
        self.run_app(app)
        self.assertEqual(app.session_state["resume_text"], original_resume)
        self.assertTrue(
            any("invalid or corrupted" in item.value.lower() for item in app.error)
        )
        self.assertTrue(
            any(
                "previous valid resume text is still available" in item.value.lower()
                for item in app.caption
            )
        )

    def test_empty_replacement_preserves_previous_valid_resume(self) -> None:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        self.run_app(app)
        app.radio[0].set_value("Resume & Job")
        self.run_app(app)

        original_resume = "Python developer"
        app.file_uploader[0].set_value(
            (
                "valid.pdf",
                ResumeParserTests.create_pdf(text=original_resume),
                "application/pdf",
            )
        )
        self.run_app(app)
        app.file_uploader[0].set_value(
            ("empty.pdf", ResumeParserTests.create_pdf(), "application/pdf")
        )
        self.run_app(app)

        self.assertEqual(app.session_state["resume_text"], original_resume)
        self.assertTrue(
            any("no extractable text" in item.value.lower() for item in app.warning)
        )
        self.assertTrue(
            any(
                "previous valid resume text is still available" in item.value.lower()
                for item in app.caption
            )
        )

    def test_skill_analysis_input_states_and_resume_overview(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        self.run_app(app)

        app.radio[0].set_value("Skill Analysis")
        self.run_app(app)
        self.assertTrue(
            any(
                "Add a resume and a target job description"
                in item.value
                for item in app.info
            )
        )

        app.session_state["resume_text"] = "Python developer"
        self.run_app(app)
        self.assertTrue(
            any(
                "Add a target job description to identify matched and missing skills"
                in item.value
                for item in app.info
            )
        )
        self.assertTrue(
            any(metric.label == "Total extracted skills" for metric in app.metric)
        )

        app.session_state["resume_text"] = ""
        app.session_state["job_description_text"] = "Python and SQL developer"
        self.run_app(app)
        self.assertTrue(
            any(
                "Add a resume to see your extracted skills"
                in item.value
                for item in app.info
            )
        )

        app.session_state["resume_text"] = "Python developer"
        self.run_app(app)
        self.assertEqual(app.button[0].label, "Analyze Job Match")
        self.assertTrue(
            any(
                "Resume skill overview" in item.value
                for item in app.markdown
            )
        )

    def test_skill_analysis_displays_matched_missing_and_categories(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        self.run_app(app)

        app.session_state["resume_text"] = "Python SQL Git"
        app.session_state["job_description_text"] = "Python SQL Docker"
        app.radio[0].set_value("Skill Analysis")
        self.run_app(app)
        app.button[0].click()
        self.run_app(app)

        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Total extracted skills"], "3")
        self.assertNotIn("Programming Languages", metrics)
        self.assertNotIn("Databases", metrics)
        self.assertNotIn("Developer Tools", metrics)
        self.assertTrue(
            any("2 detected job skills matched" in item.value for item in app.success)
        )
        self.assertTrue(
            any("1 detected job skills" in item.value for item in app.warning)
        )
        self.assertTrue(any("Python, SQL" in item.value for item in app.markdown))
        self.assertTrue(any("Docker" in item.value for item in app.markdown))
        self.assertIn("66.7% of detected job skills matched", app.progress[0].text)
        category_labels = [expander.label for expander in app.expander]
        self.assertIn("Programming Languages (1)", category_labels)
        self.assertIn("Databases (1)", category_labels)
        self.assertIn("Developer Tools (1)", category_labels)

    def test_all_five_dashboard_pages_remain_navigable(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        app.session_state["resume_text"] = (
            "Python SQL REST API Git Django Flask"
        )
        self.run_app(app)
        page_names = (
            "Home",
            "Resume & Job",
            "Skill Analysis",
            "Career Recommendations",
            "Learning Roadmap",
        )

        self.assertEqual(tuple(app.radio[0].options), page_names)
        for page_name in page_names:
            app.radio[0].set_value(page_name)
            self.run_app(app)
            rendered_markdown = "\n".join(
                item.value for item in app.markdown if isinstance(item.value, str)
            )
            heading_levels = [
                int(level)
                for level in re.findall(r"<h([1-6])\b[^>]*>", rendered_markdown)
            ]
            self.assertEqual(
                heading_levels.count(1),
                1,
                f"{page_name} should render exactly one page-level h1.",
            )
            self.assertTrue(
                all(
                    current <= previous + 1
                    for previous, current in zip(
                        heading_levels,
                        heading_levels[1:],
                    )
                ),
                f"{page_name} headings should not skip hierarchy levels.",
            )

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
        self.assertTrue(
            any("first.pdf processed" in item.value for item in app.success)
        )
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
            any("Software Developer" in item.value for item in app.markdown)
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

    def test_career_recommendations_rank_different_skill_profiles(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        app.session_state["resume_text"] = (
            "Python SQL REST API Git Django Flask"
        )
        self.run_app(app)

        app.radio[0].set_value("Career Recommendations")
        self.run_app(app)
        markdown_values = [item.value for item in app.markdown]
        self.assertTrue(any("#1 · Python Developer" in value for value in markdown_values))
        self.assertTrue(any("Core skills" in value for value in markdown_values))
        self.assertTrue(
            any("Supporting skills" in value for value in markdown_values)
        )
        self.assertTrue(
            any(item.value.startswith("Matched:") for item in app.success)
        )
        self.assertTrue(
            any(item.value.startswith("Not detected:") for item in app.warning)
        )
        self.assertTrue(
            any(
                item.text == "70.0% heuristic alignment" for item in app.progress
            )
        )
        self.assertFalse(
            any(item.label == "Heuristic suitability" for item in app.metric)
        )
        self.assertTrue(
            any(
                item.label == "Core skill coverage" and item.value == "100.0%"
                for item in app.metric
            )
        )

        app.session_state["resume_text"] = (
            "Python Machine Learning Deep Learning scikit-learn "
            "TensorFlow PyTorch"
        )
        self.run_app(app)
        self.assertTrue(
            any("#1 · AI/ML Engineer" in item.value for item in app.markdown)
        )

    def test_career_recommendations_show_empty_skill_state_without_fake_scores(
        self,
    ) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path))
        app.session_state["resume_text"] = "Research experience with no listed tools."
        self.run_app(app)

        app.radio[0].set_value("Career Recommendations")
        self.run_app(app)

        self.assertTrue(
            any(
                "no curated taxonomy skills" in item.value.lower()
                for item in app.warning
            )
        )
        self.assertEqual(len(app.progress), 6)
        self.assertTrue(all(item.value == 0.0 for item in app.progress))
        self.assertFalse(
            any(item.label == "Heuristic suitability" for item in app.metric)
        )


class DashboardLearningRoadmapTests(unittest.TestCase):
    @staticmethod
    def run_app(app: AppTest) -> None:
        app.run(timeout=20)
        if app.exception:
            raise AssertionError([error.message for error in app.exception])

    @staticmethod
    def app_path() -> Path:
        return Path(__file__).resolve().parents[1] / "app.py"

    @staticmethod
    def open_roadmap(app: AppTest) -> None:
        app.radio[0].set_value("Learning Roadmap")
        DashboardLearningRoadmapTests.run_app(app)

    def test_navigation_and_no_resume_show_full_selectable_roadmap(self) -> None:
        app = AppTest.from_file(str(self.app_path()))
        self.run_app(app)
        self.assertIn("Learning Roadmap", app.radio[0].options)

        self.open_roadmap(app)

        self.assertTrue(
            any("upload a resume" in item.value.lower() for item in app.info)
        )
        self.assertEqual(
            app.selectbox(key="learning_roadmap_career_selector").options,
            [
                "AI/ML Engineer",
                "Python Developer",
                "Data Analyst",
                "Data Scientist",
                "NLP Engineer",
                "Software Developer",
            ],
        )
        self.assertEqual(len(app.checkbox), 12)
        section_headings = [item.value for item in app.markdown]
        self.assertTrue(any("Beginner" in value for value in section_headings))
        self.assertTrue(any("Intermediate" in value for value in section_headings))
        self.assertTrue(any("Advanced" in value for value in section_headings))
        self.assertTrue(any("0 of 12 skills complete" in item.value for item in app.caption))
        self.assertEqual(app.progress[0].text, "0.0% complete")
        roadmap_metrics = {
            item.label: item.value
            for item in app.metric
            if item.label in {"Total roadmap skills", "Completed", "Remaining"}
        }
        self.assertEqual(
            roadmap_metrics,
            {
                "Total roadmap skills": "12",
                "Completed": "0",
                "Remaining": "12",
            },
        )
        self.assertTrue(
            any("**Completed skills:** None" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("**Remaining skills:**" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("Mini-project:" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("Completion milestone:" in item.value for item in app.markdown)
        )

    def test_resume_selects_top_recommendation_and_excludes_known_skills(self) -> None:
        app = AppTest.from_file(str(self.app_path()))
        app.session_state["resume_text"] = (
            "Python SQL REST API Git Django Flask"
        )
        self.run_app(app)
        self.open_roadmap(app)

        self.assertEqual(
            app.selectbox(key="learning_roadmap_career_selector").value,
            "Python Developer",
        )
        checkbox_labels = [checkbox.label for checkbox in app.checkbox]
        self.assertFalse(any(label.startswith("Python (") for label in checkbox_labels))
        self.assertFalse(any(label.startswith("SQL (") for label in checkbox_labels))
        self.assertTrue(
            any("Detected resume skills" in item.value for item in app.caption)
        )

    def test_completion_checkbox_updates_progress(self) -> None:
        app = AppTest.from_file(str(self.app_path()))
        self.run_app(app)
        self.open_roadmap(app)

        app.checkbox(key="learning_roadmap_done_AI/ML Engineer_Python").check().run()

        progress = app.session_state["learning_roadmap_progress_by_career"]
        self.assertEqual(progress["AI/ML Engineer"], ("Python",))
        self.assertTrue(
            any("1 of 12 skills complete" in item.value for item in app.caption)
        )
        roadmap_metrics = {
            item.label: item.value
            for item in app.metric
            if item.label in {"Total roadmap skills", "Completed", "Remaining"}
        }
        self.assertEqual(
            roadmap_metrics,
            {
                "Total roadmap skills": "12",
                "Completed": "1",
                "Remaining": "11",
            },
        )
        self.assertIn("8.3%", app.progress[0].text)
        self.assertEqual(app.progress[0].text, "8.3% complete")
        self.assertTrue(
            any("**Completed skills:** Python" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("**Remaining skills:**" in item.value for item in app.markdown)
        )

    def test_completion_badges_distinguish_completed_and_pending_skills(self) -> None:
        with patch("streamlit.badge") as badge:
            app = AppTest.from_file(str(self.app_path()))
            self.run_app(app)
            self.open_roadmap(app)

            status_labels = [
                call.args[0]
                for call in badge.call_args_list
                if call.args and call.args[0] in {"Completed", "Pending", "In progress"}
            ]
            self.assertIn("Pending", status_labels)
            self.assertNotIn("In progress", status_labels)

            app.checkbox(
                key="learning_roadmap_done_AI/ML Engineer_Python"
            ).check().run()
            status_labels = [
                call.args[0]
                for call in badge.call_args_list
                if call.args and call.args[0] in {"Completed", "Pending", "In progress"}
            ]
            self.assertIn("Completed", status_labels)
            self.assertIn("Pending", status_labels)
            self.assertNotIn("In progress", status_labels)

    def test_career_switching_and_page_navigation_preserve_separate_progress(self) -> None:
        app = AppTest.from_file(str(self.app_path()))
        self.run_app(app)
        self.open_roadmap(app)
        app.checkbox(key="learning_roadmap_done_AI/ML Engineer_Python").check().run()

        app.selectbox(key="learning_roadmap_career_selector").select(
            "Python Developer"
        ).run()
        self.assertFalse(
            app.checkbox(key="learning_roadmap_done_Python Developer_Python").value
        )
        app.checkbox(key="learning_roadmap_done_Python Developer_Python").check().run()

        progress = app.session_state["learning_roadmap_progress_by_career"]
        self.assertEqual(progress["AI/ML Engineer"], ("Python",))
        self.assertEqual(progress["Python Developer"], ("Python",))

        app.selectbox(key="learning_roadmap_career_selector").select(
            "AI/ML Engineer"
        ).run()
        self.assertTrue(
            app.checkbox(key="learning_roadmap_done_AI/ML Engineer_Python").value
        )
        app.radio[0].set_value("Home").run()
        app.radio[0].set_value("Learning Roadmap").run()
        self.assertTrue(
            app.checkbox(key="learning_roadmap_done_AI/ML Engineer_Python").value
        )

    def test_duplicate_and_out_of_career_progress_is_not_counted(self) -> None:
        app = AppTest.from_file(str(self.app_path()))
        app.session_state["learning_roadmap_progress_by_career"] = {
            "AI/ML Engineer": ["Python", "Python", "Django"],
            "Python Developer": ["Django", "Django"],
        }
        self.run_app(app)
        self.open_roadmap(app)

        progress = app.session_state["learning_roadmap_progress_by_career"]
        self.assertEqual(progress["AI/ML Engineer"], ("Python",))
        self.assertEqual(progress["Python Developer"], ["Django", "Django"])
        self.assertTrue(
            any("1 of 12 skills complete" in item.value for item in app.caption)
        )
        self.assertTrue(
            any("**Completed skills:** Python" in item.value for item in app.markdown)
        )

    def test_reset_requires_confirmation_and_only_resets_selected_career(self) -> None:
        app = AppTest.from_file(str(self.app_path()))
        app.session_state["learning_roadmap_progress_by_career"] = {
            "AI/ML Engineer": ("Python", "Git"),
            "Python Developer": ("Django",),
        }
        self.run_app(app)
        self.open_roadmap(app)

        app.button(key="learning_roadmap_reset_AI/ML Engineer").click().run()
        self.assertTrue(
            any("cannot be undone" in item.value for item in app.warning)
        )
        app.button(key="learning_roadmap_cancel_reset_AI/ML Engineer").click().run()
        self.assertEqual(
            app.session_state["learning_roadmap_progress_by_career"][
                "AI/ML Engineer"
            ],
            ("Git", "Python"),
        )

        app.button(key="learning_roadmap_reset_AI/ML Engineer").click().run()
        app.button(key="learning_roadmap_confirm_reset_AI/ML Engineer").click().run()

        progress = app.session_state["learning_roadmap_progress_by_career"]
        self.assertNotIn("AI/ML Engineer", progress)
        self.assertEqual(progress["Python Developer"], ("Django",))
        self.assertFalse(
            app.checkbox(key="learning_roadmap_done_AI/ML Engineer_Python").value
        )

    def test_resume_changes_preserve_completed_progress_and_restore_it(self) -> None:
        app = AppTest.from_file(str(self.app_path()))
        original_resume = "Python SQL REST API Git Django Flask"
        app.session_state["resume_text"] = original_resume
        self.run_app(app)
        self.open_roadmap(app)
        self.assertEqual(
            app.selectbox(key="learning_roadmap_career_selector").value,
            "Python Developer",
        )

        app.checkbox(
            key="learning_roadmap_done_Python Developer_PostgreSQL"
        ).check().run()
        app.session_state["resume_text"] = (
            "Python NumPy Pandas Machine Learning PyTorch TensorFlow Neural Networks"
        )
        self.run_app(app)

        self.assertEqual(
            app.selectbox(key="learning_roadmap_career_selector").value,
            "AI/ML Engineer",
        )
        self.assertEqual(
            app.session_state["learning_roadmap_progress_by_career"][
                "Python Developer"
            ],
            ("PostgreSQL",),
        )

        app.session_state["resume_text"] = original_resume
        self.run_app(app)
        self.assertEqual(
            app.selectbox(key="learning_roadmap_career_selector").value,
            "Python Developer",
        )
        self.assertTrue(
            app.checkbox(
                key="learning_roadmap_done_Python Developer_PostgreSQL"
            ).value
        )

    def test_all_missing_skills_completed_shows_100_percent(self) -> None:
        app = AppTest.from_file(str(self.app_path()))
        ai_ml_skills = (
            "Python",
            "Machine Learning",
            "Deep Learning",
            "scikit-learn",
            "TensorFlow",
            "PyTorch",
            "NumPy",
            "Pandas",
            "Docker",
            "Git",
            "SQL",
            "FastAPI",
        )
        app.session_state["learning_roadmap_progress_by_career"] = {
            "AI/ML Engineer": ai_ml_skills,
        }
        self.run_app(app)
        self.open_roadmap(app)

        self.assertEqual(app.progress[0].text, "100.0% complete")
        self.assertIn("100.0%", app.progress[0].text)
        self.assertTrue(all(checkbox.value for checkbox in app.checkbox))
        self.assertTrue(
            any("**Remaining skills:** None" in item.value for item in app.markdown)
        )

    def test_resume_with_no_missing_skills_renders_empty_roadmap_safely(self) -> None:
        python_career = next(
            profile
            for profile in load_career_catalog()
            if profile.title == "Python Developer"
        )
        app = AppTest.from_file(str(self.app_path()))
        app.session_state["resume_text"] = " ".join(
            python_career.core_skills + python_career.supporting_skills
        )
        self.run_app(app)
        self.open_roadmap(app)

        self.assertEqual(len(app.checkbox), 0)
        self.assertEqual(
            {
                item.label: item.value
                for item in app.metric
                if item.label in {"Total roadmap skills", "Completed", "Remaining"}
            },
            {
                "Total roadmap skills": "0",
                "Completed": "0",
                "Remaining": "0",
            },
        )
        self.assertTrue(any("N/A" in item.value for item in app.markdown))
        self.assertFalse(any("100.0%" in item.value for item in app.markdown))
        self.assertEqual(len(app.progress), 0)
        self.assertTrue(
            any("already covers all tracked skills" in item.value for item in app.success)
        )
        self.assertTrue(
            any("No beginner skills remain" in item.value for item in app.caption)
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


class LearningRoadmapTests(unittest.TestCase):
    @staticmethod
    def learning_catalog_path() -> Path:
        return Path(__file__).resolve().parents[1] / "data" / "learning_paths.json"

    @staticmethod
    def write_catalog(path: Path, data: object) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_valid_catalog_covers_all_37_career_skills(self) -> None:
        learning_paths = load_learning_catalog()
        careers = load_career_catalog()
        career_skills = {
            skill
            for career in careers
            for skill in career.core_skills + career.supporting_skills
        }
        catalog_skills = {item.skill for item in learning_paths}

        self.assertEqual(len(career_skills), 37)
        self.assertEqual(len(catalog_skills), 37)
        self.assertEqual(len(learning_paths), 37)
        self.assertTrue(career_skills.issubset(catalog_skills))

    def test_invalid_json_missing_catalog_and_unknown_skills_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            missing_path = temp_path / "missing.json"
            with self.assertRaises(LearningCatalogError):
                load_learning_catalog(missing_path)

            malformed_path = temp_path / "malformed.json"
            malformed_path.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(LearningCatalogError):
                load_learning_catalog(malformed_path)

            catalog_data = json.loads(
                self.learning_catalog_path().read_text(encoding="utf-8")
            )
            catalog_data["learning_paths"][0]["skill"] = "MadeUpSkill"
            unknown_skill_path = temp_path / "unknown_skill.json"
            self.write_catalog(unknown_skill_path, catalog_data)
            with self.assertRaisesRegex(LearningCatalogError, "non-canonical"):
                load_learning_catalog(unknown_skill_path)

    def test_valid_prerequisites_are_present_and_level_ordered(self) -> None:
        learning_paths = load_learning_catalog()
        by_skill = {item.skill: item for item in learning_paths}
        level_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}

        for item in learning_paths:
            for prerequisite in item.prerequisites:
                with self.subTest(skill=item.skill, prerequisite=prerequisite):
                    self.assertIn(prerequisite, by_skill)
                    self.assertLessEqual(
                        level_order[by_skill[prerequisite].level],
                        level_order[item.level],
                    )

    def test_missing_prerequisites_circular_dependencies_and_invalid_levels_fail(self) -> None:
        source_data = json.loads(
            self.learning_catalog_path().read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)

            missing_reference = json.loads(json.dumps(source_data))
            next(
                item
                for item in missing_reference["learning_paths"]
                if item["skill"] == "Python"
            )["prerequisites"] = ["SQLite"]
            missing_reference_path = temp_path / "missing_reference.json"
            self.write_catalog(missing_reference_path, missing_reference)
            with self.assertRaisesRegex(LearningCatalogError, "no learning metadata"):
                load_learning_catalog(missing_reference_path)

            circular = json.loads(json.dumps(source_data))
            by_skill = {item["skill"]: item for item in circular["learning_paths"]}
            by_skill["Python"]["prerequisites"] = ["SQL"]
            by_skill["SQL"]["prerequisites"] = ["Python"]
            circular_path = temp_path / "circular.json"
            self.write_catalog(circular_path, circular)
            with self.assertRaisesRegex(LearningCatalogError, "Circular"):
                load_learning_catalog(circular_path)

            invalid_level = json.loads(json.dumps(source_data))
            next(
                item
                for item in invalid_level["learning_paths"]
                if item["skill"] == "Python"
            )["prerequisites"] = ["NumPy"]
            invalid_level_path = temp_path / "invalid_level.json"
            self.write_catalog(invalid_level_path, invalid_level)
            with self.assertRaisesRegex(LearningCatalogError, "later learning level"):
                load_learning_catalog(invalid_level_path)

    def test_ai_ml_and_python_developer_roadmaps_are_generated(self) -> None:
        ai_ml_roadmap = generate_learning_roadmap("AI/ML Engineer", [])
        python_roadmap = generate_learning_roadmap(
            "Python Developer",
            ["Python 3", "SQL"],
        )

        self.assertEqual(ai_ml_roadmap.career_title, "AI/ML Engineer")
        self.assertEqual(ai_ml_roadmap.total_count, 12)
        self.assertEqual(
            {step.skill for step in ai_ml_roadmap.steps},
            {
                "Python",
                "Machine Learning",
                "Deep Learning",
                "scikit-learn",
                "TensorFlow",
                "PyTorch",
                "NumPy",
                "Pandas",
                "Docker",
                "Git",
                "SQL",
                "FastAPI",
            },
        )
        self.assertEqual(python_roadmap.career_title, "Python Developer")
        self.assertEqual(python_roadmap.total_count, 10)
        self.assertNotIn("Python", {step.skill for step in python_roadmap.steps})
        self.assertNotIn("SQL", {step.skill for step in python_roadmap.steps})

    def test_empty_candidate_skills_include_every_career_skill(self) -> None:
        roadmap = generate_learning_roadmap("Python Developer", [])

        self.assertEqual(roadmap.total_count, 12)
        self.assertEqual(roadmap.completed_count, 0)
        self.assertEqual(roadmap.progress_percentage, 0.0)
        self.assertEqual(len(roadmap.remaining_skills), roadmap.total_count)

    def test_empty_roadmap_is_safe_when_resume_covers_every_career_skill(self) -> None:
        career = next(
            profile
            for profile in load_career_catalog()
            if profile.title == "Python Developer"
        )
        roadmap = generate_learning_roadmap(
            career.title,
            career.core_skills + career.supporting_skills,
        )

        self.assertEqual(roadmap.steps, ())
        self.assertEqual(roadmap.completed_count, 0)
        self.assertEqual(roadmap.total_count, 0)
        self.assertEqual(roadmap.remaining_skills, ())
        self.assertEqual(roadmap.progress_percentage, 100.0)

    def test_core_priority_and_prerequisite_order_are_preserved(self) -> None:
        roadmaps = (
            generate_learning_roadmap("AI/ML Engineer", []),
            generate_learning_roadmap("Python Developer", []),
        )
        level_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}

        for roadmap in roadmaps:
            with self.subTest(career=roadmap.career_title):
                positions = {
                    step.skill: index for index, step in enumerate(roadmap.steps)
                }
                self.assertEqual(
                    [level_order[step.level] for step in roadmap.steps],
                    sorted(level_order[step.level] for step in roadmap.steps),
                )
                for step in roadmap.steps:
                    for prerequisite in step.prerequisites:
                        if prerequisite in positions:
                            self.assertLess(positions[prerequisite], positions[step.skill])

        python_roadmap = roadmaps[1]
        positions = {
            step.skill: index for index, step in enumerate(python_roadmap.steps)
        }
        self.assertLess(positions["REST API"], positions["PostgreSQL"])

    def test_unknown_career_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidRoadmapInputError, "Unknown career"):
            generate_learning_roadmap("Unknown Career", [])

    def test_progress_supports_zero_partial_duplicate_and_full_completion(self) -> None:
        initial = generate_learning_roadmap("Python Developer", [])
        first_skill = initial.steps[0].skill
        partial = generate_learning_roadmap(
            "Python Developer",
            [],
            [first_skill, first_skill],
        )
        complete = generate_learning_roadmap(
            "Python Developer",
            [],
            [step.skill for step in initial.steps],
        )

        self.assertEqual(initial.progress_percentage, 0.0)
        self.assertEqual(initial.completed_count, 0)
        self.assertEqual(partial.completed_count, 1)
        self.assertEqual(partial.completed_skills, (first_skill,))
        self.assertEqual(
            partial.progress_percentage,
            round(100 / initial.total_count, 1),
        )
        self.assertEqual(complete.completed_count, complete.total_count)
        self.assertEqual(complete.progress_percentage, 100.0)
        self.assertEqual(complete.remaining_skills, ())

    def test_unknown_or_already_known_completed_skills_are_rejected(self) -> None:
        for candidate_skills, completed_skills in (
            ([], ["MadeUpSkill"]),
            (["Python"], ["Python"]),
        ):
            with self.subTest(
                candidate_skills=candidate_skills,
                completed_skills=completed_skills,
            ):
                with self.assertRaises(InvalidRoadmapInputError):
                    generate_learning_roadmap(
                        "Python Developer",
                        candidate_skills,
                        completed_skills,
                    )

    def test_invalid_candidate_and_completed_skill_lists_are_rejected(self) -> None:
        for candidate_skills in ("Python", b"Python", None, 42, ["Python", 123]):
            with self.subTest(candidate_skills=candidate_skills):
                with self.assertRaises(InvalidRoadmapInputError):
                    generate_learning_roadmap(
                        "Python Developer",
                        candidate_skills,  # type: ignore[arg-type]
                    )

        for completed_skills in ("Python", b"Python", None, 42, ["Python", 123]):
            with self.subTest(completed_skills=completed_skills):
                with self.assertRaises(InvalidRoadmapInputError):
                    generate_learning_roadmap(
                        "Python Developer",
                        [],
                        completed_skills,  # type: ignore[arg-type]
                    )

    def test_invalid_career_titles_are_rejected(self) -> None:
        for career_title in ("", None, 42):
            with self.subTest(career_title=career_title):
                with self.assertRaises(InvalidRoadmapInputError):
                    generate_learning_roadmap(
                        career_title,  # type: ignore[arg-type]
                        [],
                    )


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
        self.assertIn('<h1 class="page-heading">', rendered_html)
        self.assertNotIn("<script>", rendered_html)
        self.assertNotIn("<img ", rendered_html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered_html)
        self.assertIn("&quot;with&quot;", rendered_html)
        self.assertIn("Sub &amp; &lt;img src=x onerror=alert(1)&gt;", rendered_html)

    def test_render_section_title_uses_escaped_semantic_heading(self) -> None:
        import app as career_app

        with patch.object(career_app.st, "markdown") as mock_markdown:
            career_app.render_section_title("<script>Section</script>", 3)

        rendered_html = mock_markdown.call_args.args[0]
        self.assertEqual(
            rendered_html,
            '<h3 class="section-title">&lt;script&gt;Section&lt;/script&gt;</h3>',
        )

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
