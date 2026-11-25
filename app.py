"""Streamlit entry point for the AI Career Navigator prototype."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

import streamlit as st

from modules.resume_parser import (
    EmptyPDFError,
    InvalidPDFError,
    PDFPageLimitError,
    PDFTooLargeError,
    ScannedPDFError,
    MAX_UPLOAD_SIZE_BYTES,
    extract_text_from_pdf,
)
from modules.career_recommender import (
    CareerCatalogError,
    CareerRecommendation,
    InvalidCareerRecommendationInputError,
    load_career_catalog,
    recommend_careers_from_skills,
    recommend_careers_from_text,
)
from modules.job_matcher import (
    InvalidJobMatchingInputError,
    JobMatchAnalysis,
    match_job,
)
from modules.learning_roadmap import (
    InvalidRoadmapInputError,
    LearningCatalogError,
    RoadmapStep,
    generate_learning_roadmap,
)
from modules.skill_extractor import (
    SkillExtractionError,
    extract_skills,
    extract_skills_by_category,
)


st.set_page_config(
    page_title="AI Career Navigator",
    page_icon=":material/explore:",
    layout="wide",
    initial_sidebar_state="expanded",
)


PAGE_NAMES = (
    "Home",
    "Resume & Job",
    "Skill Analysis",
    "Career Recommendations",
    "Learning Roadmap",
)

_ROADMAP_PROGRESS_STATE = "learning_roadmap_progress_by_career"
_ROADMAP_SELECTED_CAREER_STATE = "learning_roadmap_selected_career"
_ROADMAP_CAREER_WIDGET_KEY = "learning_roadmap_career_selector"
_ROADMAP_RESUME_SKILL_SIGNATURE = "learning_roadmap_resume_skill_signature"
_ROADMAP_RESET_CONFIRMATION_STATE = "learning_roadmap_reset_confirmation"
_RESUME_UPLOAD_RESULT_STATE = "resume_upload_result"
_STYLESHEET_PATH = Path(__file__).resolve().parent / "assets" / "styles.css"


def apply_styles() -> None:
    """Apply the shared stylesheet for app-owned visual elements."""
    st.html(_STYLESHEET_PATH)


def render_heading(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="page-heading">{html.escape(title)}</h1>'
        f'<p class="page-subtitle">{html.escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )


def render_section_title(title: str, heading_level: int = 2) -> None:
    tag = f"h{heading_level}"
    st.markdown(
        f'<{tag} class="section-title">{html.escape(title)}</{tag}>',
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def clear_stale_job_match_analysis() -> bool:
    """Discard displayed results when either session input has changed."""
    if st.session_state.get("job_match_analysis") is None:
        return False
    current_inputs = (
        st.session_state.get("resume_text", ""),
        st.session_state.get("job_description_text", ""),
    )
    if st.session_state.get("job_match_inputs") == current_inputs:
        return False
    st.session_state.pop("job_match_analysis", None)
    st.session_state.pop("job_match_inputs", None)
    return True


def render_home() -> None:
    resume_text = st.session_state.get("resume_text", "")
    job_description = st.session_state.get("job_description_text", "")
    has_resume = isinstance(resume_text, str) and bool(resume_text.strip())
    has_job_description = (
        isinstance(job_description, str) and bool(job_description.strip())
    )
    detected_skills: list[str] = []
    skill_extraction_failed = False
    if has_resume:
        try:
            detected_skills = extract_skills(resume_text)
        except SkillExtractionError as error:
            skill_extraction_failed = True
            st.error(f"Resume skills are unavailable: {error}")

    current_inputs = (resume_text, job_description)
    analysis_is_current = (
        st.session_state.get("job_match_analysis") is not None
        and st.session_state.get("job_match_inputs") == current_inputs
    )
    if analysis_is_current:
        analysis_status = "Ready - analysis complete"
    elif has_resume and has_job_description:
        analysis_status = "Ready to analyze"
    elif has_resume:
        analysis_status = "Add job description"
    elif has_job_description:
        analysis_status = "Add resume"
    else:
        analysis_status = "Not started"

    with st.container(border=True):
        render_heading(
            "Career workspace",
            "Welcome to AI Career Navigator",
            "Turn your experience into a clear next step.",
        )
        if not has_resume:
            st.info(
                "Upload a text-based PDF in Resume & Job to extract skills and "
                "start exploring career paths.",
                title="Start with your resume",
            )
        elif not has_job_description:
            st.info(
                "Add a target job description in Resume & Job to compare role "
                "requirements with your extracted skills.",
                title="Add a target role",
            )
        elif not analysis_is_current:
            st.info(
                "Open Skill Analysis and select Analyze Job Match to review "
                "your skill coverage.",
                title="Run your match analysis",
            )
        else:
            st.success(
                "Review your match in Skill Analysis or explore ranked paths "
                "in Career Recommendations.",
                title="Your analysis is ready",
            )

    first, second = st.columns(2)
    with first:
        render_metric("Resume", "Text extracted" if has_resume else "Not provided")
    with second:
        if not has_resume:
            skills_status = "Awaiting resume"
        elif skill_extraction_failed:
            skills_status = "Unavailable"
        else:
            skills_status = f"{len(detected_skills)} found"
        render_metric("Extracted skills", skills_status)

    third, fourth = st.columns(2)
    with third:
        render_metric(
            "Job description",
            "Description added" if has_job_description else "Not provided",
        )
    with fourth:
        render_metric("Career analysis", analysis_status)

    render_section_title("Skills detected")
    with st.container(border=True):
        if not has_resume:
            st.caption("Detected skills will appear here after you add a resume.")
        elif skill_extraction_failed:
            st.caption("Skills could not be listed because extraction failed.")
        elif detected_skills:
            st.write(", ".join(detected_skills))
        else:
            st.info(
                "No skills from the current taxonomy were detected in this resume.",
                title="No matching skills found",
            )

    render_section_title("Setup progress")
    steps = (
        ("Add a resume", has_resume),
        ("Add a job description", has_job_description),
        ("Analyze the match", analysis_is_current),
    )
    completed_steps = sum(is_complete for _, is_complete in steps)
    st.progress(
        completed_steps / len(steps),
        text=f"{completed_steps} of {len(steps)} steps complete",
    )
    step_columns = st.columns(len(steps))
    for column, (label, is_complete) in zip(step_columns, steps):
        with column:
            icon = ":material/check_circle:" if is_complete else ":material/pending:"
            st.markdown(f"{icon} **{label}**")


def render_resume_and_job() -> None:
    render_heading(
        "Candidate inputs",
        "Build your candidate profile",
        "Add a resume and target role to see how your experience aligns.",
    )
    with st.container(border=True):
        render_section_title("Resume")
        st.caption("PDF only · maximum 10 MB · maximum 20 pages")
        uploaded_file = st.file_uploader(
            "Upload your resume",
            type=["pdf"],
            accept_multiple_files=False,
            key="resume_upload",
            max_upload_size=10,
            help=(
                "PDF files only; maximum 10 MB and 20 pages. "
                "The resume is processed in memory and is not saved."
            ),
        )
        if uploaded_file is not None:
            st.caption(
                f"Processing {uploaded_file.name} in memory; the resume is not saved."
            )
            upload_size = getattr(uploaded_file, "size", None)
            if (
                isinstance(upload_size, int)
                and upload_size > MAX_UPLOAD_SIZE_BYTES
            ):
                upload_signature = f"oversize:{upload_size}"
            else:
                upload_signature = hashlib.sha256(
                    uploaded_file.getvalue()
                ).hexdigest()
            upload_result = st.session_state.get(_RESUME_UPLOAD_RESULT_STATE)
            if (
                not isinstance(upload_result, tuple)
                or len(upload_result) != 3
                or upload_result[0] != upload_signature
            ):
                try:
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    if not extracted_text.strip():
                        raise EmptyPDFError(
                            "The PDF contains no extractable text."
                        )
                except PDFTooLargeError as error:
                    upload_result = (upload_signature, "error", str(error))
                except PDFPageLimitError as error:
                    upload_result = (upload_signature, "error", str(error))
                except ScannedPDFError as error:
                    upload_result = (upload_signature, "info", str(error))
                except EmptyPDFError as error:
                    upload_result = (upload_signature, "warning", str(error))
                except InvalidPDFError as error:
                    upload_result = (upload_signature, "error", str(error))
                else:
                    st.session_state["resume_text"] = extracted_text
                    upload_result = (
                        upload_signature,
                        "success",
                        f"{len(extracted_text):,} characters extracted.",
                    )
                st.session_state[_RESUME_UPLOAD_RESULT_STATE] = upload_result

            _, upload_status, upload_message = upload_result
            if upload_status == "success":
                st.success(
                    f"{uploaded_file.name} processed; {upload_message}"
                )
            elif upload_status == "info":
                st.info(upload_message)
            elif upload_status == "warning":
                st.warning(upload_message)
            else:
                st.error(upload_message)

            resume_text = st.session_state.get("resume_text", "")
            if isinstance(resume_text, str) and resume_text.strip():
                if upload_status != "success":
                    st.caption(
                        "The previous valid resume text is still available in "
                        "this session."
                    )
                st.text_area(
                    "Extracted resume text",
                    value=resume_text,
                    height=260,
                    disabled=True,
                )

        else:
            resume_text = st.session_state.get("resume_text", "")
            if isinstance(resume_text, str) and resume_text.strip():
                st.success("Resume text is available in this session.")
                st.caption(
                    "The original PDF is not retained; only extracted text remains "
                    "in this active session."
                )
                if st.button("Clear extracted resume text"):
                    st.session_state.pop("resume_text", None)
                    st.session_state.pop(_RESUME_UPLOAD_RESULT_STATE, None)
                    st.session_state.pop("job_match_analysis", None)
                    st.session_state.pop("job_match_inputs", None)
                    st.success("Resume text cleared from this session.")
                else:
                    st.text_area(
                        "Extracted resume text",
                        value=resume_text,
                        height=260,
                        disabled=True,
                    )
            else:
                st.info(
                    "Upload a text-based PDF to extract skills and prepare your "
                    "career analysis.",
                    title="No resume added yet",
                )

    with st.container(border=True):
        render_section_title("Target job description")
        st.caption("Paste the responsibilities and skills required for your target role.")
        if "job_description_input" not in st.session_state:
            st.session_state["job_description_input"] = st.session_state.get(
                "job_description_text",
                "",
            )
        st.text_area(
            "Target job description",
            key="job_description_input",
            height=220,
            placeholder="Paste a job description here…",
        )
        st.session_state["job_description_text"] = st.session_state[
            "job_description_input"
        ]
        if clear_stale_job_match_analysis():
            st.info("Inputs changed. Previous job-match results were cleared.")
        if st.session_state["job_description_text"].strip():
            st.success("Job description is ready for analysis.")
        else:
            st.caption("No target job description added yet.")

    st.caption(
        "Privacy: uploaded PDFs are not saved. Extracted resume text and the job "
        "description remain in this active Streamlit session."
    )


def render_categorized_skills(
    title: str,
    skills_by_category: dict[str, list[str]],
    empty_message: str,
    *,
    heading_level: int = 3,
) -> None:
    render_section_title(title, heading_level)
    populated_categories = [
        (category, skills)
        for category, skills in skills_by_category.items()
        if skills
    ]
    if not populated_categories:
        st.caption(empty_message)
        return
    for category, skills in populated_categories:
        with st.expander(f"{category} ({len(skills)})", expanded=True):
            st.write(", ".join(skills))


def render_skill_overview(
    skills_by_category: dict[str, list[str]],
) -> None:
    total_skills = sum(len(skills) for skills in skills_by_category.values())
    with st.container(border=True):
        render_section_title("Resume skill overview")
        st.caption(
            "Counts include skills found in the existing taxonomy; "
            "unrecognized terms are not included."
        )
        st.metric("Total extracted skills", total_skills, border=True)
        render_categorized_skills(
            "Skills by category",
            skills_by_category,
            "No taxonomy skills were detected in the resume.",
            heading_level=3,
        )


def render_job_match_results(analysis: JobMatchAnalysis) -> None:
    render_section_title("Match overview")
    render_metric(
        "Heuristic match estimate",
        (
            f"{analysis.overall_match_score:.1f}%"
            if analysis.overall_match_score is not None
            else "Unavailable"
        ),
    )
    st.caption(
        "The heuristic estimate combines skill coverage and lexical similarity; "
        "it is not a probability of getting hired. "
        f"{analysis.text_similarity_method} measures word overlap, not meaning."
    )
    if analysis.score_unavailable_reason:
        st.info(analysis.score_unavailable_reason, title="Score unavailable")

    missing_skill_sources = []
    if not analysis.candidate_skills:
        missing_skill_sources.append("the resume")
    if not analysis.required_skills:
        missing_skill_sources.append("the job description")
    if missing_skill_sources:
        st.warning(
            "No taxonomy skills were detected in "
            + " and ".join(missing_skill_sources)
            + ". Skill-based results may be incomplete; review the text or taxonomy."
        )

    render_skill_overview(analysis.candidate_skills_by_category)

    with st.container(border=True):
        render_section_title("Job-skill coverage")
        if analysis.skill_coverage_score is not None:
            st.progress(
                analysis.skill_coverage_score / 100,
                text=(
                    f"{analysis.skill_coverage_score:.1f}% of detected job skills "
                    "matched"
                ),
            )
            st.caption(
                "Coverage is the share of taxonomy skills detected in the job "
                "description that also appear in the resume."
            )
        else:
            st.caption(
                "Coverage is unavailable because no comparable job-skill set "
                "could be calculated."
            )

    if analysis.text_similarity_score is not None:
        with st.container(border=True):
            render_section_title("Text similarity")
            st.progress(
                analysis.text_similarity_score / 100,
                text=f"{analysis.text_similarity_score:.1f}% lexical similarity",
            )
            st.caption(
                "TF-IDF cosine similarity reflects word overlap between the "
                "resume and job description, not semantic meaning."
            )

    render_section_title("Skill comparison")
    matching_column, missing_column = st.columns(2)
    with matching_column:
        with st.container(border=True):
            render_section_title("Matched skills", 3)
            st.caption("Required taxonomy skills also found in the resume.")
            if analysis.matching_skills:
                st.success(
                    f"{len(analysis.matching_skills)} detected job skills matched."
                )
                render_categorized_skills(
                    "Matched by category",
                    analysis.matching_skills_by_category,
                    "No detected job skills matched the resume.",
                    heading_level=4,
                )
            elif analysis.required_skills:
                st.info("No detected job skills matched the resume.")
            else:
                st.info("No taxonomy job skills are available to compare.")
    with missing_column:
        with st.container(border=True):
            render_section_title("Missing skills", 3)
            st.caption(
                "Detected job requirements not found in the resume text; "
                "add them only if they reflect your experience."
            )
            if analysis.missing_skills:
                st.warning(
                    f"{len(analysis.missing_skills)} detected job skills "
                    "were not found."
                )
                render_categorized_skills(
                    "Missing by category",
                    analysis.missing_skills_by_category,
                    "No detected job skills are missing from the resume.",
                    heading_level=4,
                )
            elif analysis.required_skills:
                st.success("All detected job skills were found in the resume.")
            else:
                st.info("No taxonomy job skills are available to compare.")

    with st.container(border=True):
        render_section_title("Additional resume skills")
        st.caption("These taxonomy skills were detected in the resume, not the job description.")
        render_categorized_skills(
            "Additional by category",
            analysis.additional_skills_by_category,
            "No additional taxonomy skills were detected.",
            heading_level=3,
        )

    strengths_column, improvement_column = st.columns(2)
    with strengths_column:
        render_section_title("Strengths", 3)
        if analysis.strengths:
            for strength in analysis.strengths:
                st.markdown(f"- {strength}")
        else:
            st.caption("No explicit matching or additional skills were detected.")
    with improvement_column:
        render_section_title("Areas for improvement", 3)
        if analysis.improvement_suggestions:
            for suggestion in analysis.improvement_suggestions:
                st.markdown(f"- {suggestion}")
        else:
            st.caption("No missing taxonomy skills were detected.")


def render_skill_analysis() -> None:
    render_heading(
        "Evidence-based comparison",
        "Job match analysis",
        (
            "Review detected resume skills against a target role using taxonomy "
            "matching and lexical text similarity."
        ),
    )
    resume_text = st.session_state.get("resume_text", "")
    job_description = st.session_state.get("job_description_text", "")
    has_resume = isinstance(resume_text, str) and bool(resume_text.strip())
    has_job_description = (
        isinstance(job_description, str) and bool(job_description.strip())
    )
    current_inputs = (resume_text, job_description)

    with st.container(border=True):
        render_section_title("Analysis inputs")
        st.caption("A comparison needs resume text and a target job description.")
        resume_column, job_column = st.columns(2)
        with resume_column:
            st.markdown("**Resume**")
            if has_resume:
                st.success("Resume text is available in this session.")
            else:
                st.caption("Not added yet.")
        with job_column:
            st.markdown("**Job description**")
            if has_job_description:
                st.success("Job description is available in this session.")
            else:
                st.caption("Not added yet.")

    skill_taxonomy_failed = False
    if st.button("Analyze Job Match", type="primary"):
        st.session_state.pop("job_match_analysis", None)
        st.session_state.pop("job_match_inputs", None)
        invalid_input = False
        if not isinstance(resume_text, str):
            st.error("Resume text is invalid. Upload a readable PDF in Resume & Job.")
            invalid_input = True
        elif not resume_text.strip():
            st.warning("Resume is missing. Upload a text-based PDF in Resume & Job.")
            invalid_input = True
        if not isinstance(job_description, str):
            st.error("Job description is invalid. Enter text in Resume & Job.")
            invalid_input = True
        elif not job_description.strip():
            st.warning("Job description is missing. Add it in Resume & Job.")
            invalid_input = True
        if not invalid_input:
            try:
                with st.spinner("Comparing resume and job description..."):
                    analysis = match_job(resume_text, job_description)
            except InvalidJobMatchingInputError as error:
                st.error(str(error))
            except SkillExtractionError as error:
                st.error(f"Skill taxonomy is unavailable: {error}")
                skill_taxonomy_failed = True
            except ImportError as error:
                st.error(f"Scikit-learn TF-IDF is unavailable: {error}")
            else:
                st.session_state["job_match_analysis"] = analysis
                st.session_state["job_match_inputs"] = current_inputs

    if not has_resume and not has_job_description:
        st.info(
            "Add a resume and a target job description to compare skills.",
            title="Inputs needed",
        )
    elif not has_resume:
        st.info(
            "Add a resume to see your extracted skills and compare them with "
            "the job requirements.",
            title="Resume needed",
        )
    elif not has_job_description:
        st.info(
            "Add a target job description to identify matched and missing skills.",
            title="Job description needed",
        )

    if clear_stale_job_match_analysis():
        st.info("Inputs changed. Click Analyze Job Match to calculate updated results.")
    analysis = st.session_state.get("job_match_analysis")
    if analysis is not None:
        render_job_match_results(analysis)
    elif has_resume:
        try:
            skills_by_category = extract_skills_by_category(resume_text)
        except SkillExtractionError as error:
            if not skill_taxonomy_failed:
                st.error(f"Skill taxonomy is unavailable: {error}")
        else:
            render_skill_overview(skills_by_category)
            if has_job_description:
                st.caption(
                    "Select Analyze Job Match to compare these resume skills "
                    "with the target role."
                )


def render_pending_page(title: str, subtitle: str, details: str) -> None:
    render_heading("Coming later", title, subtitle)
    st.markdown(
        f'<div class="status-panel" role="status">'
        f'<strong>Not implemented</strong><span>{html.escape(details)}</span></div>',
        unsafe_allow_html=True,
    )


def render_career_recommendation_card(
    rank: int,
    recommendation: CareerRecommendation,
) -> None:
    """Render an explainable career recommendation card."""
    with st.container(border=True):
        render_section_title(f"#{rank} · {recommendation.title}")
        st.caption(recommendation.description)
        with st.grid(2, min_column_width=180, gap="small"):
            st.metric(
                "Core skill coverage",
                f"{recommendation.core_coverage_score:.1f}%",
                border=True,
            )
            st.metric(
                "Supporting skill coverage",
                f"{recommendation.supporting_coverage_score:.1f}%",
                border=True,
            )
        st.progress(
            recommendation.suitability_score / 100,
            text=f"{recommendation.suitability_score:.1f}% heuristic alignment",
        )

        st.markdown("**Why this role?**")
        st.write(recommendation.explanation)

        core_column, supporting_column = st.columns(2)
        with core_column:
            with st.container(border=True):
                render_section_title("Core skills", 3)
                st.caption("Primary requirements for this career path.")
                if recommendation.matching_core_skills:
                    st.success(
                        "Matched: "
                        + ", ".join(recommendation.matching_core_skills)
                    )
                else:
                    st.info("No core skills were matched in the resume.")
                if recommendation.missing_core_skills:
                    st.warning(
                        "Not detected: "
                        + ", ".join(recommendation.missing_core_skills)
                    )
                else:
                    st.success("All tracked core skills were detected.")

        with supporting_column:
            with st.container(border=True):
                render_section_title("Supporting skills", 3)
                st.caption("Additional skills that strengthen this role match.")
                if recommendation.matching_supporting_skills:
                    st.success(
                        "Matched: "
                        + ", ".join(recommendation.matching_supporting_skills)
                    )
                else:
                    st.caption("No supporting skills were matched in the resume.")
                if recommendation.missing_supporting_skills:
                    st.warning(
                        "Not detected: "
                        + ", ".join(recommendation.missing_supporting_skills)
                    )
                elif recommendation.matching_supporting_skills:
                    st.success("All tracked supporting skills were detected.")
                else:
                    st.caption("No supporting skills are tracked for this role.")

        with st.container(border=True):
            render_section_title("Suggested skills to learn next", 3)
            if recommendation.next_skills_to_learn:
                st.write(", ".join(recommendation.next_skills_to_learn[:5]))
            else:
                st.success("All tracked skills for this role are present.")


def render_career_recommendations() -> None:
    render_heading(
        "Explainable role guidance",
        "Career recommendations",
        "Ranked career paths based on taxonomy skills detected in your uploaded resume.",
    )
    resume_text = st.session_state.get("resume_text", "")
    if not isinstance(resume_text, str) or not resume_text.strip():
        st.warning(
            "Resume is missing. Upload a text-based PDF in Resume & Job to "
            "generate career recommendations.",
            title="Resume needed",
        )
        return

    try:
        detected_skills = extract_skills(resume_text)
        recommendations = recommend_careers_from_text(resume_text)
    except (
        CareerCatalogError,
        InvalidCareerRecommendationInputError,
        SkillExtractionError,
    ) as error:
        st.error(f"Career recommendations are unavailable: {error}")
        return

    if detected_skills:
        st.caption(
            f"Detected resume skills ({len(detected_skills)}): "
            + ", ".join(detected_skills)
        )
    else:
        st.warning(
            "No curated taxonomy skills were detected in the resume. "
            "All roles show 0.0% alignment; review extracted text or expand skills."
        )

    st.caption(
        "Suitability scores are heuristic skill-alignment estimates (70% core "
        "skills, 30% supporting skills), not a probability of getting hired."
    )
    if not recommendations:
        st.info("No career recommendations are available for this resume.")
        return
    for index, recommendation in enumerate(recommendations, start=1):
        render_career_recommendation_card(index, recommendation)


def update_learning_roadmap_progress(
    career_title: str,
    skill: str,
    checkbox_key: str,
) -> None:
    """Persist one checkbox change without mixing progress across careers."""
    completed_by_career = dict(
        st.session_state.get(_ROADMAP_PROGRESS_STATE, {})
    )
    completed_skills = set(completed_by_career.get(career_title, ()))
    if st.session_state.get(checkbox_key, False):
        completed_skills.add(skill)
    else:
        completed_skills.discard(skill)
    completed_by_career[career_title] = tuple(sorted(completed_skills))
    st.session_state[_ROADMAP_PROGRESS_STATE] = completed_by_career


def begin_learning_roadmap_reset(career_title: str) -> None:
    """Ask for confirmation before clearing one career's progress."""
    st.session_state[_ROADMAP_RESET_CONFIRMATION_STATE] = career_title


def clear_learning_roadmap_reset_confirmation() -> None:
    """Dismiss the pending progress reset confirmation."""
    st.session_state.pop(_ROADMAP_RESET_CONFIRMATION_STATE, None)


def reset_learning_roadmap_progress(
    career_title: str,
    career_skills: tuple[str, ...],
) -> None:
    """Clear only the selected career's saved progress and checkbox state."""
    completed_by_career = dict(st.session_state.get(_ROADMAP_PROGRESS_STATE, {}))
    completed_by_career.pop(career_title, None)
    st.session_state[_ROADMAP_PROGRESS_STATE] = completed_by_career
    for skill in career_skills:
        checkbox_key = f"learning_roadmap_done_{career_title}_{skill}"
        st.session_state[checkbox_key] = False
    clear_learning_roadmap_reset_confirmation()


def render_learning_roadmap_step(
    career_title: str,
    step: RoadmapStep,
) -> None:
    """Render one roadmap step and its session-persisted completion checkbox."""
    checkbox_key = f"learning_roadmap_done_{career_title}_{step.skill}"
    if checkbox_key not in st.session_state:
        st.session_state[checkbox_key] = step.is_completed

    with st.container(border=True):
        st.checkbox(
            f"{step.skill} ({step.level})",
            key=checkbox_key,
            on_change=update_learning_roadmap_progress,
            args=(career_title, step.skill, checkbox_key),
        )
        st.badge(
            "Completed" if step.is_completed else "Pending",
            color="green" if step.is_completed else "orange",
        )
        st.badge(
            "Core skill" if step.is_core_skill else "Supporting skill",
            color="blue" if step.is_core_skill else "gray",
        )
        prerequisites = ", ".join(step.prerequisites) or "None"
        st.write(f"Prerequisites: {prerequisites}")
        st.markdown(f"**Mini-project:** {step.mini_project}")
        st.markdown(f"**Completion milestone:** {step.milestone}")


def render_learning_roadmap() -> None:
    """Render a resume-personalized or complete career learning roadmap."""
    render_heading(
        "Career development",
        "Learning roadmap",
        "Track practical learning milestones for a career path that interests you.",
    )
    try:
        careers = load_career_catalog()
    except CareerCatalogError as error:
        st.error(f"Learning roadmaps are unavailable: {error}")
        return

    career_titles = [career.title for career in careers]
    resume_text = st.session_state.get("resume_text")
    if resume_text is not None and not isinstance(resume_text, str):
        st.error("Resume text is invalid. Upload a readable PDF in Resume & Job.")
        return

    has_resume = isinstance(resume_text, str) and bool(resume_text.strip())
    candidate_skills: list[str] = []
    recommended_career_title: str | None = None
    if has_resume:
        try:
            candidate_skills = extract_skills(resume_text)
            recommendations = recommend_careers_from_skills(
                candidate_skills,
                top_n=1,
            )
        except (
            CareerCatalogError,
            InvalidCareerRecommendationInputError,
            SkillExtractionError,
        ) as error:
            st.error(f"Career recommendations are unavailable: {error}")
            return
        if not recommendations:
            st.error("No career recommendations are available.")
            return

        recommended_career_title = recommendations[0].title
        skill_signature = tuple(candidate_skills)
        if st.session_state.get(_ROADMAP_RESUME_SKILL_SIGNATURE) != skill_signature:
            st.session_state[_ROADMAP_RESUME_SKILL_SIGNATURE] = skill_signature
            st.session_state[_ROADMAP_SELECTED_CAREER_STATE] = (
                recommended_career_title
            )
            st.session_state[_ROADMAP_CAREER_WIDGET_KEY] = (
                recommended_career_title
            )
        elif st.session_state.get(_ROADMAP_CAREER_WIDGET_KEY) not in career_titles:
            selected_career = st.session_state.get(
                _ROADMAP_SELECTED_CAREER_STATE
            )
            st.session_state[_ROADMAP_CAREER_WIDGET_KEY] = (
                selected_career
                if selected_career in career_titles
                else recommended_career_title
            )

        if candidate_skills:
            st.caption(
                f"Detected resume skills ({len(candidate_skills)}): "
                + ", ".join(candidate_skills)
            )
        else:
            st.info(
                "No curated taxonomy skills were detected. The selected career's "
                "complete roadmap is shown; review the resume text or skill taxonomy."
            )
    else:
        st.session_state[_ROADMAP_RESUME_SKILL_SIGNATURE] = None
        if st.session_state.get(_ROADMAP_CAREER_WIDGET_KEY) not in career_titles:
            selected_career = st.session_state.get(
                _ROADMAP_SELECTED_CAREER_STATE
            )
            st.session_state[_ROADMAP_CAREER_WIDGET_KEY] = (
                selected_career if selected_career in career_titles else career_titles[0]
            )
        st.info(
            "Upload a resume in Resume & Job to personalize this roadmap and "
            "automatically select your highest-ranked career recommendation."
        )

    with st.container(border=True):
        render_section_title("Choose a career path")
        career_title = st.selectbox(
            "Career path",
            career_titles,
            key=_ROADMAP_CAREER_WIDGET_KEY,
            on_change=clear_learning_roadmap_reset_confirmation,
        )
        if has_resume:
            st.caption(
                "The highest-ranked recommendation is selected automatically; "
                "you can choose another career at any time."
            )
        else:
            st.caption(
                "Choose any career to explore its complete learning roadmap."
            )
    st.session_state[_ROADMAP_SELECTED_CAREER_STATE] = career_title
    career = next(profile for profile in careers if profile.title == career_title)

    completed_by_career = st.session_state.setdefault(
        _ROADMAP_PROGRESS_STATE,
        {},
    )
    if not isinstance(completed_by_career, dict):
        st.error("Learning progress is invalid for this session.")
        return
    career_skills = set(career.core_skills + career.supporting_skills)
    raw_saved_completed_skills = completed_by_career.get(career_title, ())
    if (
        not isinstance(raw_saved_completed_skills, (tuple, list, set))
        or any(
            not isinstance(skill, str)
            for skill in raw_saved_completed_skills
        )
    ):
        st.error("Saved learning progress is invalid for this career.")
        return
    saved_completed_skills = (
        set(raw_saved_completed_skills) & career_skills
    )
    if saved_completed_skills:
        completed_by_career[career_title] = tuple(
            sorted(saved_completed_skills)
        )
    else:
        completed_by_career.pop(career_title, None)
    st.session_state[_ROADMAP_PROGRESS_STATE] = completed_by_career
    currently_missing_skills = career_skills - set(candidate_skills)
    completed_skills = saved_completed_skills & currently_missing_skills
    try:
        roadmap = generate_learning_roadmap(
            career_title,
            candidate_skills,
            completed_skills,
        )
    except (InvalidRoadmapInputError, LearningCatalogError) as error:
        st.error(f"Learning roadmap is unavailable: {error}")
        return

    with st.container(border=True):
        render_section_title(career.title)
        st.write(career.description)
        with st.grid(3, min_column_width=160, gap="small"):
            st.metric(
                "Total roadmap skills",
                roadmap.total_count,
                border=True,
            )
            st.metric(
                "Completed",
                roadmap.completed_count,
                border=True,
            )
            st.metric(
                "Remaining",
                len(roadmap.remaining_skills),
                border=True,
            )
        if roadmap.total_count > 0:
            st.progress(
                roadmap.progress_percentage / 100,
                text=f"{roadmap.progress_percentage:.1f}% complete",
            )
        else:
            render_metric("Overall completion", "N/A")
        st.caption(
            f"{roadmap.completed_count} of {roadmap.total_count} skills complete."
        )
        st.markdown(
            "**Completed skills:** "
            + (", ".join(roadmap.completed_skills) or "None")
        )
        st.markdown(
            "**Remaining skills:** "
            + (", ".join(roadmap.remaining_skills) or "None")
        )

    if st.session_state.get(_ROADMAP_RESET_CONFIRMATION_STATE) == career_title:
        st.warning(
            f"Reset all learning progress for {career_title}? "
            "This cannot be undone."
        )
        confirm_column, cancel_column = st.columns(2)
        with confirm_column:
            st.button(
                "Confirm reset",
                key=f"learning_roadmap_confirm_reset_{career_title}",
                on_click=reset_learning_roadmap_progress,
                args=(career_title, tuple(sorted(career_skills))),
            )
        with cancel_column:
            st.button(
                "Keep progress",
                key=f"learning_roadmap_cancel_reset_{career_title}",
                on_click=clear_learning_roadmap_reset_confirmation,
            )
    else:
        st.button(
            "Reset progress for this career",
            key=f"learning_roadmap_reset_{career_title}",
            on_click=begin_learning_roadmap_reset,
            args=(career_title,),
        )

    if not roadmap.steps:
        st.success("Your resume already covers all tracked skills for this career.")

    for stage_name, stage_steps in (
        ("Beginner", roadmap.beginner_steps),
        ("Intermediate", roadmap.intermediate_steps),
        ("Advanced", roadmap.advanced_steps),
    ):
        with st.container(border=True):
            render_section_title(stage_name)
            if not stage_steps:
                st.caption(
                    f"No {stage_name.lower()} skills remain in this roadmap."
                )
                continue
            st.caption(
                f"{len(stage_steps)} skill"
                + ("" if len(stage_steps) == 1 else "s")
                + " in this stage."
            )
            for step in stage_steps:
                render_learning_roadmap_step(career_title, step)


def main() -> None:
    apply_styles()
    with st.sidebar:
        st.markdown(
            '<div class="brand-mark"><span class="brand-icon">N</span>'
            '<span>Career Navigator</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("CAREER WORKSPACE")
        selected_page = st.radio(
            "Navigation",
            PAGE_NAMES,
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown('<span class="prototype-tag">EARLY PROTOTYPE</span>', unsafe_allow_html=True)

    if selected_page == "Home":
        render_home()
    elif selected_page == "Resume & Job":
        render_resume_and_job()
    elif selected_page == "Skill Analysis":
        render_skill_analysis()
    elif selected_page == "Career Recommendations":
        render_career_recommendations()
    else:
        render_learning_roadmap()


if __name__ == "__main__":
    main()
