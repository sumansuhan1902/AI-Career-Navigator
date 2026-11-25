"""Streamlit entry point for the AI Career Navigator prototype."""

from __future__ import annotations

import html

import streamlit as st

from modules.resume_parser import (
    EmptyPDFError,
    InvalidPDFError,
    PDFPageLimitError,
    PDFTooLargeError,
    ScannedPDFError,
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
from modules.skill_extractor import SkillExtractionError, extract_skills


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


def apply_styles() -> None:
    """Apply the dashboard's small set of shared visual styles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

        :root {
            --ink: #172b3a;
            --muted: #647783;
            --line: #dce5e8;
            --surface: #ffffff;
            --canvas: #f3f7f6;
            --teal: #087e78;
            --teal-soft: #e3f3f0;
            --amber: #a96321;
        }
        html, body, [class*="css"] {
            font-family: 'DM Sans', 'Segoe UI', sans-serif;
            color: var(--ink);
        }
        .stApp { background: var(--canvas); }
        [data-testid="stSidebar"] {
            background: #edf4f2;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, h1, h2, h3 {
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            color: var(--ink);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            font-weight: 600;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label,
        [data-testid="stSidebar"] [data-testid="stRadio"] label p,
        [data-testid="stSidebar"] p {
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] .stCaptionContainer p {
            color: var(--muted) !important;
        }
        .brand-mark {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0 0 24px;
            font: 800 17px 'Manrope', 'Segoe UI', sans-serif;
            letter-spacing: 0;
            color: var(--ink);
        }
        .brand-icon {
            display: grid;
            place-items: center;
            width: 34px;
            height: 34px;
            border-radius: 9px;
            color: white;
            background: var(--teal);
        }
        .eyebrow {
            color: var(--teal);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .page-heading {
            margin: 4px 0 6px;
            color: var(--ink);
            font: 800 30px 'Manrope', 'Segoe UI', sans-serif;
            letter-spacing: 0;
            line-height: 1.2;
        }
        .page-subtitle { color: var(--muted); margin: 0 0 24px; }
        .metric-card {
            min-height: 112px;
            padding: 18px 20px;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(23, 43, 58, 0.035);
        }
        .metric-label { color: var(--muted); font-size: 13px; font-weight: 600; }
        .metric-value {
            margin-top: 11px;
            color: var(--ink);
            font: 700 20px 'Manrope', 'Segoe UI', sans-serif;
            letter-spacing: 0;
        }
        .section-title {
            margin: 28px 0 10px;
            font: 700 18px 'Manrope', 'Segoe UI', sans-serif;
        }
        .status-panel {
            padding: 18px 20px;
            background: var(--teal-soft);
            border-left: 3px solid var(--teal);
            border-radius: 4px;
            color: var(--ink);
        }
        .status-panel strong { display: block; margin-bottom: 4px; }
        .status-panel span { color: #4c666c; font-size: 14px; }
        .prototype-tag {
            display: inline-block;
            padding: 5px 9px;
            border: 1px solid #e8cfb4;
            border-radius: 4px;
            color: var(--amber);
            background: #fbf2e8;
            font-size: 12px;
            font-weight: 700;
        }
        div[data-testid="stFileUploader"] section,
        div[data-testid="stTextArea"] textarea {
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_heading(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="page-heading">{html.escape(title)}</div>'
        f'<p class="page-subtitle">{html.escape(subtitle)}</p>',
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
    render_heading(
        "Workspace overview",
        "AI Career Navigator",
        "A focused workspace for exploring how your experience aligns with your next role.",
    )
    resume_text = st.session_state.get("resume_text", "")
    job_description = st.session_state.get("job_description_text", "")
    resume_status = "Text extracted" if resume_text else "Not provided"
    job_status = "Description added" if job_description.strip() else "Not provided"
    analysis_status = (
        "Ready"
        if resume_text and job_description.strip()
        else "Inputs incomplete"
    )

    first, second, third = st.columns(3)
    with first:
        render_metric("Resume", resume_status)
    with second:
        render_metric("Job description", job_status)
    with third:
        render_metric("Analysis status", analysis_status)

    st.markdown('<div class="section-title">Start here</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="status-panel"><strong>Skill comparison and career guidance use curated taxonomy matching</strong>'
        '<span>Add a resume and target job description in Resume &amp; Job to '
        'inspect detected and missing skills, or open Career Recommendations '
        'to explore ranked career paths from your resume.</span></div>',
        unsafe_allow_html=True,
    )


def render_resume_and_job() -> None:
    render_heading(
        "Candidate inputs",
        "Resume & job description",
        "Upload a text-based resume and add the target role requirements.",
    )
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
        st.caption(f"Processing {uploaded_file.name} in memory; the resume is not saved.")
        st.session_state.pop("resume_text", None)
        try:
            extracted_text = extract_text_from_pdf(uploaded_file)
        except PDFTooLargeError as error:
            st.error(str(error))
        except PDFPageLimitError as error:
            st.error(str(error))
        except ScannedPDFError as error:
            st.info(str(error))
        except EmptyPDFError as error:
            st.warning(str(error))
        except InvalidPDFError as error:
            st.error(str(error))
        else:
            st.session_state["resume_text"] = extracted_text
            st.success("Resume text extracted.")
            st.text_area(
                "Extracted resume text",
                value=extracted_text,
                height=260,
                disabled=True,
            )
    else:
        resume_text = st.session_state.get("resume_text", "")
        if resume_text:
            st.caption(
                "Extracted resume text remains in this active session; "
                "the original PDF is not retained."
            )
            if st.button("Clear extracted resume text"):
                st.session_state.pop("resume_text", None)
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
    st.caption(
        "Uploaded files are not written to disk. Extracted text and the job description "
        "remain only in this active Streamlit session."
    )


def render_categorized_skills(
    title: str,
    skills_by_category: dict[str, list[str]],
    empty_message: str,
) -> None:
    st.subheader(title)
    populated_categories = [
        (category, skills)
        for category, skills in skills_by_category.items()
        if skills
    ]
    if not populated_categories:
        st.caption(empty_message)
        return
    for category, skills in populated_categories:
        with st.expander(category, expanded=True):
            st.write(", ".join(skills))


def render_job_match_results(analysis: JobMatchAnalysis) -> None:
    score_columns = st.columns(3)
    with score_columns[0]:
        render_metric(
            "Heuristic match estimate",
            (
                f"{analysis.overall_match_score:.1f}%"
                if analysis.overall_match_score is not None
                else "Unavailable"
            ),
        )
    with score_columns[1]:
        render_metric(
            "Skill coverage",
            (
                f"{analysis.skill_coverage_score:.1f}%"
                if analysis.skill_coverage_score is not None
                else "Unavailable"
            ),
        )
    with score_columns[2]:
        render_metric(
            "Text similarity",
            (
                f"{analysis.text_similarity_score:.1f}%"
                if analysis.text_similarity_score is not None
                else "Unavailable"
            ),
        )
    st.caption(
        "Heuristic compatibility estimate, not a probability of getting hired. "
        f"{analysis.text_similarity_method} measures lexical overlap, not meaning."
    )
    if analysis.score_unavailable_reason:
        st.info(analysis.score_unavailable_reason)

    if analysis.skill_coverage_score is not None:
        st.progress(analysis.skill_coverage_score / 100)
        st.caption("Skill coverage: matching detected requirements / detected requirements.")
    if analysis.text_similarity_score is not None:
        st.progress(analysis.text_similarity_score / 100)
        st.caption("Text similarity: TF-IDF cosine similarity of the two text inputs.")

    missing_skill_sources = []
    if not analysis.candidate_skills:
        missing_skill_sources.append("the resume")
    if not analysis.required_skills:
        missing_skill_sources.append("the job description")
    if missing_skill_sources:
        st.warning(
            "No taxonomy skills were detected in "
            + " and ".join(missing_skill_sources)
            + ". Review the text or taxonomy before relying on skill-based results."
        )

    candidate_column, required_column = st.columns(2)
    with candidate_column:
        render_categorized_skills(
            "Candidate skills",
            analysis.candidate_skills_by_category,
            "No taxonomy skills were detected in the resume.",
        )
    with required_column:
        render_categorized_skills(
            "Required job skills",
            analysis.required_skills_by_category,
            "No taxonomy skills were detected in the job description.",
        )

    matching_column, missing_column, additional_column = st.columns(3)
    with matching_column:
        render_categorized_skills(
            "Matching skills",
            analysis.matching_skills_by_category,
            "No required skills were detected in the resume.",
        )
    with missing_column:
        render_categorized_skills(
            "Missing skills",
            analysis.missing_skills_by_category,
            "No detected job skills are missing from the resume.",
        )
    with additional_column:
        render_categorized_skills(
            "Additional candidate skills",
            analysis.additional_skills_by_category,
            "No additional taxonomy skills were detected.",
        )

    strengths_column, improvement_column = st.columns(2)
    with strengths_column:
        st.subheader("Strengths")
        if analysis.strengths:
            for strength in analysis.strengths:
                st.markdown(f"- {strength}")
        else:
            st.caption("No explicit matching or additional skills were detected.")
    with improvement_column:
        st.subheader("Areas for improvement")
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
    current_inputs = (resume_text, job_description)
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
            except ImportError as error:
                st.error(f"Scikit-learn TF-IDF is unavailable: {error}")
            else:
                st.session_state["job_match_analysis"] = analysis
                st.session_state["job_match_inputs"] = current_inputs

    if clear_stale_job_match_analysis():
        st.info("Inputs changed. Click Analyze Job Match to calculate updated results.")
    analysis = st.session_state.get("job_match_analysis")
    if analysis is not None:
        render_job_match_results(analysis)


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
    title = f"#{rank} — {recommendation.title}"
    with st.expander(f"{title} ({recommendation.suitability_score:.1f}% alignment)", expanded=(rank <= 3)):
        st.write(recommendation.description)
        st.progress(recommendation.suitability_score / 100)
        metric_cols = st.columns(3)
        with metric_cols[0]:
            render_metric("Heuristic suitability", f"{recommendation.suitability_score:.1f}%")
        with metric_cols[1]:
            render_metric("Core skill coverage", f"{recommendation.core_coverage_score:.1f}%")
        with metric_cols[2]:
            render_metric(
                "Supporting skill coverage",
                f"{recommendation.supporting_coverage_score:.1f}%",
            )
        st.markdown(f"**Why recommended:** {recommendation.explanation}")

        matched_all = (
            recommendation.matching_core_skills
            + recommendation.matching_supporting_skills
        )
        col_matched, col_missing, col_next = st.columns(3)
        with col_matched:
            st.markdown("**Matching skills**")
            st.write(", ".join(matched_all) if matched_all else "None detected")
        with col_missing:
            st.markdown("**Missing core skills**")
            st.write(
                ", ".join(recommendation.missing_core_skills)
                if recommendation.missing_core_skills
                else "None"
            )
        with col_next:
            st.markdown("**Suggested skills to learn next**")
            st.write(
                ", ".join(recommendation.next_skills_to_learn[:5])
                if recommendation.next_skills_to_learn
                else "All tracked skills present"
            )


def render_career_recommendations() -> None:
    render_heading(
        "Explainable role guidance",
        "Career recommendations",
        "Ranked career paths based on taxonomy skills detected in your uploaded resume.",
    )
    resume_text = st.session_state.get("resume_text", "")
    if not isinstance(resume_text, str) or not resume_text.strip():
        st.warning("Resume is missing. Upload a text-based PDF in Resume & Job to generate career recommendations.")
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
        st.caption(f"Detected resume skills ({len(detected_skills)}): {', '.join(detected_skills)}")
    else:
        st.warning(
            "No curated taxonomy skills were detected in the resume. "
            "All roles show 0.0% alignment; review extracted text or expand skills."
        )

    st.caption(
        "Suitability scores are heuristic skill-alignment estimates (70% core skills, "
        "30% supporting skills), not a probability of getting hired."
    )
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
        st.caption(
            "Core skill" if step.is_core_skill else "Supporting skill"
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

    career_title = st.selectbox(
        "Career path",
        career_titles,
        key=_ROADMAP_CAREER_WIDGET_KEY,
        on_change=clear_learning_roadmap_reset_confirmation,
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

    st.subheader(career.title)
    st.write(career.description)
    render_metric(
        "Overall completion",
        f"{roadmap.progress_percentage:.1f}%",
    )
    st.progress(roadmap.progress_percentage / 100)
    st.caption(
        f"{roadmap.completed_count} of {roadmap.total_count} skills complete."
    )
    st.write(
        "**Completed skills:** "
        + (", ".join(roadmap.completed_skills) or "None")
    )
    st.write(
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
        st.subheader(stage_name)
        if not stage_steps:
            st.caption(f"No {stage_name.lower()} skills remain in this roadmap.")
            continue
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
