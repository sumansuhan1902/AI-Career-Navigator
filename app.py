"""Streamlit entry point for the AI Career Navigator prototype."""

from __future__ import annotations

import streamlit as st

from modules.resume_parser import (
    EmptyPDFError,
    InvalidPDFError,
    PDFPageLimitError,
    PDFTooLargeError,
    ScannedPDFError,
    extract_text_from_pdf,
)
from modules.job_matcher import (
    InvalidJobMatchingInputError,
    JobMatchAnalysis,
    match_job,
)
from modules.skill_extractor import SkillExtractionError


st.set_page_config(
    page_title="AI Career Navigator",
    page_icon=":material/explore:",
    layout="wide",
    initial_sidebar_state="expanded",
)


PAGE_NAMES = ("Home", "Resume & Job", "Skill Analysis", "Career Recommendations")


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
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<div class="page-heading">{title}</div>'
        f'<p class="page-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
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
        '<div class="status-panel"><strong>Skill comparison uses curated keyword matching</strong>'
        '<span>Add a resume and target job description in Resume &amp; Job to '
        'inspect detected and missing skills. '
        'Career recommendations are not implemented.</span></div>',
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
        f'<strong>Not implemented</strong><span>{details}</span></div>',
        unsafe_allow_html=True,
    )


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
    else:
        render_pending_page(
            "Career recommendations",
            "Explore role directions grounded in your experience and interests.",
            "Career recommendations are not implemented. No roles are being generated or ranked.",
        )


if __name__ == "__main__":
    main()
