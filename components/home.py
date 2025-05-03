import streamlit as st
from pathlib import Path
import json

def navigate_to(page):
    """Set session state to navigate to the specified page."""
    st.session_state.page = page

def display_home():
    st.title("Welcome to the AI Red Teaming Platform")
    st.markdown("""
    This platform provides a comprehensive suite of tools for testing the security and robustness of AI models. Navigate through the sidebar or use the Quick Start buttons below to access red teaming tools and view results.

    ### Available Tools
    - **Garak Scanner**: Probes large language models (LLMs) for vulnerabilities using the Garak framework.
    - **Adversarial Robustness Toolbox (ART)**: Performs adversarial attacks (e.g., FGM, PGD) on machine learning models, focusing on image classification.
    - **Counterfit**: Simulates adversarial attacks on AI models, supporting various attack types and model formats.
    - **PyRIT**: Conducts red teaming for LLMs and ML models using the Python Risk Identification Toolkit.
    - **Giskard**: Scans LLMs and ML models for vulnerabilities, including RAG-based testing.
    - **ModelScan**: Analyzes model files for security issues, supporting multiple frameworks.

    ### Results & Reports
    Access the "Results & Reports" section to view, download, or share generated reports and logs from all tools. Reports are stored in the `./reports` directory and include JSON reports, logs, and additional artifacts (e.g., adversarial images).

    **Note**: Ensure you have permission to test models and handle results securely. API keys for OpenAI and Anthropic are required for some tools (configured via `.env`).
    """)

    # Quick Start Section
    st.subheader("Quick Start")
    col1, col2, col3, col4, col5, col6, col7,col8 = st.columns(8)
    with col1:
        st.button("Run Garak", on_click=navigate_to, args=("Garak Scanner",))
    with col2:
        st.button("Run ART", on_click=navigate_to, args=("Adversarial Robustness Toolbox",))
    with col3:
        st.button("Run Counterfit", on_click=navigate_to, args=("Counterfit",))
    with col4:
        st.button("Run PyRIT", on_click=navigate_to, args=("PyRIT",))
    with col5:
        st.button("Run Giskard", on_click=navigate_to, args=("Giskard",))
    with col6:
        st.button("Run ModelScan", on_click=navigate_to, args=("ModelScan",))
    with col7:
        st.button("Run most common LLM attaacks", on_click=navigate_to, args=("Most Common LLM Attacks",))
    with col8:
        st.button("View Results", on_click=navigate_to, args=("Results & Reports",))

    # Recent Activity Section
    st.subheader("Recent Activity")
    reports_dir = Path("./reports")
    if not reports_dir.exists():
        st.markdown("No recent activities to display.")
        return

    reports = list(reports_dir.glob("*.json"))
    if not reports:
        st.markdown("No recent reports found.")
        return

    st.markdown("**Recent Reports**")
    for report in sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
        try:
            with open(report, "r") as f:
                report_data = json.load(f)
            scan_id = report_data.get("scan_id", report.name)
            tool = report.name.split("_")[0].capitalize()
            timestamp = report_data.get("timestamp", "Unknown")
            st.download_button(
                label=f"{tool}: {scan_id} ({timestamp})",
                data=open(report, "r").read(),
                file_name=report.name,
                mime="application/json",
                key=f"recent_{report.name}"
            )
        except Exception as e:
            st.warning(f"Error reading {report.name}: {str(e)}")

# if __name__ == "__main__":
#     display_home()