import streamlit as st
from components.home import display_home
from components.giskard_module import display_giskard_section
from components.pyrit_module import display_red_teaming_sections
from components.art_module import display_art_section
from components.modelscan_module import display_model_scan_section
from components.garak_module import display_garak
from components.counterfit_module import display_counterfit_section
from components.results_module import display_results_section
from components.most_common_attacks_module import display_most_common_llm_attacks_section
# Configure the main page
st.set_page_config(page_title="AI Red Teaming Platform", layout="wide")

# Sidebar for navigation
st.sidebar.title("AI Red Teaming Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Garak Scanner",
        "Most Common LLM Attacks",
        "Adversarial Robustness Toolbox",
        "Counterfit",
        "PyRIT",
        "Giskard",
        "ModelScan",
        "Results & Reports"
    ]
)

# Main section switching
if page == "Home":
    display_home()
elif page == "Garak Scanner":
    display_garak()
elif page == "Adversarial Robustness Toolbox":
    display_art_section()
elif page == "Counterfit":
    display_counterfit_section()
elif page == "Most Common LLM Attacks":
    display_most_common_llm_attacks_section()
elif page == "PyRIT":
    display_red_teaming_sections()
elif page == "Giskard":
    display_giskard_section()
elif page == "ModelScan":
    display_model_scan_section()
elif page == "Results & Reports":
    display_results_section()