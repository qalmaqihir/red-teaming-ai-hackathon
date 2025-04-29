

import streamlit as st
from components import home, art_module, counterfit_module, pyrit_module, results, model_scan_module,most_common_attacks_module, garak

# Configure the main page
st.set_page_config(page_title="AI Red Teaming Platform", layout="wide")

# Sidebar for navigation
st.sidebar.title("AI Red Teaming Navigation")
page = st.sidebar.radio(
    "Go to", ["Home", "LLM vulnerability scanner - garak","Most Common LLM Attacks", "Adversarial Robustness Toolbox", "Counterfit", "PyRIT", "Model Scan", "Results & Reports"]
)


# Main section switching
if page == "Home":
    home.display_home()
elif page == "Most Common LLM Attacks":
    most_common_attacks_module.display_most_common_llm_attacks_page()
elif page == "LLM vulnerability scanner - garak":
    garak.display_garak()
elif page == "Adversarial Robustness Toolbox":
    art_module.display_art_section()
elif page == "Counterfit":
    counterfit_module.display_counterfit_section()
elif page == "PyRIT":
    pyrit_module.display_pyrit_section()
elif page == "Model Scan":
    model_scan_module.display_model_scan_section()
elif page == "Results & Reports":
    results.display_results_section()
