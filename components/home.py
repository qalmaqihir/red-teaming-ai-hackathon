import streamlit as st
from components import home, art_module, counterfit_module, pyrit_module, results, most_common_attacks_module

def display_home():
    st.title("Welcome to the AI Red Teaming Platform")
    st.markdown(
        """
        This platform allows you to test your AI models for vulnerabilities using state-of-the-art tools:
        - **Most Common LLM Attacks**
        - **LLM vulnerability scanner**  
        - **Adversarial Robustness Toolbox (ART)** for adversarial attack simulations.
        - **Counterfit** for attack surface exploration.
        - **PyRIT** for risk identification and compliance evaluation.
        - **Human Red Teaming AI/LLMs**

        Navigate through the sidebar to start testing and evaluating your AI models.
        """
    )
    

    # Quick Start Section
    st.subheader("Quick Start")
    col1, col2, col3, col4, col5, col6, col7,col8  = st.columns(8)
    with col1:
        st.button("Run Most Common LLM Attacks")
    with col2:
        st.button("Run Adversarial Robustness Toolbox")
     
    with col3:
        st.button("Run Garak")

    with col4:
        st.button("Run PyRIT")

    with col5:
        st.button("Run Model Scan") 

    with col6:
        st.button("Run Counterfit")
        
    with col7:
        st.button("Run Giskard")
        
    with col8:
        st.button("Get Results & Reports")

    # Display recent activity
    st.subheader("Recent Activity")
    st.markdown("No recent activities to display.")


