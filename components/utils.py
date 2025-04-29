import streamlit as st

# Helper functions, constants, and styling settings
def set_styling():
    st.markdown(
        """
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            .stSidebar {
                background-color: #f4f4f4;
            }
        </style>
        """, unsafe_allow_html=True
    )
