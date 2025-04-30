# import streamlit as st

# def display_results_section():
#     st.title("Results & Reports")

#     st.subheader("Previous Tests")
#     st.markdown("No results to display. Please run a test to see the results here.")

#     # Placeholder for displaying stored results
#     # Use Streamlit tables or charts to show results
#     st.info("Results and reports will be shown here after a test is run.")

#     # Option to download report
#     if st.button("Download Report"):
#         st.info("Downloading report...")


### Version 2
import streamlit as st
import os
import pandas as pd
import json
from datetime import datetime
import humanize

REPORTS_DIR = "./reports"

def get_report_details():
    """Get details of all reports in the reports directory."""
    if not os.path.exists(REPORTS_DIR):
        return []

    reports = []
    for filename in os.listdir(REPORTS_DIR):
        file_path = os.path.join(REPORTS_DIR, filename)
        if os.path.isfile(file_path):
            stats = os.stat(file_path)
            reports.append({
                "Filename": filename,
                "Created": datetime.fromtimestamp(stats.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "Size": humanize.naturalsize(stats.st_size),
                "Path": file_path
            })
    return reports

def display_report_content(file_path):
    """Display the content of a report file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".json":
            with open(file_path, 'r') as f:
                content = json.load(f)
            st.json(content)
        elif ext in [".log", ".txt"]:
            with open(file_path, 'r') as f:
                content = f.read()
            st.text_area("Report Content", content, height=300)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            st.dataframe(df)
        else:
            st.warning(f"Unsupported file type: {ext}. Download to view.")
    except Exception as e:
        st.error(f"Failed to display report content: {str(e)}")

def display_results_section():
    st.title("Results & Reports")

    st.subheader("Previous Tests")
    reports = get_report_details()

    if not reports:
        st.markdown("No results to display. Please run a test to see the results here.")
        st.info("Results and reports will be shown here after a test is run.")
    else:
        # Display reports in a table
        df = pd.DataFrame(reports)
        st.dataframe(df[["Filename", "Created", "Size"]], use_container_width=True)

        # Allow selection of a report
        selected_report = st.selectbox("Select a report to view or download", 
                                      options=[r["Filename"] for r in reports],
                                      format_func=lambda x: x)

        if selected_report:
            report = next(r for r in reports if r["Filename"] == selected_report)
            file_path = report["Path"]

            # Display report content
            st.subheader(f"Content of {selected_report}")
            display_report_content(file_path)

            # Download button
            with open(file_path, 'rb') as f:
                st.download_button(
                    label=f"Download {selected_report}",
                    data=f,
                    file_name=selected_report,
                    mime="application/octet-stream"
                )

            # Share option (display file path)
            st.subheader("Share Report")
            st.write(f"File path: `{file_path}`")
            st.info("Copy the file path to share the report manually or download and share the file.")
            