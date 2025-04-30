### Version 2

import streamlit as st
import logging
import os
import uuid
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List
import requests
import tempfile
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modelscan_red_teaming.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Supported model file extensions
SUPPORTED_EXTENSIONS = ["pkl", "joblib", "h5", "pt", "pth", "tf", "onnx", "safetensors"]

# Check if modelscan CLI is available
def check_modelscan_availability() -> bool:
    try:
        result = subprocess.run(["modelscan", "--help"], capture_output=True, text=True, check=True)
        logger.info("ModelScan CLI is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"ModelScan CLI is not available: {str(e)}")
        return False

MODELSCAN_AVAILABLE = check_modelscan_availability()

def run_modelscan_scan(file_path: str, scan_config: Dict) -> Dict:
    """
    Run modelscan scan command and return the results.
    """
    try:
        # Construct modelscan command
        cmd = ["modelscan", "-p", file_path]
        
        # Add configuration based on scan type
        scan_type = scan_config.get("scan_type", "Basic Vulnerability Scan")
        if scan_type == "Deep Inspection":
            cmd.extend(["--deep"])
        elif scan_type == "Custom Scan" and scan_config.get("checks"):
            # Map custom checks to modelscan options (adjust based on actual CLI options)
            for check in scan_config.get("checks", []):
                if check == "Pickle Vulnerability":
                    cmd.extend(["--check", "pickle"])
                elif check == "Joblib Security":
                    cmd.extend(["--check", "joblib"])
                # Add more mappings as needed
        
        logger.info(f"Executing ModelScan command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=scan_config.get("timeout", 300)
        )
        
        # Parse output (assuming JSON or structured text; adjust based on actual output)
        try:
            scan_report = json.loads(result.stdout)
        except json.JSONDecodeError:
            # Fallback to parsing text output if not JSON
            scan_report = {"raw_output": result.stdout, "errors": result.stderr}
        
        scan_report["success"] = True
        logger.info("ModelScan scan completed successfully")
        return scan_report
    
    except subprocess.TimeoutExpired as e:
        logger.error(f"ModelScan scan timed out: {str(e)}")
        return {"success": False, "error": "Scan timed out", "details": str(e)}
    except subprocess.CalledProcessError as e:
        logger.error(f"ModelScan scan failed: {str(e)}, stderr: {e.stderr}")
        return {"success": False, "error": "Scan failed", "details": e.stderr}
    except Exception as e:
        logger.error(f"Unexpected error during ModelScan scan: {str(e)}")
        return {"success": False, "error": "Unexpected error", "details": str(e)}

def display_model_scan_section():
    st.title("ModelScan: Model Serialization Attack Detection")
    logger.info("ModelScan section accessed")

    # Warn if ModelScan is not available
    if not MODELSCAN_AVAILABLE:
        st.error(
            "ModelScan CLI is not installed or not found in PATH. "
            "Please install it with: `pip install modelscan` and ensure `modelscan` is executable. "
            "Check the documentation: https://github.com/protectai/modelscan. "
            "If you see TensorFlow/CUDA errors, try creating a clean virtual environment."
        )
        logger.warning("ModelScan functionality disabled due to CLI unavailability")
        return

    # Information about Model Serialization Attacks
    if st.button("Learn about Model Serialization Attacks"):
        with st.expander("Model Serialization Attacks Explained", expanded=True):
            st.markdown("""
            # Model Serialization Attacks Explained
            Machine Learning (ML) models are shared publicly, within teams, and across organizations. The rise of Foundation Models has increased the use of public ML models for further training or fine-tuning. ML models are critical for decision-making and mission-critical applications, yet they are not scanned with the same rigor as other files, such as PDFs.

            **ModelScan**, an open-source project by Protect AI, scans models to detect unsafe code. It supports multiple formats, including H5, Pickle, SavedModel, PyTorch, TensorFlow, Keras, Sklearn, XGBoost, ONNX, and SafeTensors, with more formats planned.

            ## Why Scan Models?
            Models are serialized (saved to disk) for distribution. A **Model Serialization Attack** embeds malicious code in a model during serialization, which executes upon loading (e.g., `torch.load(PATH)` in PyTorch). Such attacks can enable:
            - **Credential Theft**: Stealing cloud credentials for data access.
            - **Data Theft**: Exfiltrating input data sent to the model.
            - **Data Poisoning**: Altering output data post-processing.
            - **Model Poisoning**: Manipulating model outputs.
            - **Malware Execution**: Running malicious code embedded in the model.
            - **Version Tampering**: Using outdated or malicious model versions.

            Scanning models ensures safety and integrity before deployment.

            ##### Visit the [ModelScan GitHub Page](https://github.com/protectai/modelscan)
            """)
            logger.info("Model Serialization Attacks information expanded")

    st.subheader("ModelScan Analysis")

    # API Configuration (placeholder for future API support)
    st.write("### 1. API Configuration (Optional)")
    use_api = st.checkbox("Use External API (Not Available)", disabled=True, help="API support is not currently available. Uses local ModelScan CLI.")
    api_key = None
    api_endpoint = None
    if use_api:
        st.warning("API scanning is not supported in the current ModelScan version. Using local CLI.")
        logger.warning("API scanning attempted but not supported")

    # Scan Type Selection
    st.write("### 2. Choose Scan Type")
    scan_types = [
        "Basic Vulnerability Scan",
        "Deep Inspection",
        "Compliance Check",
        "Custom Scan",
        "Dependency Vulnerability Scan",
        "Model Integrity Check",
        "Advanced Malware Detection",
        "Model Versioning Check"
    ]
    scan_type = st.selectbox("Scan Type", scan_types, help="Select the type of scan to perform")
    logger.info(f"Scan type selected: {scan_type}")

    # Scan Configuration
    st.write("### 3. Configure Scan Parameters")
    scan_config = {"scan_type": scan_type}
    
    if scan_type == "Basic Vulnerability Scan":
        st.write("Quick scan for common serialization vulnerabilities")
        scan_config["depth"] = st.slider("Scan Depth", 1, 5, 3, help="Higher depth increases thoroughness but extends scan time")
        scan_config["check_types"] = st.multiselect(
            "Vulnerability Types",
            ["pickle", "joblib", "yaml", "json"],
            default=["pickle", "joblib"],
            help="Select types of vulnerabilities to check"
        )

    elif scan_type == "Deep Inspection":
        st.write("In-depth analysis of model structure and vulnerabilities")
        scan_config["structure_analysis"] = st.checkbox("Include Model Structure Analysis", value=True)
        scan_config["dependency_check"] = st.checkbox("Check for Vulnerable Dependencies", value=True)
        scan_config["timeout"] = st.number_input("Scan Timeout (seconds)", min_value=30, value=300, step=10)
        scan_config["check_types"] = st.multiselect(
            "Vulnerability Types",
            ["pickle", "joblib", "yaml", "json", "safetensors"],
            default=["pickle", "joblib", "safetensors"],
            help="Select types of vulnerabilities to check"
        )

    elif scan_type == "Compliance Check":
        st.write("Verify model against security standards or regulations")
        scan_config["standards"] = st.multiselect(
            "Compliance Standards",
            ["NIST", "ISO 27001", "GDPR", "HIPAA", "OWASP", "CIS"],
            help="Select applicable standards"
        )
        scan_config["generate_report"] = st.checkbox("Generate Compliance Report", value=True)
        if not scan_config["standards"]:
            st.warning("Please select at least one compliance standard")
            logger.warning("No compliance standards selected")

    elif scan_type == "Custom Scan":
        st.write("Configure a custom scan with specific checks")
        scan_config["checks"] = st.multiselect(
            "Custom Checks",
            [
                "Pickle Vulnerability",
                "Joblib Security",
                "YAML Deserialization",
                "JSON Insecure Deserialization",
                "Dependency Confusion",
                "SafeTensors Validation",
                "ONNX Structure Check",
                "TensorFlow Graph Security"
            ],
            help="Select specific vulnerability checks"
        )
        scan_config["depth"] = st.slider("Custom Scan Depth", 1, 10, 5)
        if not scan_config["checks"]:
            st.warning("Please select at least one custom check")
            logger.warning("No custom checks selected")

    elif scan_type == "Dependency Vulnerability Scan":
        st.write("Scan model dependencies for known vulnerabilities")
        scan_config["dependency_db"] = st.selectbox(
            "Dependency Database",
            ["NVD", "OSV", "Snyk", "GitHub Advisory"],
            help="Select vulnerability database"
        )
        scan_config["include_dev_deps"] = st.checkbox("Include Development Dependencies", value=False)
        scan_config["timeout"] = st.number_input("Scan Timeout (seconds)", min_value=30, value=180)

    elif scan_type == "Model Integrity Check":
        st.write("Verify model integrity and authenticity")
        scan_config["hash_verification"] = st.checkbox("Verify Model Hash", value=True)
        scan_config["signature_check"] = st.checkbox("Check Digital Signature", value=False)
        scan_config["provenance_check"] = st.checkbox("Verify Model Provenance", value=False)
        scan_config["expected_hash"] = st.text_input("Expected Model Hash (Optional)", help="Provide expected hash for verification")

    elif scan_type == "Advanced Malware Detection":
        st.write("Advanced scan for embedded malware or malicious code")
        scan_config["malware_db"] = st.selectbox(
            "Malware Database",
            ["ClamAV", "VirusTotal", "Custom"],
            help="Select malware database for scanning"
        )
        scan_config["deep_scan"] = st.checkbox("Perform Deep Malware Scan", value=False)
        scan_config["timeout"] = st.number_input("Scan Timeout (seconds)", min_value=30, value=300)

    elif scan_type == "Model Versioning Check":
        st.write("Check for outdated or tampered model versions")
        scan_config["version_check"] = st.checkbox("Verify Model Version", value=True)
        scan_config["version_db"] = st.selectbox(
            "Version Database",
            ["ModelHub", "Custom Registry"],
            help="Select model version database"
        )
        scan_config["expected_version"] = st.text_input("Expected Model Version (Optional)", help="Provide expected model version")

    # Model Upload
    st.write("### 4. Upload Model File")
    model_file = st.file_uploader(
        "Upload Model File",
        type=SUPPORTED_EXTENSIONS,
        help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
    )

    # Run Scan
    if st.button("Run ModelScan Analysis"):
        if not model_file:
            st.error("Please upload a model file for analysis")
            logger.error("No model file uploaded")
            return

        if scan_type == "Custom Scan" and not scan_config.get("checks"):
            st.error("Please select at least one custom check for Custom Scan")
            logger.error("No custom checks selected for Custom Scan")
            return

        if scan_type == "Compliance Check" and not scan_config.get("standards"):
            st.error("Please select at least one compliance standard for Compliance Check")
            logger.error("No compliance standards selected for Compliance Check")
            return

        st.info(f"Running {scan_type} using ModelScan...")
        logger.info(f"Starting {scan_type} for uploaded model: {model_file.name}")

        # Save uploaded file to temporary location
        try:
            file_ext = os.path.splitext(model_file.name)[1].lower()
            if file_ext.lstrip('.') not in SUPPORTED_EXTENSIONS:
                st.error(f"Unsupported file extension: {file_ext}. Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
                logger.error(f"Unsupported file extension: {file_ext}")
                return

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(model_file.getbuffer())
                tmp_file_path = tmp_file.name
            logger.info(f"Model file saved to temporary path: {tmp_file_path}")
        except Exception as e:
            st.error(f"Failed to save model file: {str(e)}")
            logger.error(f"File save error: {str(e)}")
            return

        # Perform the scan
        start_time = datetime.now()
        scan_report = {}
        scan_id = str(uuid.uuid4())
        
        try:
            # Run ModelScan scan
            scan_report = run_modelscan_scan(tmp_file_path, scan_config)

            if not scan_report.get("success"):
                st.error(f"Scan failed: {scan_report.get('error')}. Details: {scan_report.get('details')}")
                logger.error(f"Scan failed: {scan_report.get('error')}")
                return

            # Enhance report with additional metadata
            scan_report["scan_id"] = scan_id
            scan_report["scan_type"] = scan_type
            scan_report["timestamp"] = datetime.now().isoformat()
            scan_report["config"] = scan_config
            scan_report["model_file"] = model_file.name

            # Display results
            st.success("Scan completed successfully!")
            st.subheader("ModelScan Report")
            st.json(scan_report)

            # Detailed analysis based on scan type
            if scan_type == "Basic Vulnerability Scan":
                st.subheader("Vulnerability Summary")
                vulnerabilities = scan_report.get("vulnerabilities", [])
                if vulnerabilities:
                    st.write(f"Found {len(vulnerabilities)} potential vulnerabilities")
                    df = pd.DataFrame(vulnerabilities)
                    st.dataframe(df)
                else:
                    st.write("No vulnerabilities detected")

            elif scan_type == "Deep Inspection":
                st.subheader("Model Structure Analysis")
                structure = scan_report.get("structure", {})
                if structure:
                    st.write("Model architecture details:")
                    st.json(structure)
                else:
                    st.write("No structural issues detected")
                if scan_config.get("dependency_check"):
                    st.subheader("Dependency Check")
                    deps = scan_report.get("dependencies", [])
                    if deps:
                        st.write(f"Found vulnerabilities in {len(deps)} dependencies")
                        df = pd.DataFrame(deps)
                        st.dataframe(df)
                    else:
                        st.write("No vulnerable dependencies detected")

            elif scan_type == "Compliance Check" and scan_config.get("generate_report"):
                st.subheader("Compliance Report")
                compliance_results = scan_report.get("compliance", {})
                if compliance_results:
                    st.write(f"Compliance status for selected standards: {', '.join(scan_config['standards'])}")
                    st.json(compliance_results)
                else:
                    st.write("No compliance issues detected")

            elif scan_type == "Custom Scan":
                st.subheader("Custom Scan Results")
                results = scan_report.get("custom_checks", {})
                if results:
                    df = pd.DataFrame([
                        {"Check": check, "Status": result}
                        for check, result in results.items()
                    ])
                    st.dataframe(df)
                else:
                    st.write("No issues found in custom checks")

            elif scan_type == "Dependency Vulnerability Scan":
                st.subheader("Dependency Vulnerabilities")
                deps = scan_report.get("dependencies", [])
                if deps:
                    st.write(f"Found vulnerabilities in {len(deps)} dependencies")
                    df = pd.DataFrame(deps)
                    st.dataframe(df)
                else:
                    st.write("No vulnerable dependencies detected")

            elif scan_type == "Model Integrity Check":
                st.subheader("Integrity Verification")
                integrity = scan_report.get("integrity", {})
                if integrity.get("hash_verified"):
                    st.write("Model hash verification: Passed")
                else:
                    st.warning("Model hash verification: Failed")
                if scan_config.get("signature_check") and integrity.get("signature_verified"):
                    st.write("Digital signature: Verified")
                elif scan_config.get("signature_check"):
                    st.warning("Digital signature: Not verified")
                if scan_config.get("provenance_check") and integrity.get("provenance_verified"):
                    st.write("Provenance: Verified")
                elif scan_config.get("provenance_check"):
                    st.warning("Provenance: Not verified")

            elif scan_type == "Advanced Malware Detection":
                st.subheader("Malware Scan Results")
                malware = scan_report.get("malware", [])
                if malware:
                    st.write(f"Found {len(malware)} potential malware instances")
                    df = pd.DataFrame(malware)
                    st.dataframe(df)
                else:
                    st.write("No malware detected")

            elif scan_type == "Model Versioning Check":
                st.subheader("Model Versioning Results")
                versioning = scan_report.get("versioning", {})
                if versioning.get("version_verified"):
                    st.write(f"Model version verified: {versioning.get('version')}")
                else:
                    st.warning("Model version verification failed")
                    if scan_config.get("expected_version"):
                        st.write(f"Expected version: {scan_config['expected_version']}")

            # Downloadable report
            report_json = json.dumps(scan_report, indent=2)
            st.download_button(
                label="Download Scan Report",
                data=report_json,
                file_name=f"modelscan_report_{scan_id}.json",
                mime="application/json"
            )

            # Execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            st.write(f"**Scan Execution Time**: {execution_time:.2f} seconds")
            logger.info(f"Scan completed in {execution_time:.2f} seconds")

        except Exception as e:
            st.error(
                f"Scan failed: {str(e)}. "
                "Please ensure ModelScan is installed correctly (`pip install modelscan --upgrade`) "
                "and check the documentation: https://github.com/protectai/modelscan. "
                "If TensorFlow/CUDA errors persist, try a clean virtual environment."
            )
            logger.error(f"Scan error: {str(e)}")

        finally:
            # Clean up temporary file
            try:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    logger.info(f"Temporary file deleted: {tmp_file_path}")
            except Exception as e:
                logger.error(f"Failed to delete temporary file: {str(e)}")

    # Security warning
    st.warning(
        "Ensure you have permission to scan the uploaded model. Handle model files and scan results securely to prevent unauthorized access."
    )