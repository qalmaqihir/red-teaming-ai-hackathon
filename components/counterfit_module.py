### Version 2

import streamlit as st
import logging
import os
import uuid
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import requests
import tempfile
from PIL import Image
import io
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('counterfit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Supported formats and constraints
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png"]
SUPPORTED_MODEL_FORMATS = ["h5", "pkl"]
MAX_IMAGE_SIZE_MB = 5
MAX_MODEL_SIZE_MB = 10
RECOMMENDED_IMAGE_SIZE = (224, 224)

def validate_url(url: str) -> bool:
    """Validate if the URL is well-formed and reachable (placeholder)."""
    try:
        result = requests.head(url, timeout=5)
        return result.status_code == 200
    except Exception as e:
        logger.error(f"URL validation failed: {str(e)}")
        return False

def validate_image(file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> Optional[Image.Image]:
    """Validate and preprocess image input."""
    if not file:
        return None
    try:
        file_ext = os.path.splitext(file.name)[1].lower().lstrip('.')
        if file_ext not in SUPPORTED_IMAGE_FORMATS:
            st.error(f"Unsupported image format: {file_ext}. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}")
            logger.error(f"Unsupported image format: {file_ext}")
            return None
        if file.size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            st.error(f"Image too large: {file.size / (1024 * 1024):.2f}MB. Max: {MAX_IMAGE_SIZE_MB}MB")
            logger.error(f"Image size exceeds limit: {file.size}")
            return None
        img = Image.open(file)
        img = img.resize(RECOMMENDED_IMAGE_SIZE, Image.Resampling.LANCZOS)
        logger.info(f"Image validated and resized to {RECOMMENDED_IMAGE_SIZE}")
        return img
    except Exception as e:
        st.error(f"Failed to process image: {str(e)}")
        logger.error(f"Image processing error: {str(e)}")
        return None

def validate_model_file(file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> Optional[str]:
    """Validate and save model file to temporary path."""
    if not file:
        return None
    try:
        file_ext = os.path.splitext(file.name)[1].lower().lstrip('.')
        if file_ext not in SUPPORTED_MODEL_FORMATS:
            st.error(f"Unsupported model format: {file_ext}. Supported: {', '.join(SUPPORTED_MODEL_FORMATS)}")
            logger.error(f"Unsupported model format: {file_ext}")
            return None
        if file.size > MAX_MODEL_SIZE_MB * 1024 * 1024:
            st.error(f"Model file too large: {file.size / (1024 * 1024):.2f}MB. Max: {MAX_MODEL_SIZE_MB}MB")
            logger.error(f"Model file size exceeds limit: {file.size}")
            return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
            tmp_file.write(file.getbuffer())
            file_path = tmp_file.name
        logger.info(f"Model file saved to: {file_path}")
        return file_path
    except Exception as e:
        st.error(f"Failed to process model file: {str(e)}")
        logger.error(f"Model file processing error: {str(e)}")
        return None

def validate_text(text: str) -> bool:
    """Validate text input."""
    if not text.strip():
        st.error("Text input cannot be empty")
        logger.error("Empty text input")
        return False
    if len(text) > 1000:
        st.error("Text input too long (max 1000 characters)")
        logger.error(f"Text input length: {len(text)}")
        return False
    logger.info("Text input validated")
    return True

def validate_tabular_data(data: str) -> bool:
    """Validate CSV-format tabular data."""
    try:
        csv.reader(io.StringIO(data))
        logger.info("Tabular data validated")
        return True
    except Exception as e:
        st.error(f"Invalid CSV format: {str(e)}")
        logger.error(f"Tabular data validation error: {str(e)}")
        return False

def run_counterfit_attack(attack_type: str, model_type: str, model_input: Dict, attack_config: Dict, sample_input: Optional[any] = None) -> Dict:
    """Simulate running a Counterfit attack (placeholder for actual integration)."""
    try:
        model_url = model_input.get("url")
        model_file = model_input.get("file")
        logger.info(f"Running {attack_type} attack on {model_type} model (URL: {model_url or 'N/A'}, File: {model_file or 'N/A'})")
        scan_id = str(uuid.uuid4())
        result = {
            "scan_id": scan_id,
            "attack_type": attack_type,
            "model_type": model_type,
            "model_url": model_url or "N/A",
            "model_file": model_file or "N/A",
            "timestamp": datetime.now().isoformat(),
            "config": attack_config,
            "success_rate": 0.0,
            "vulnerabilities": [],
            "recommendations": [],
            "perturbed_output": None
        }

        # Placeholder for model loading (if file-based)
        if model_file:
            # Simulate loading model (e.g., keras.models.load_model for .h5, pickle.load for .pkl)
            logger.info(f"Simulating model loading from {model_file}")

        # Placeholder attack simulation
        if attack_type == "FGSM":
            result["success_rate"] = min(0.7 + attack_config.get("epsilon", 0.0) * 0.3, 1.0)
            result["vulnerabilities"] = [{"type": "Evasion", "severity": "Medium"}]
            result["recommendations"] = ["Add adversarial training", "Increase model robustness"]

        elif attack_type == "PGD":
            result["success_rate"] = min(0.8 + attack_config.get("iterations", 0) * 0.01, 1.0)
            result["vulnerabilities"] = [{"type": "Evasion", "severity": "High"}]
            result["recommendations"] = ["Use defensive distillation", "Regularize model weights"]

        elif attack_type == "DeepFool":
            result["success_rate"] = min(0.75 + attack_config.get("epsilon", 0.0) * 0.5, 1.0)
            result["vulnerabilities"] = [{"type": "Evasion", "severity": "Medium"}]
            result["recommendations"] = ["Implement gradient masking", "Monitor input anomalies"]

        elif attack_type == "CarliniWagner":
            result["success_rate"] = min(0.9 + attack_config.get("confidence", 0.0) * 0.1, 1.0)
            result["vulnerabilities"] = [{"type": "Evasion", "severity": "Critical"}]
            result["recommendations"] = ["Use certified defenses", "Limit query access"]

        elif attack_type == "Boundary Attack":
            result["success_rate"] = min(0.65 + attack_config.get("max_queries", 0) * 0.001, 1.0)
            result["vulnerabilities"] = [{"type": "Evasion", "severity": "Medium"}]
            result["recommendations"] = ["Restrict model access", "Add noise to inputs"]

        elif attack_type == "HopSkipJump":
            result["success_rate"] = min(0.7 + attack_config.get("max_queries", 0) * 0.001, 1.0)
            result["vulnerabilities"] = [{"type": "Evasion", "severity": "High"}]
            result["recommendations"] = ["Use ensemble models", "Monitor query patterns"]

        elif attack_type == "Model Stealing":
            result["success_rate"] = min(0.85 + attack_config.get("query_limit", 0) * 0.0001, 1.0)
            result["vulnerabilities"] = [{"type": "Model Extraction", "severity": "Critical"}]
            result["recommendations"] = ["Implement query rate limiting", "Use watermarking"]

        # Simulate perturbed output (placeholder)
        if model_type == "Image Classification" and sample_input:
            result["perturbed_output"] = "Perturbed image generated (placeholder)"

        logger.info(f"Counterfit {attack_type} attack completed successfully")
        return result
    except Exception as e:
        logger.error(f"Counterfit {attack_type} attack failed: {str(e)}")
        return {
            "scan_id": str(uuid.uuid4()),
            "attack_type": attack_type,
            "model_type": model_type,
            "model_url": model_input.get("url", "N/A"),
            "model_file": model_input.get("file", "N/A"),
            "timestamp": datetime.now().isoformat(),
            "config": attack_config,
            "success_rate": 0.0,
            "error": str(e)
        }

def display_counterfit_section():
    st.title("Counterfit: AI Security Testing")
    logger.info("Counterfit section accessed")

    if st.button("Learn about Counterfit"):
        with st.expander("Counterfit Explained", expanded=True):
            st.markdown("""
            **Counterfit** is an open-source tool by Microsoft for testing AI model security. It supports:

            - **Attack Types**: Evasion (FGSM, PGD), poisoning, model stealing, and more.
            - **Model Support**: Works with TensorFlow, PyTorch, scikit-learn, and custom APIs.
            - **Customization**: Create custom attacks and metrics.
            - **Reporting**: Detailed vulnerability reports.

            Use this tool to identify and mitigate vulnerabilities in your AI models.

            ##### Visit [GitHub](https://github.com/Azure/counterfit)
            """)
            logger.info("Counterfit information expanded")

    st.subheader("Counterfit Attack Configuration")

    # Model Configuration
    st.write("### Model Configuration")
    model_type = st.selectbox(
        "Model Type",
        ["Image Classification", "Text Classification", "Tabular Data"],
        help="Select the type of model to attack (e.g., image classifier, sentiment analyzer)"
    )
    model_input_type = st.radio(
        "Model Input Method",
        ["Upload Model File", "Provide API Endpoint"],
        help="Upload a local model file (.h5, .pkl) or provide the URL of a deployed model API"
    )
    model_input = {}
    if model_input_type == "Upload Model File":
        uploaded_file = st.file_uploader(
            "Upload Model File",
            type=SUPPORTED_MODEL_FORMATS,
            help=f"Supported formats: {', '.join(SUPPORTED_MODEL_FORMATS)}. Max size: {MAX_MODEL_SIZE_MB}MB"
        )
        model_input["file"] = validate_model_file(uploaded_file)
        if model_input["file"]:
            st.write(f"**Uploaded File**: {os.path.basename(model_input['file'])}")
    else:
        model_input["url"] = st.text_input(
            "Model API URL",
            help="Enter the model's API endpoint (e.g., http://localhost:5000/predict)"
        )
    logger.info(f"Model type: {model_type}, Input: {model_input}")

    # Attack Selection
    st.write("### Attack Selection")
    attack_types = [
        "FGSM",
        "PGD",
        "DeepFool",
        "CarliniWagner",
        "Boundary Attack",
        "HopSkipJump",
        "Model Stealing"
    ]
    attack_type = st.selectbox(
        "Attack Type",
        attack_types,
        help="Choose an attack. Evasion attacks (FGSM, PGD) perturb inputs; Model Stealing extracts model details."
    )
    logger.info(f"Attack type selected: {attack_type}")

    # Attack Parameters
    st.write("### Attack Parameters")
    attack_config = {}

    if attack_type == "FGSM":
        st.write("Fast Gradient Sign Method: Simple evasion attack")
        attack_config["epsilon"] = st.slider("Epsilon", 0.0, 1.0, 0.1, step=0.01, help="Perturbation size")
        attack_config["norm"] = st.selectbox("Norm", ["inf", "1", "2"], help="Norm for perturbation calculation")

    elif attack_type == "PGD":
        st.write("Projected Gradient Descent: Iterative evasion attack")
        attack_config["epsilon"] = st.slider("Epsilon", 0.0, 1.0, 0.1, step=0.01, help="Perturbation size")
        attack_config["iterations"] = st.slider("Iterations", 1, 100, 10, help="Number of iterations")
        attack_config["step_size"] = st.slider("Step Size", 0.01, 1.0, 0.1, step=0.01, help="Step size per iteration")

    elif attack_type == "DeepFool":
        st.write("DeepFool: Finds minimal perturbations for misclassification")
        attack_config["max_iterations"] = st.slider("Max Iterations", 10, 1000, 100, help="Maximum iterations")
        attack_config["epsilon"] = st.slider("Epsilon", 0.0001, 1.0, 0.02, step=0.0001, help="Overshoot parameter")

    elif attack_type == "CarliniWagner":
        st.write("Carlini & Wagner: Optimization-based evasion attack")
        attack_config["confidence"] = st.slider("Confidence", 0.0, 1.0, 0.5, step=0.1, help="Confidence of adversarial examples")
        attack_config["learning_rate"] = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001, help="Optimization learning rate")
        attack_config["binary_search_steps"] = st.slider("Binary Search Steps", 1, 20, 9, help="Number of binary search steps")

    elif attack_type == "Boundary Attack":
        st.write("Boundary Attack: Decision-based black-box attack")
        attack_config["max_queries"] = st.number_input("Max Queries", min_value=100, value=1000, step=100, help="Maximum queries to model")
        attack_config["epsilon"] = st.slider("Epsilon", 0.0, 1.0, 0.1, step=0.01, help="Perturbation size")

    elif attack_type == "HopSkipJump":
        st.write("HopSkipJump: Query-efficient black-box attack")
        attack_config["max_queries"] = st.number_input("Max Queries", min_value=100, value=1000, step=100, help="Maximum queries to model")
        attack_config["norm"] = st.selectbox("Norm", ["2", "inf"], help="Norm for perturbation calculation")

    elif attack_type == "Model Stealing":
        st.write("Model Stealing: Extracts model parameters or replicates model")
        attack_config["query_limit"] = st.number_input("Query Limit", min_value=100, value=1000, step=100, help="Maximum queries to model")
        attack_config["sample_size"] = st.number_input("Sample Size", min_value=10, value=100, step=10, help="Number of samples to query")

    # Sample Input
    st.write("### Sample Input")
    sample_input = None
    if model_type == "Image Classification":
        st.write("Upload a sample image for the attack (optional for Model Stealing)")
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=SUPPORTED_IMAGE_FORMATS,
            help=f"Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}. Max size: {MAX_IMAGE_SIZE_MB}MB"
        )
        sample_input = validate_image(uploaded_file)
        if sample_input:
            st.image(sample_input, caption="Sample Image", width=200)

    elif model_type == "Text Classification":
        st.write("Enter sample text for the attack (optional for Model Stealing)")
        text_input = st.text_area("Sample Text", help="Enter text (e.g., a sentence for sentiment analysis)")
        if text_input and validate_text(text_input):
            sample_input = text_input
            st.write("**Preview**:")
            st.write(sample_input)

    else:  # Tabular Data
        st.write("Enter sample tabular data in CSV format (optional for Model Stealing)")
        csv_input = st.text_area("Sample CSV Data", help="e.g., feature1,feature2\n1.0,2.0")
        if csv_input and validate_tabular_data(csv_input):
            sample_input = csv_input
            st.write("**Preview**:")
            st.code(sample_input)

    # Attack Configuration Preview
    st.write("### Attack Configuration Preview")
    preview = {
        "Attack Type": attack_type,
        "Model Type": model_type,
        "Model Input": model_input.get("file", model_input.get("url", "Not Provided")),
        "Parameters": attack_config,
        "Sample Input": "Provided" if sample_input else "Not Provided"
    }
    st.json(preview)

    # Run Attack
    if st.button("Run Counterfit Attack"):
        if not model_input.get("file") and not model_input.get("url"):
            st.error("Please provide either a model file or an API URL")
            logger.error("No model input provided")
            return
        if model_input.get("url") and not validate_url(model_input["url"]):
            st.error("Invalid or unreachable model URL")
            logger.error("Invalid model URL")
            return
        if attack_type != "Model Stealing" and not sample_input:
            st.error("Please provide a sample input for this attack")
            logger.error("No sample input provided")
            return

        st.info(f"Running {attack_type} attack on {model_type} model...")
        logger.info(f"Starting {attack_type} attack")

        start_time = datetime.now()
        try:
            attack_result = run_counterfit_attack(
                attack_type=attack_type,
                model_type=model_type,
                model_input=model_input,
                attack_config=attack_config,
                sample_input=sample_input
            )

            if "error" in attack_result:
                st.error(f"Attack failed: {attack_result['error']}")
                logger.error(f"Attack failed: {attack_result['error']}")
                return

            st.success("Attack completed successfully!")
            st.subheader("Attack Report")
            st.json(attack_result)

            st.subheader("Detailed Results")
            st.write(f"Success Rate: {attack_result['success_rate']:.2%}")
            vulnerabilities = attack_result.get("vulnerabilities", [])
            if vulnerabilities:
                st.write(f"Found {len(vulnerabilities)} vulnerabilities")
                df = pd.DataFrame(vulnerabilities)
                st.dataframe(df)
            else:
                st.write("No vulnerabilities detected")
            if attack_result.get("recommendations"):
                st.write("Recommendations:")
                for rec in attack_result["recommendations"]:
                    st.write(f"- {rec}")

            if model_type == "Image Classification" and attack_result.get("perturbed_output"):
                st.subheader("Perturbation Visualization")
                st.write("Perturbed image (placeholder)")
                st.image(sample_input, caption="Original Image", width=200)
                st.image(sample_input, caption="Perturbed Image (Placeholder)", width=200)

            report_json = json.dumps(attack_result, indent=2)
            st.download_button(
                label="Download Attack Report",
                data=report_json,
                file_name=f"counterfit_report_{attack_result['scan_id']}.json",
                mime="application/json"
            )

            with open('counterfit.log', 'r') as log_file:
                log_content = log_file.read()
            st.download_button(
                label="Download Log File",
                data=log_content,
                file_name=f"counterfit_log_{attack_result['scan_id']}.log",
                mime="text/plain"
            )

            execution_time = (datetime.now() - start_time).total_seconds()
            st.write(f"**Execution Time**: {execution_time:.2f} seconds")
            logger.info(f"Attack completed in {execution_time:.2f} seconds")

        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            logger.error(f"Unexpected attack error: {str(e)}")
        finally:
            if model_input.get("file") and os.path.exists(model_input["file"]):
                try:
                    os.unlink(model_input["file"])
                    logger.info(f"Temporary model file deleted: {model_input['file']}")
                except Exception as e:
                    logger.error(f"Failed to delete temporary file: {str(e)}")

    st.warning("Ensure you have permission to attack the model. This is a simulated interface; integrate Counterfit for full functionality.")

# Run the app
# display_counterfit_section()