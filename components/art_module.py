import streamlit as st
import numpy as np
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, CarliniL2Method
from art.estimators.classification import KerasClassifier
import tensorflow as tf
import logging
import os
import uuid
import json
from datetime import datetime
from PIL import Image
import sys
from pathlib import Path
# from typing import Any
from typing import Optional

# Configure independent logging
REPORTS_DIR = "./reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
logger = logging.getLogger('art')
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent parent logger interference
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(os.path.join(REPORTS_DIR, 'art_red_teaming.log'))
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.handlers.clear()  # Clear any existing handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def validate_model_file(model_path: str) -> bool:
    """Validate if the model file is a valid Keras model."""
    try:
        tf.keras.models.load_model(model_path)
        logger.info(f"Model file validated: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Invalid model file: {str(e)}")
        return False

def validate_image(file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[np.ndarray]:
    """Validate and preprocess image input."""
    try:
        image = Image.open(file)
        image = image.resize((224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        logger.info("Image validated and preprocessed")
        return image_array
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None

def save_adversarial_image(adversarial_sample: np.ndarray, scan_id: str) -> str:
    """Save adversarial image to ./reports."""
    try:
        # Denormalize image for saving (assuming input was normalized)
        img = adversarial_sample[0]
        img = (img - img.min()) / (img.max() - img.min()) * 255  # Scale to 0-255
        img = img.astype(np.uint8)
        image = Image.fromarray(img)
        image_path = os.path.join(REPORTS_DIR, f"art_adversarial_{scan_id}.png")
        image.save(image_path)
        logger.info(f"Adversarial image saved to {image_path}")
        return image_path
    except Exception as e:
        logger.error(f"Failed to save adversarial image: {str(e)}")
        return ""

def compute_metrics(classifier: KerasClassifier, original_image: np.ndarray, adversarial_image: np.ndarray) -> dict:
    """Compute confidence scores and perturbation magnitude."""
    try:
        original_pred = classifier.predict(original_image)
        adversarial_pred = classifier.predict(adversarial_image)
        original_confidence = float(np.max(original_pred))
        adversarial_confidence = float(np.max(adversarial_pred))
        perturbation_magnitude = float(np.linalg.norm(adversarial_image - original_image))
        return {
            "original_confidence": original_confidence,
            "adversarial_confidence": adversarial_confidence,
            "perturbation_magnitude": perturbation_magnitude
        }
    except Exception as e:
        logger.error(f"Failed to compute metrics: {str(e)}")
        return {
            "original_confidence": 0.0,
            "adversarial_confidence": 0.0,
            "perturbation_magnitude": 0.0
        }

def display_art_section():
    st.title("Adversarial Robustness Toolbox")
    logger.info("ART section accessed")
    scan_id = str(uuid.uuid4())  # Generate scan_id at the start

    if st.button("Learn about ART"):
        with st.expander("Adversarial Robustness Toolbox Explained", expanded=True):
            st.markdown("""
            The Adversarial Robustness Toolbox (ART) is a Python library for Machine Learning Security. It provides tools to defend and evaluate Machine Learning models and applications against adversarial threats.

            **Key features of ART:**
            1. **Adversarial Attacks**: Implements state-of-the-art adversarial attack algorithms.
            2. **Defenses**: Offers various defense methods to make ML models more robust.
            3. **Robustness Metrics**: Provides metrics to quantify the robustness of ML models.
            4. **Detection Methods**: Includes techniques to detect adversarial examples.
            5. **Framework Support**: Compatible with popular ML frameworks like TensorFlow, Keras, PyTorch, and more.
                        
            ##### Visit their [Github Page](https://github.com/Trusted-AI/adversarial-robustness-toolbox?tab=readme-ov-file)
            """)
            logger.info("ART information expanded")

    # Sidebar for history
    with st.sidebar:
        st.header("History")
        reports_dir = Path(REPORTS_DIR)
        if reports_dir.exists():
            reports = list(reports_dir.glob("art_*.json"))
            if reports:
                st.write("Recent Reports:")
                for report in sorted(reports, reverse=True)[:5]:
                    with open(report, 'r') as f:
                        report_data = json.load(f)
                    report_name = report.name
                    timestamp = report_data.get("timestamp", "Unknown")
                    attack_type = report_data.get("attack_type", "Unknown")
                    st.download_button(
                        label=f"{report_name} ({attack_type}, {timestamp})",
                        data=open(report, 'r').read(),
                        file_name=report_name,
                        mime="application/json"
                    )
            else:
                st.write("No previous reports found")
        else:
            st.write("No reports directory")

    st.subheader("ART Attack Simulation")

    # Model Configuration
    st.write("### 1. Model Configuration")
    st.write("Upload Model File")
    model_file = st.file_uploader("Upload Keras Model File", type=["h5"], help="Upload a .h5 file containing a Keras model")
    model_path = None
    if model_file:
        try:
            with open(os.path.join(REPORTS_DIR, f"model_{scan_id}.h5"), "wb") as f:
                f.write(model_file.getbuffer())
            model_path = os.path.join(REPORTS_DIR, f"model_{scan_id}.h5")
            if validate_model_file(model_path):
                st.write(f"**Uploaded Model**: {model_file.name}")
            else:
                st.error("Invalid Keras model file")
                model_path = None
        except Exception as e:
            st.error(f"Failed to process model file: {str(e)}")
            logger.error(f"Model file processing error: {str(e)}")
            model_path = None

    # Attack Selection
    st.write("### 2. Choose Attack Type")
    attack_type = st.selectbox("Attack Type", ["Fast Gradient Method (FGM)", "Projected Gradient Descent (PGD)", "DeepFool", "Carlini & Wagner L2"])

    # Attack Configuration
    st.write("### 3. Configure Attack Parameters")
    attack_config = {}
    if attack_type == "Fast Gradient Method (FGM)":
        st.write("Fast Gradient Method (FGM)")
        attack_config["epsilon"] = st.slider("Epsilon", 0.0, 1.0, 0.1, help="Perturbation size")
        attack_config["norm"] = st.selectbox("Norm", ["inf", "1", "2"], help="Type of norm to use")

    elif attack_type == "Projected Gradient Descent (PGD)":
        st.write("Projected Gradient Descent (PGD)")
        attack_config["epsilon"] = st.slider("Epsilon", 0.0, 1.0, 0.1, help="Perturbation size")
        attack_config["max_iter"] = st.slider("Max Iterations", 1, 100, 10, help="Maximum number of iterations")
        attack_config["step_size"] = st.slider("Step Size", 0.01, 1.0, 0.1, help="Step size for each iteration")

    elif attack_type == "DeepFool":
        st.write("DeepFool Attack")
        attack_config["max_iter"] = st.slider("Max Iterations", 10, 1000, 100, help="Maximum number of iterations")
        attack_config["epsilon"] = st.slider("Epsilon", 0.0001, 1.0, 0.02, help="Overshoot parameter")

    elif attack_type == "Carlini & Wagner L2":
        st.write("Carlini & Wagner L2 Attack")
        attack_config["confidence"] = st.slider("Confidence", 0.0, 1.0, 0.5, help="Confidence of adversarial examples")
        attack_config["max_iter"] = st.slider("Max Iterations", 1, 1000, 100, help="Maximum number of iterations")
        attack_config["learning_rate"] = st.slider("Learning Rate", 0.001, 0.1, 0.01, help="Learning rate for optimization")

    # Sample Input
    st.write("### 4. Provide Sample Input")
    sample_input = st.file_uploader("Upload sample image", type=["jpg", "png"], help="Upload a JPG or PNG image (will be resized to 224x224)")

    # Run Attack
    if st.button("Run ART Attack"):
        if not model_path or not sample_input:
            st.error("Please provide both a model file and sample input.")
            logger.error("Missing model file or sample input")
            return

        image_array = validate_image(sample_input)
        if image_array is None:
            st.error("Invalid sample image")
            logger.error("Invalid sample image")
            return

        st.info(f"Running {attack_type} using ART...")
        logger.info(f"Starting {attack_type} attack")

        start_time = datetime.now()
        try:
            # Load the model
            model = tf.keras.models.load_model(model_path)
            classifier = KerasClassifier(model=model)
            logger.info("Model loaded successfully")

            # Create and run the attack
            if attack_type == "Fast Gradient Method (FGM)":
                attack = FastGradientMethod(estimator=classifier, eps=attack_config["epsilon"], norm=int(attack_config["norm"]) if attack_config["norm"].isdigit() else attack_config["norm"])
            elif attack_type == "Projected Gradient Descent (PGD)":
                attack = ProjectedGradientDescent(estimator=classifier, eps=attack_config["epsilon"], max_iter=attack_config["max_iter"], eps_step=attack_config["step_size"])
            elif attack_type == "DeepFool":
                attack = DeepFool(classifier=classifier, max_iter=attack_config["max_iter"], epsilon=attack_config["epsilon"])
            elif attack_type == "Carlini & Wagner L2":
                attack = CarliniL2Method(classifier=classifier, confidence=attack_config["confidence"], max_iter=attack_config["max_iter"], learning_rate=attack_config["learning_rate"])

            adversarial_sample = attack.generate(x=image_array)
            logger.info("Attack completed successfully")

            # Compute metrics
            metrics = compute_metrics(classifier, image_array, adversarial_sample)

            # Save adversarial image
            image_path = save_adversarial_image(adversarial_sample, scan_id)

            # Prepare report
            report = {
                "scan_id": scan_id,
                "attack_type": attack_type,
                "model_path": os.path.basename(model_path),
                "timestamp": datetime.now().isoformat(),
                "attack_config": attack_config,
                "metrics": metrics,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "adversarial_image_path": os.path.basename(image_path) if image_path else "N/A"
            }

            # Save report to ./reports
            report_path = os.path.join(REPORTS_DIR, f"art_{scan_id}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {report_path}")

            # Save log to ./reports
            log_path = os.path.join(REPORTS_DIR, f"art_log_{scan_id}.log")
            with open(os.path.join(REPORTS_DIR, 'art_red_teaming.log'), 'r') as log_file:
                log_content = log_file.read()
            with open(log_path, 'w') as f:
                f.write(log_content)
            logger.info(f"Log saved to {log_path}")

            # Display results
            st.success(f"Attack completed successfully! Report saved to {report_path}")
            st.subheader("Attack Report")
            st.json(report)

            st.subheader("Image Comparison")
            st.write("Original Image:")
            st.image(sample_input, use_column_width=True)
            st.write("Adversarial Image:")
            st.image(adversarial_sample[0], use_column_width=True)

            st.subheader("Metrics")
            st.write(f"Original Confidence: {metrics['original_confidence']:.2%}")
            st.write(f"Adversarial Confidence: {metrics['adversarial_confidence']:.2%}")
            st.write(f"Perturbation Magnitude: {metrics['perturbation_magnitude']:.4f}")

            # Download buttons
            with open(report_path, 'r') as f:
                report_json = f.read()
            st.download_button(
                label="Download Attack Report",
                data=report_json,
                file_name=f"art_{scan_id}.json",
                mime="application/json"
            )

            with open(log_path, 'r') as f:
                log_content = f.read()
            st.download_button(
                label="Download Log File",
                data=log_content,
                file_name=f"art_log_{scan_id}.log",
                mime="text/plain"
            )

            if image_path:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                st.download_button(
                    label="Download Adversarial Image",
                    data=image_data,
                    file_name=f"art_adversarial_{scan_id}.png",
                    mime="image/png"
                )

            st.write(f"**Execution Time**: {report['execution_time']:.2f} seconds")

        except Exception as e:
            st.error(f"An error occurred during the attack: {str(e)}")
            logger.error(f"Attack failed: {str(e)}")

            # Save error report
            error_report = {
                "scan_id": scan_id,
                "attack_type": attack_type,
                "model_path": os.path.basename(model_path) if model_path else "N/A",
                "timestamp": datetime.now().isoformat(),
                "attack_config": attack_config,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            report_path = os.path.join(REPORTS_DIR, f"art_{scan_id}.json")
            with open(report_path, 'w') as f:
                json.dump(error_report, f, indent=2)
            logger.info(f"Error report saved to {report_path}")

            # Save error log
            log_path = os.path.join(REPORTS_DIR, f"art_log_{scan_id}.log")
            with open(os.path.join(REPORTS_DIR, 'art_red_teaming.log'), 'r') as log_file:
                log_content = log_file.read()
            with open(log_path, 'w') as f:
                f.write(log_content)
            logger.info(f"Error log saved to {log_path}")

            # Download error report
            with open(report_path, 'r') as f:
                report_json = f.read()
            st.download_button(
                label="Download Error Report",
                data=report_json,
                file_name=f"art_{scan_id}.json",
                mime="application/json"
            )

            # Download error log
            with open(log_path, 'r') as f:
                log_content = f.read()
            st.download_button(
                label="Download Log File",
                data=log_content,
                file_name=f"art_log_{scan_id}.log",
                mime="text/plain"
            )

            st.write(f"**Execution Time**: {error_report['execution_time']:.2f} seconds")

        finally:
            # Clean up model file
            if model_path and os.path.exists(model_path):
                try:
                    os.unlink(model_path)
                    logger.info(f"Temporary model file deleted: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to delete temporary model file: {str(e)}")

    st.warning("Note: Ensure that you have the necessary permissions and rights to use the model and perform adversarial attacks. Always handle model files and attack results with appropriate security measures.")
