import streamlit as st
import subprocess
import os
import re
import shutil
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
import uuid
import sys
import glob

# Configure independent logging
REPORTS_DIR = "./reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
logger = logging.getLogger('garak')
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent parent logger interference
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(os.path.join(REPORTS_DIR, 'garak_red_teaming.log'))
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.handlers.clear()  # Clear any existing handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Define model configurations
MODEL_CONFIGS = {
    "openai": {
        "models": [
            "gpt-3.5-turbo", 
            "gpt-4", 
            "gpt-4-turbo", 
            "gpt-4o"
        ],
        "requires_api_key": True,
        "env_var": "OPENAI_API_KEY",
        "description": "OpenAI's GPT models"
    },
    "anthropic": {
        "models": [
            "claude-3-opus-20240229", 
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-3.5-sonnet-20240620",
            "claude-3.7-sonnet"
        ],
        "requires_api_key": True,
        "env_var": "ANTHROPIC_API_KEY",
        "description": "Anthropic's Claude models"
    },
    "huggingface": {
        "models": ["meta-llama/Llama-2-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
        "custom_model": True,
        "requires_api_key": False,
        "env_var": "HUGGINGFACE_API_KEY",
        "description": "Hugging Face model hub"
    },
    "replicate": {
        "models": [
            "stability-ai/stablelm-tuned-alpha-7b:c49dae36",
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
        ],
        "custom_model": True,
        "requires_api_key": True,
        "env_var": "REPLICATE_API_KEY",
        "description": "Replicate hosted models"
    },
    "cohere": {
        "models": ["command", "command-r", "command-r-plus"],
        "requires_api_key": True,
        "env_var": "COHERE_API_KEY",
        "description": "Cohere's command models"
    },
    "groq": {
        "models": ["llama-3-8b-8192", "llama-3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        "requires_api_key": True,
        "env_var": "GROQ_API_KEY",
        "description": "Groq's high-performance inference"
    },
    "deepseek": {
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "requires_api_key": True,
        "env_var": "DEEPSEEK_API_KEY",
        "description": "DeepSeek AI models"
    },
    "qwen": {
        "models": ["qwen-max", "qwen-max-0428", "qwen-plus", "qwen-turbo"],
        "requires_api_key": True,
        "env_var": "QWEN_API_KEY",
        "description": "Alibaba's Qwen models"
    },
    "ollama": {
        "models": ["llama3", "llama2", "mistral", "mixtral"],
        "custom_model": True,
        "requires_api_key": False,
        "description": "Locally hosted models via Ollama"
    },
    "local": {
        "models": [],
        "custom_model": True,
        "requires_api_key": False,
        "description": "Locally installed models (requires specifying model path)"
    }
}

# Define available probes with descriptions
AVAILABLE_PROBES = {
    "lmrc.Profanity": "Tests if the model can be prompted to generate profane content",
    "encoding": "Checks for issues with text encoding and character handling",
    "dan.Dan_11_0": "Tests the DAN 11.0 jailbreak prompt",
    "realtoxicityprompts.RTPSevere_Toxicity": "Tests with severe toxicity prompts",
    "knownbadsignatures.ForbesPrefix": "Tests for content that mimics Forbes articles",
    "knownbadsignatures.ThreatPrefix": "Tests for content that could be threatening",
    "integrity.Deception": "Tests if model can be prompted to deceive",
    "integrity.Irresponsible": "Tests for irresponsible use or advice",
    "integrity.LegalAdvice": "Tests if model can be tricked into giving legal advice",
    "dan": "Tests various 'Do Anything Now' jailbreak prompts",
    "leak.PromptExtraction": "Tests if prompts can be extracted from the model",
    "misinformation.Conspiracy": "Tests resistance to generating conspiracy theories",
    "evaluators.TranscendentPromptMitigation": "Tests defenses against transcendent prompts",
    "all": "Run all available probes (warning: very time-consuming)"
}

# Function to run garak command with enhanced error handling
def run_garak_command(command, timeout=300):
    logger.info(f"Running command: {' '.join(command)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=timeout
        )
        elapsed_time = time.time() - start_time
        logger.info(f"Command completed in {elapsed_time:.2f} seconds")
        logger.debug(f"Command output: {result.stdout}")
        logger.debug(f"Command stderr: {result.stderr}")
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        return False, e.stderr, e.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds")
        return False, f"Error: Command timed out after {timeout} seconds", ""
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False, f"Unexpected error: {str(e)}", ""

# Function to test if garak is installed and working
def test_garak():
    logger.info("Testing garak installation")
    command = ["python", "-m", "garak", "--model_type", "test.Blank", "--probes", "test.Test"]
    success, output, stderr = run_garak_command(command, timeout=30)
    
    if success:
        logger.info("Garak test completed successfully")
        return True, output
    else:
        logger.error("Garak test failed")
        return False, output

# Function to parse garak output
def parse_garak_output(output):
    logger.info("Parsing garak output")
    results = []
    
    # Updated regex to match garak 0.9.2.3 output format
    summary_matches = re.finditer(r'probe: (.*?)\s+\|\s+(PASS|FAIL|ERROR)\s+\|\s+(\d+)/(\d+)', output, re.MULTILINE)
    
    for match in summary_matches:
        probe = match.group(1).strip()
        result = match.group(2)
        ok_count = int(match.group(3))
        total_count = int(match.group(4))
        
        result_item = {
            "probe": probe,
            "result": result,
            "ok_count": ok_count,
            "total_count": total_count,
            "percentage": (ok_count / total_count * 100) if total_count > 0 else 0
        }
        results.append(result_item)
    
    if not results:
        logger.warning("No results could be parsed from output")
        return []
    
    logger.info(f"Parsed {len(results)} results")
    return results

# Function to extract HTML report path
def extract_html_report_path(output):
    logger.info("Extracting HTML report path")
    # Try multiple patterns to match garak 0.9.2.3 output
    patterns = [
        r'report html summary written to (.+?\.html)',
        r'📜 report html summary being written to (.+?\.html)',
        r'html report written to (.+?\.html)'
    ]
    for pattern in patterns:
        report_match = re.search(pattern, output)
        if report_match:
            path = report_match.group(1).strip()
            logger.info(f"Found HTML report path: {path}")
            return path
    
    # Fallback: search for HTML files in /tmp/garak.*
    logger.info("Falling back to searching /tmp/garak.* for HTML reports")
    temp_dirs = glob.glob("/tmp/garak.*")
    for temp_dir in temp_dirs:
        html_files = glob.glob(os.path.join(temp_dir, "*.html"))
        if html_files:
            path = html_files[0]  # Take the first HTML file found
            logger.info(f"Found HTML report in temp dir: {path}")
            return path
    
    logger.warning("No HTML report path found in output or /tmp")
    return None

# Function to copy HTML report to ./reports
def copy_html_report(source_path, scan_id):
    logger.info(f"Copying HTML report from {source_path}")
    
    dest_path = os.path.join(REPORTS_DIR, f"garak_html_{scan_id}.html")
    
    try:
        shutil.copy(source_path, dest_path)
        logger.info(f"Successfully copied report to {dest_path}")
        return True, dest_path
    except Exception as e:
        logger.error(f"Failed to copy report: {str(e)}")
        return False, str(e)

# Function to save scan results to ./reports
def save_scan_results(results, model_type, model_name, probe, output, stderr, scan_id):
    logger.info("Saving scan results")
    
    filename = f"garak_{scan_id}.json"
    path = os.path.join(REPORTS_DIR, filename)
    
    # Prepare data to save
    data = {
        "scan_id": scan_id,
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "model_name": model_name,
        "probe": probe,
        "results": results,
        "raw_output": output,
        "raw_stderr": stderr
    }
    
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {path}")
        return True, path
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        return False, str(e)

# Streamlit UI for LLM Vulnerability Scanner
def display_garak():
    # Add custom CSS for better UI
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .report-container {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin-top: 1rem;
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 Advanced LLM Vulnerability Scanner")
    st.write("Test and evaluate LLMs for vulnerabilities using garak")
    scan_id = str(uuid.uuid4())  # Generate scan_id at the start
    
    # Sidebar for settings and history
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool uses [garak](https://github.com/leondz/garak) to test LLMs for 
        vulnerabilities including:
        - Prompt injection
        - Jailbreaks
        - Data leakage
        - Harmful content generation
        - Misinformation
        """)
        
        st.header("Settings")
        timeout = st.slider("Timeout (seconds)", 60, 1200, 300, 60)
        advanced_mode = st.checkbox("Advanced Mode", value=False)
        
        st.header("History")
        reports_dir = Path(REPORTS_DIR)
        if reports_dir.exists():
            reports = list(reports_dir.glob("garak_*.json")) + list(reports_dir.glob("garak_html_*.html"))
            if reports:
                st.write("Recent Reports:")
                for report in sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                    if report.suffix == ".json":
                        with open(report, 'r') as f:
                            report_data = json.load(f)
                        timestamp = report_data.get("timestamp", "Unknown")
                        model_name = report_data.get("model_name", "Unknown")
                        label = f"{report.name} ({model_name}, {timestamp})"
                        mime = "application/json"
                    else:  # .html
                        timestamp = datetime.fromtimestamp(report.stat().st_mtime).isoformat()
                        label = f"{report.name} (HTML, {timestamp})"
                        mime = "text/html"
                    st.download_button(
                        label=label,
                        data=open(report, 'rb').read(),
                        file_name=report.name,
                        mime=mime,
                        key=f"history_{report.name}"
                    )
            else:
                st.write("No previous reports found")
        else:
            st.write("No reports directory")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # About garak section
        with st.expander("ℹ️ About garak", expanded=False):
            st.markdown("""
            **garak** is a Generative AI Red-teaming & Assessment Kit developed to evaluate 
            Large Language Models (LLMs) for security vulnerabilities and potential misuse.
            
            #### Key Features:
            - Evaluates models against a wide range of security probes
            - Tests for jailbreaks, prompt injection, harmful outputs
            - Works with major LLM providers and local models
            - Generates detailed HTML reports
            
            #### Learn More:
            - [GitHub Repository](https://github.com/leondz/garak)
            - [Documentation](https://garak.ai/docs/)
            """)
        
        st.subheader("1. Select Model")
        # Model selection
        model_type = st.selectbox(
            "Model Provider", 
            list(MODEL_CONFIGS.keys()),
            format_func=lambda x: f"{x} - {MODEL_CONFIGS[x]['description']}"
        )
        
        # Get config for selected model type
        model_config = MODEL_CONFIGS[model_type]
        
        # Check for Ollama models if that provider is selected
        if model_type == "ollama":
            with st.spinner("Checking for installed Ollama models..."):
                available_models = get_ollama_models()
                model_config["models"] = available_models
        
        # Model name selection
        if model_config.get("custom_model", False):
            if model_config["models"]:
                model_option = st.radio(
                    "Model Selection",
                    ["Choose from list", "Enter custom model name"]
                )
                if model_option == "Choose from list":
                    model_name = st.selectbox("Select Model", model_config["models"])
                else:
                    model_name = st.text_input("Enter Model Name/Path", "")
            else:
                model_name = st.text_input("Enter Model Name/Path", "")
        else:
            model_name = st.selectbox("Select Model", model_config["models"])
        
        # API key input if required
        if model_config.get("requires_api_key", False):
            api_key = st.text_input(
                f"{model_type.capitalize()} API Key", 
                type="password",
                help=f"Required to access {model_type} models"
            )
            if model_config.get("env_var"):
                st.info(f"You can also set the {model_config['env_var']} environment variable instead of entering it here.")
        else:
            api_key = None
            if model_type == "local":
                st.info("Make sure you have the model correctly installed and accessible.")
            elif model_type == "ollama":
                if len(model_config["models"]) > 0:
                    st.success(f"Found {len(model_config['models'])} installed Ollama models.")
                else:
                    st.warning("No Ollama models detected. Make sure Ollama is installed and running.")
                    st.info("You can install models using 'ollama pull MODEL_NAME' in your terminal.")
        
        # Advanced model options
        if advanced_mode:
            with st.expander("Advanced Model Options", expanded=False):
                model_args = st.text_area(
                    "Additional Model Arguments (JSON format)",
                    value="{}",
                    help="Extra parameters in JSON format to pass to the model"
                )
                try:
                    json.loads(model_args)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for model arguments")
    
    with col2:
        st.subheader("2. Select Probe")
        probe_options = list(AVAILABLE_PROBES.keys())
        probe = st.selectbox(
            "Select Probe Type", 
            probe_options,
            format_func=lambda x: f"{x.split('.')[-1] if '.' in x else x}"
        )
        
        st.info(AVAILABLE_PROBES.get(probe, "No description available"))
        
        if advanced_mode:
            with st.expander("Advanced Probe Options", expanded=False):
                num_generations = st.number_input(
                    "Number of Generations", 
                    min_value=1, 
                    max_value=50, 
                    value=2,
                    help="Number of responses to generate for each probe"
                )
        else:
            num_generations = 3
    
    st.markdown("---")
    
    st.subheader("3. Run Scan")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("✅ Test garak Installation"):
            with st.spinner("Testing garak..."):
                test_passed, test_output = test_garak()
            
            if test_passed:
                st.success("✅ garak is installed and working correctly!")
            else:
                st.error("❌ garak test failed. Please check your installation.")
                st.error("Make sure you've installed garak using: `pip install garak`")
            
            with st.expander("Test Output Details", expanded=False):
                st.text_area("Output", test_output, height=200)
    
    with col2:
        run_disabled = False
        run_tooltip = ""
        
        if model_config.get("requires_api_key", False) and not api_key:
            run_disabled = True
            run_tooltip = f"Please provide a {model_type} API key"
        
        if not model_name:
            run_disabled = True
            run_tooltip = "Please select or enter a model name"
        
        run_button = st.button(
            "🚀 Run Vulnerability Scan", 
            disabled=run_disabled,
            help=run_tooltip if run_disabled else "Start the vulnerability scan"
        )
    
    if run_button:
        if model_config.get("requires_api_key", False) and not api_key:
            st.error(f"Please provide a {model_type} API key")
        elif not model_name:
            st.error("Please select or enter a model name")
        else:
            with st.status("Running vulnerability scan...", expanded=True) as status:
                st.write(f"Starting scan on {model_type}/{model_name}...")
                
                # Set the API key as an environment variable if provided
                original_env = None
                if api_key and model_config.get("env_var"):
                    original_env = os.environ.get(model_config["env_var"])
                    os.environ[model_config["env_var"]] = api_key
                
                try:
                    # Construct the garak command
                    command = [
                        "python", "-m", "garak",
                        "--model_type", model_type,
                        "--model_name", model_name,
                        "--probes", probe,
                        "--generations", str(num_generations)
                    ]
                    
                    # Add advanced options
                    if advanced_mode:
                        try:
                            model_args_dict = json.loads(model_args)
                            if model_args_dict:
                                for key, value in model_args_dict.items():
                                    command.extend([f"--{key}", str(value)])
                        except (json.JSONDecodeError, NameError):
                            pass
                    
                    st.write(f"Running garak with {probe} probe...")
                    
                    # Run garak
                    start_time = datetime.now()
                    success, output, stderr = run_garak_command(command, timeout=timeout)
                    
                    # Save log to ./reports
                    log_path = os.path.join(REPORTS_DIR, f"garak_log_{scan_id}.log")
                    with open(os.path.join(REPORTS_DIR, 'garak_red_teaming.log'), 'r') as log_file:
                        log_content = log_file.read()
                    with open(log_path, 'w') as f:
                        f.write(log_content + f"\nRaw Output:\n{output}\nRaw Stderr:\n{stderr}")
                    logger.info(f"Log saved to {log_path}")
                    
                    if success:
                        st.write("Processing results...")
                        
                        # Parse results
                        results = parse_garak_output(output)
                        
                        # Save results
                        save_success, json_report_path = save_scan_results(
                            results, model_type, model_name, probe, output, stderr, scan_id
                        )
                        
                        # Extract and copy HTML report
                        html_report_path = extract_html_report_path(output)
                        html_path = None
                        if html_report_path and os.path.exists(html_report_path):
                            report_success, html_path = copy_html_report(html_report_path, scan_id)
                        else:
                            st.warning("⚠️ No HTML report generated. Check the log for details.")
                        
                        # Clean up temporary garak directories
                        temp_dirs = glob.glob("/tmp/garak.*")
                        for temp_dir in temp_dirs:
                            try:
                                shutil.rmtree(temp_dir)
                                logger.info(f"Cleaned up temporary directory {temp_dir}")
                            except Exception as e:
                                logger.warning(f"Failed to clean up temp directory {temp_dir}: {str(e)}")
                        
                        # Update status
                        status.update(label="Vulnerability scan completed!", state="complete")
                        
                        # Display results
                        st.success(f"✅ Vulnerability scan completed for {model_type}/{model_name}! Report saved to {json_report_path}")
                        
                        if results:
                            st.subheader("📊 Scan Results")
                            results_data = []
                            for r in results:
                                status_icon = "✅" if r["result"] == "PASS" else "❌"
                                results_data.append({
                                    "Probe": r["probe"],
                                    "Status": f"{status_icon} {r['result']}",
                                    "Pass Rate": f"{r['ok_count']}/{r['total_count']} ({r['percentage']:.1f}%)"
                                })
                            st.table(results_data)
                        else:
                            st.warning("⚠️ No parsed results available. Check the full output below.")
                        
                        st.subheader("📄 Reports")
                        with open(json_report_path, 'r') as f:
                            report_json = f.read()
                        st.download_button(
                            label="Download JSON Report",
                            data=report_json,
                            file_name=f"garak_{scan_id}.json",
                            mime="application/json"
                        )
                        
                        if html_path and os.path.exists(html_path):
                            with open(html_path, 'r') as f:
                                report_html = f.read()
                            st.download_button(
                                label="Download HTML Report",
                                data=report_html,
                                file_name=f"garak_html_{scan_id}.html",
                                mime="text/html"
                            )
                            # Display HTML report
                            st.subheader("HTML Report Preview")
                            st.components.v1.html(report_html, height=600, scrolling=True)
                        
                        # Download log
                        with open(log_path, 'r') as f:
                            log_content = f.read()
                        st.download_button(
                            label="Download Log File",
                            data=log_content,
                            file_name=f"garak_log_{scan_id}.log",
                            mime="text/plain"
                        )
                        
                        # Show raw output
                        st.subheader("Full Scan Output")
                        st.text_area("Raw Output", output + f"\n\nStderr:\n{stderr}", height=400)
                        
                        # Execution time
                        execution_time = (datetime.now() - start_time).total_seconds()
                        st.write(f"**Scan Execution Time**: {execution_time:.2f} seconds")
                        logger.info(f"Scan completed in {execution_time:.2f} seconds")
                    
                    else:
                        status.update(label="Vulnerability scan failed", state="error")
                        st.error("❌ Error running garak")
                        st.text_area("Error Details", output + f"\n\nStderr:\n{stderr}", height=200)
                        
                        # Save error report
                        error_report = {
                            "scan_id": scan_id,
                            "timestamp": datetime.now().isoformat(),
                            "model_type": model_type,
                            "model_name": model_name,
                            "probe": probe,
                            "error": output,
                            "stderr": stderr,
                            "execution_time": (datetime.now() - start_time).total_seconds()
                        }
                        json_report_path = os.path.join(REPORTS_DIR, f"garak_{scan_id}.json")
                        with open(json_report_path, 'w') as f:
                            json.dump(error_report, f, indent=2)
                        logger.info(f"Error report saved to {json_report_path}")
                        
                        # Download error report
                        with open(json_report_path, 'r') as f:
                            report_json = f.read()
                        st.download_button(
                            label="Download Error Report",
                            data=report_json,
                            file_name=f"garak_{scan_id}.json",
                            mime="application/json"
                        )
                        
                        # Download log
                        with open(log_path, 'r') as f:
                            log_content = f.read()
                        st.download_button(
                            label="Download Log File",
                            data=log_content,
                            file_name=f"garak_log_{scan_id}.log",
                            mime="text/plain"
                        )
                        
                        st.subheader("🔧 Troubleshooting")
                        st.markdown("""
                        Common issues:
                        - Invalid API key
                        - Model name not accessible
                        - Network connection issues
                        - garak installation problems
                        
                        Try running the 'Test garak Installation' button to verify your setup.
                        """)
                
                except Exception as e:
                    status.update(label="Unexpected error occurred", state="error")
                    st.error(f"❌ Unexpected error: {str(e)}")
                    logger.error(f"Unexpected error during scan: {str(e)}")
                
                finally:
                    if api_key and model_config.get("env_var"):
                        if original_env:
                            os.environ[model_config["env_var"]] = original_env
                        else:
                            os.environ.pop(model_config["env_var"], None)
                    # Clean up any remaining /tmp/garak.* directories
                    temp_dirs = glob.glob("/tmp/garak.*")
                    for temp_dir in temp_dirs:
                        try:
                            shutil.rmtree(temp_dir)
                            logger.info(f"Cleaned up temporary directory {temp_dir}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up temp directory {temp_dir}: {str(e)}")

# Function to get installed Ollama models
def get_ollama_models():
    logger.info("Checking for installed Ollama models")
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=10
        )
        
        models = []
        for line in result.stdout.strip().split('\n'):
            if line and not line.startswith('NAME'):
                parts = line.split()
                if parts:
                    models.append(parts[0])
        
        if models:
            logger.info(f"Found {len(models)} Ollama models: {', '.join(models)}")
            return models
        else:
            logger.warning("No Ollama models found")
            return ["llama3", "llama2", "mistral", "mixtral"]
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error checking Ollama models: {str(e)}")
        return ["llama3", "llama2", "mistral", "mixtral"]
