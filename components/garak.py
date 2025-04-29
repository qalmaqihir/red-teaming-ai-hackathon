# import streamlit as st
# from components.model_configs import MODEL_CONFIGS
# import subprocess
# import os
# import re
# import shutil  # Import shutil for file operations

# def run_garak_command(command):
#     try:
#         result = subprocess.run(command, capture_output=True, text=True, check=True)
#         return result.stdout
#     except subprocess.CalledProcessError as e:
#         return f"Error: {e.stderr}"

# import re

# def test_garak():
#     command = ["python", "-m", "garak", "--model_type", "test.Blank", "--probes", "test.Test"]
#     output = run_garak_command(command)
#     return True, output

#     # # Check if the test passed using regex
#     # if re.search(r'test\.Test always\.Pass: PASS', output):
#     #     return True, output
#     # else:
#     #     return False, output



# def parse_garak_output(output):
#     # Extract the summary line
#     summary_match = re.search(r'(.*): (PASS|FAIL)\s+ok on\s+(\d+)/\s*(\d+)', output)
#     if summary_match:
#         probe = summary_match.group(1).strip()
#         result = summary_match.group(2)
#         ok_count = int(summary_match.group(3))
#         total_count = int(summary_match.group(4))
#         return f"{probe}: {result} ({ok_count}/{total_count} passed)"
#     else:
#         return "Unable to parse output"

# def extract_html_report_path(output):
#     # Extract the HTML report path from the output
#     report_match = re.search(r'📜 report html summary being written to (.+)', output)
#     if report_match:
#         return report_match.group(1).strip()
#     return None

# def copy_html_report(source_path, dest_path):
#     try:
#         shutil.copy(source_path, dest_path)
#         return True
#     except Exception as e:
#         st.error(f"Failed to copy report: {e}")
#         return False

# def display_garak():
#     st.title("LLM Vulnerability Scanner - garak")
    
#     if st.button("Learn about garak"):
#         with st.expander("garak Explained", expanded=True):
#             st.markdown("""
#             garak is a Generative AI Red-teaming & Assessment Kit. It checks if an LLM can be made to fail in undesirable ways, probing for issues like hallucination, data leakage, prompt injection, misinformation, toxicity generation, and jailbreaks.

#             ##### Visit their [Github Page](https://github.com/leondz/garak/tree/main?tab=readme-ov-file) for more information.
                        
#             -----------     
#             """)

#     # Test if garak is working
#     if st.button("Test garak"):
#         with st.spinner("Testing garak..."):
#             test_passed, test_output = test_garak()
#         if test_passed:
#             st.success("garak is working correctly!")
#         else:
#             st.error("garak test failed. Please check your installation.")
#         st.text_area("Test Output", test_output, height=200)

#     st.write("Select the model and provide necessary credentials to test:")

#     model_type = st.selectbox("Select Model Type", ["openai", "huggingface", "replicate", "cohere", "groq"])
    
#     if model_type == "openai":
#         model_name = st.selectbox("Select OpenAI Model", ["gpt-3.5-turbo", "gpt-4"])
#         api_key = st.text_input("OpenAI API Key", type="password")
#     elif model_type == "huggingface":
#         model_name = st.text_input("Enter Hugging Face Model Name", value="")
#         api_key = st.text_input("Hugging Face API Token (optional)", type="password")
#     elif model_type == "replicate":
#         model_name = st.text_input("Enter Replicate Model Name", value="stability-ai/stablelm-tuned-alpha-7b:c49dae36")
#         api_key = st.text_input("Replicate API Token", type="password")
#     elif model_type == "cohere":
#         model_name = st.selectbox("Select Cohere Model", ["command"])
#         api_key = st.text_input("Cohere API Key", type="password")
#     elif model_type == "groq":
#         model_name = st.text_input("Enter Groq Model Name")
#         api_key = st.text_input("Groq API Key", type="password")

#     probe = st.selectbox("Select Probe", ["lmrc.Profanity", "encoding", "dan.Dan_11_0", "realtoxicityprompts.RTPSevere_Toxicity"])

#     if st.button("Run garak"):
#         if not api_key and model_type != "huggingface":
#             st.error("Please provide an API key.")
#         else:
#             st.info("Running garak...")
            
#             # Set the API key as an environment variable
#             if api_key:
#                 os.environ[f"{model_type.upper()}_API_KEY"] = api_key

#             # Construct the garak command
#             command = [
#                 "python", "-m", "garak",
#                 "--model_type", model_type,
#                 "--model_name", model_name,
#                 "--probes", probe
#             ]

#             # Run garak
#             output = run_garak_command(command)
            
#             # Parse and display the results
#             parsed_output = parse_garak_output(output)
#             st.subheader("Scan Results")
#             st.write(parsed_output)

#             # Extract the HTML report path
#             html_report_path = extract_html_report_path(output)
#             if html_report_path:
#                 # Define destination path for the copied report
#                 dest_report_path = os.path.join(os.getcwd(), "garak_report.html")
#                 # Copy the HTML report to the same directory as the app
#                 if copy_html_report(html_report_path, dest_report_path):
#                     # Ensure the path is valid for markdown rendering
#                     if os.path.isfile(dest_report_path):
#                         # st.markdown(f"[View HTML Report](./garak_report.html)", unsafe_allow_html=True)
#                         st.markdown(f"[View HTML Report]({dest_report_path})", unsafe_allow_html=True)
#                     else:
#                         st.error("Report file not found.")

#             # Display the full output
#             st.subheader("Full garak Output")
#             st.text_area("", output, height=300)

#             # Remove the API key from environment variables for security
#             if api_key:
#                 os.environ.pop(f"{model_type.upper()}_API_KEY", None)


#### Version 2

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
from pathlib import Path
import urllib.parse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("garak_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("garak_scanner")

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
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        return False, f"Error: {e.stderr}"
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds")
        return False, f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False, f"Unexpected error: {str(e)}"

# Function to test if garak is installed and working
def test_garak():
    logger.info("Testing garak installation")
    command = ["python", "-m", "garak", "--model_type", "test.Blank", "--probes", "test.Test"]
    success, output = run_garak_command(command, timeout=30)
    
    if success:
        logger.info("Garak test completed successfully")
        # Don't check for specific patterns, just consider it successful if the command ran
        return True, output
    else:
        logger.error("Garak test failed")
        return False, output

# Function to parse garak output
def parse_garak_output(output):
    logger.info("Parsing garak output")
    results = []
    
    # Extract all summary lines
    summary_matches = re.finditer(r'(.*?): (PASS|FAIL|ERROR)\s+ok on\s+(\d+)/\s*(\d+)', output)
    
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
        return None
    
    return results

# Function to extract HTML report path
def extract_html_report_path(output):
    logger.info("Extracting HTML report path")
    report_match = re.search(r'📜 report html summary being written to (.+)', output)
    if report_match:
        path = report_match.group(1).strip()
        logger.info(f"Found HTML report path: {path}")
        return path
    logger.warning("No HTML report path found in output")
    return None

# Function to copy HTML report to a more accessible location
def copy_html_report(source_path, dest_dir="reports"):
    logger.info(f"Copying HTML report from {source_path}")
    
    # Create reports directory if it doesn't exist
    Path(dest_dir).mkdir(exist_ok=True)
    
    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"garak_report_{timestamp}.html"
    dest_path = os.path.join(dest_dir, filename)
    
    try:
        shutil.copy(source_path, dest_path)
        logger.info(f"Successfully copied report to {dest_path}")
        return True, dest_path
    except Exception as e:
        logger.error(f"Failed to copy report: {str(e)}")
        return False, str(e)

# Function to save scan results
def save_scan_results(results, model_type, model_name, probe, output):
    logger.info("Saving scan results")
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scan_{model_type}_{timestamp}.json"
    path = os.path.join("results", filename)
    
    # Prepare data to save
    data = {
        "timestamp": timestamp,
        "model_type": model_type,
        "model_name": model_name,
        "probe": probe,
        "results": results,
        "raw_output": output
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
    
    # Sidebar for settings and info
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
        
        # st.header("History")
        # reports_dir = Path("reports")
        # if reports_dir.exists():
        #     reports = list(reports_dir.glob("*.html"))
        #     if reports:
        #         for report in sorted(reports, reverse=True)[:5]:
        #             st.markdown(f"[{report.name}](./{report})")
        #     else:
        #         st.write("No previous reports found")
        # else:
        #     st.write("No reports directory")
   

        st.header("History")
        reports_dir = Path("reports")

        if reports_dir.exists():
            reports = list(reports_dir.glob("*.html"))
            if reports:
                for report in sorted(reports, reverse=True)[:5]:
                    file_path = report.resolve()
                    file_url = f"file://{urllib.parse.quote(str(file_path))}"
                    # st.markdown(f"[{report.name}]({file_url})")
                    st.markdown(
                        f'<a href="{file_url}" target="_blank">{report.name}</a>',
                        unsafe_allow_html=True,
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
                # Update the model config with the found models
                model_config["models"] = available_models
        
        # Model name selection
        if model_config.get("custom_model", False):
            # Allow custom model input
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
            # Only allow selection from predefined list
            model_name = st.selectbox("Select Model", model_config["models"])
        
        # API key input if required
        if model_config.get("requires_api_key", False):
            api_key = st.text_input(
                f"{model_type.capitalize()} API Key", 
                type="password",
                help=f"Required to access {model_type} models"
            )
            # Show env var info
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
        # Probe selection with description
        probe_options = list(AVAILABLE_PROBES.keys())
        probe = st.selectbox(
            "Select Probe Type", 
            probe_options,
            format_func=lambda x: f"{x.split('.')[-1] if '.' in x else x}"
        )
        
        # Show probe description
        st.info(AVAILABLE_PROBES.get(probe, "No description available"))
        
        # Advanced probe options
        if advanced_mode:
            with st.expander("Advanced Probe Options", expanded=False):
                num_generations = st.number_input(
                    "Number of Generations", 
                    min_value=1, 
                    max_value=50, 
                    value=2,
                    help="Number of responses to generate for each probe"
                )
                
                buffer_tokens = st.number_input(
                    "Buffer Tokens", 
                    min_value=0, 
                    max_value=1000, 
                    value=200,
                    help="Number of tokens to use as buffer in requests"
                )
        else:
            num_generations = 3
            buffer_tokens = 200
    
    # Horizontal line separator
    st.markdown("---")
    
    # Run section
    st.subheader("3. Run Scan")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Test if garak is installed
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
        # Run button with validation
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
    
    # Run the scan
    if run_button:
        # Validate inputs
        if model_config.get("requires_api_key", False) and not api_key:
            st.error(f"Please provide a {model_type} API key")
        elif not model_name:
            st.error("Please select or enter a model name")
        else:
            # Create progress container
            progress_container = st.empty()
            with progress_container.container():
                st.info(f"🔄 Starting vulnerability scan on {model_type}/{model_name}...")
                progress_bar = st.progress(0)
            
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
                
                # Don't add buffer_tokens as it's not a valid argument
                # This caused the error
                
                # Add any advanced options
                if advanced_mode:
                    try:
                        model_args_dict = json.loads(model_args)
                        if model_args_dict:
                            for key, value in model_args_dict.items():
                                command.extend([f"--{key}", str(value)])
                    except (json.JSONDecodeError, NameError):
                        # If model_args not defined or invalid, ignore
                        pass
                
                # Update progress
                progress_bar.progress(10)
                progress_container.info(f"🔄 Running garak with {probe} probe...")
                
                # Run garak
                success, output = run_garak_command(command, timeout=timeout)
                
                # Update progress
                progress_bar.progress(70)
                
                if success:
                    progress_container.info("🔄 Processing results...")
                    
                    # Parse results
                    results = parse_garak_output(output)
                    
                    # Save results
                    save_success, save_path = save_scan_results(
                        results, model_type, model_name, probe, output
                    )
                    
                    # Extract and copy HTML report
                    html_report_path = extract_html_report_path(output)
                    report_path = None
                    if html_report_path:
                        report_success, report_path = copy_html_report(html_report_path)
                        progress_bar.progress(90)
                    
                    # Remove progress container and show results
                    progress_container.empty()
                    
                    # Display results
                    st.success(f"✅ Vulnerability scan completed for {model_type}/{model_name}")
                    
                    if results:
                        # Create result table
                        st.subheader("📊 Scan Results")
                        
                        # Create dataframe for results
                        results_data = []
                        for r in results:
                            status_icon = "✅" if r["result"] == "PASS" else "❌"
                            results_data.append({
                                "Probe": r["probe"],
                                "Status": f"{status_icon} {r['result']}",
                                "Pass Rate": f"{r['ok_count']}/{r['total_count']} ({r['percentage']:.1f}%)"
                            })
                        
                        # Display as table
                        st.table(results_data)
                        
                        # Show HTML report link if available
                        if report_path:
                            st.subheader("📄 Detailed Report")
                            st.markdown(f"""
                            <div class="report-container">
                                <h4>HTML Report Generated</h4>
                                <p>A detailed HTML report has been generated with comprehensive results.</p>
                                <a href="./{report_path}" target="_blank">
                                    <button style="background-color:#4CAF50;color:white;padding:10px 15px;border:none;border-radius:5px;cursor:pointer;">
                                        Open Full Report
                                    </button>
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No parsed results available. Check the full output below.")
                    
                    # Show raw output in expandable section
                    with st.expander("Full Scan Output", expanded=False):
                        st.text_area("", output, height=400)
                        
                else:
                    # Error case
                    progress_container.empty()
                    st.error("❌ Error running garak")
                    st.text_area("Error Details", output, height=200)
                    
                    # Provide troubleshooting help
                    st.subheader("🔧 Troubleshooting")
                    st.markdown("""
                    Common issues:
                    - Invalid API key
                    - Model name not accessible
                    - Network connection issues
                    - garak installation problems
                    
                    Try running the 'Test garak Installation' button to verify your setup.
                    """)
            
            finally:
                # Restore original environment variable if it was changed
                if api_key and model_config.get("env_var"):
                    if original_env:
                        os.environ[model_config["env_var"]] = original_env
                    else:
                        os.environ.pop(model_config["env_var"], None)

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
        
        # Parse the output to extract model names
        models = []
        for line in result.stdout.strip().split('\n'):
            if line and not line.startswith('NAME'):  # Skip header line
                parts = line.split()
                if parts:
                    models.append(parts[0])
        
        if models:
            logger.info(f"Found {len(models)} Ollama models: {', '.join(models)}")
            return models
        else:
            logger.warning("No Ollama models found")
            return ["llama3", "llama2", "mistral", "mixtral"]  # Default models as fallback
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error checking Ollama models: {str(e)}")
        return ["llama3", "llama2", "mistral", "mixtral"]  # Default models as fallback