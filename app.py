from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import re
import sys
from io import StringIO
import contextlib
import base64
import io
import traceback
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq client with your API key
client = Groq(api_key="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")  # Replace with your actual Groq API key

# Define MODELS dictionary
MODELS = {
    "Llama3-70B": "llama3-70b-8192",
    "Mixtral-8x7B": "mixtral-8x7b-32768",
    "Gemma-7B": "gemma-7b-it"
}

def safe_execute_code(code: str, df: pd.DataFrame):
    """Execute generated code safely with restricted environment."""
    code = re.sub(r'pd\.read_csv\(.*?\)', 'df', code)
    
    # Prevent dangerous operations
    forbidden = ['pd.read_csv', 'pd.read_excel', 'open(', 'os.', 'sys.']
    for keyword in forbidden:
        if keyword in code:
            raise ValueError(f"Forbidden operation detected: {keyword}")

    # Set up safe execution environment
    env = {
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'df': df.copy(),
        '__builtins__': {**__builtins__, 'open': None}
    }

    output = StringIO()
    figures = []
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            exec(code, env)
            for i in plt.get_fignums():
                buf = io.BytesIO()
                plt.figure(i).savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                figures.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
                buf.close()
            plt.close('all')
        except Exception as e:
            raise RuntimeError(f"Execution error: {str(e)}")
    
    return output.getvalue(), figures, env

def generate_analysis_code(df: pd.DataFrame, query: str, model: str):
    """Generate analysis code using Groq API."""
    system_prompt = f"""You are analyzing a DataFrame called 'df' with columns: {list(df.columns)}
    
    Important Rules:
    1. Use the existing 'df' variable - DO NOT load data from files
    2. Never use pd.read_csv() or any file loading functions
    3. Create visualizations using plt.show()
    4. Include proper error handling
    
    Example:
    ```python
    # Data cleaning
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except:
        pass
    
    # Visualization
    plt.figure(figsize=(10,6))
    sns.countplot(x='Category', data=df)
    plt.show()
    ```"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        model=model,
        temperature=0.3,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle GET and POST requests for the main route."""
    if request.method == 'POST':
        try:
            logger.info("Processing POST request")
            file = request.files.get('csv_file')
            model = request.form.get('model')
            query = request.form.get('query')

            # Validate inputs
            if not file or not model or not query:
                logger.error("Missing required fields: file=%s, model=%s, query=%s", file, model, query)
                return jsonify({'error': 'Missing required fields'}), 400

            # Validate model selection
            if model not in MODELS:
                logger.error("Invalid model selected: %s", model)
                return jsonify({'error': 'Invalid model selected'}), 400

            # Read CSV and generate data previews
            df = pd.read_csv(file)
            preview = df.head(3).to_html(classes='table table-striped')
            dtypes = df.dtypes.astype(str).to_frame('Type').to_html(classes='table table-striped')

            # Generate and clean analysis code
            code_response = generate_analysis_code(df, query, MODELS[model])
            code_match = re.search(r'```python(.*?)```', code_response, re.DOTALL)
            if not code_match:
                logger.error("No valid Python code found in response: %s", code_response)
                return jsonify({'error': 'No valid code found in response'}), 400
            
            clean_code = code_match.group(1).strip()
            clean_code = re.sub(r'pd\.read_csv\(.*?\)', '# Removed file loading', clean_code)
            
            # Execute the code
            output, figures, env = safe_execute_code(clean_code, df)
            
            logger.info("POST request completed successfully")
            return jsonify({
                'preview': preview,
                'dtypes': dtypes,
                'code': clean_code,
                'output': output,
                'figures': figures
            })
        except Exception as e:
            logger.error("POST request failed: %s\n%s", str(e), traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    # Handle GET request
    try:
        logger.info("Rendering template with models: %s", list(MODELS.keys()))
        return render_template('index.html', models=list(MODELS.keys()))
    except Exception as e:
        logger.error("GET request failed: %s\n%s", str(e), traceback.format_exc())
        return "Internal Server Error", 500

if __name__ == '__main__':
    # Run development server only if script is executed directly
    app.run(host='0.0.0.0', port=5000, debug=True)
