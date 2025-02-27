from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tempfile
from article_summarizer import ArticleSummarizer  # Import from our module

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize your summarizer with environment variables or defaults
#If applicable you can enter your own API keys to your favoruite LLM.

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/chat")

summarizer = ArticleSummarizer(
    model_name=LLM_MODEL,
    api_url=LLM_API_URL,
    provider=LLM_PROVIDER,
    api_key=LLM_API_KEY
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize_article():
    if request.method == 'POST':
        source_type = request.form.get('source_type', 'url')

        if source_type == 'url':
            url = request.form.get('url', '')
            if not url:
                flash('Please enter a valid URL')
                return redirect(url_for('index'))

            # Fix arXiv URL if needed
            if 'arxiv.org' in url and not url.startswith(('http://', 'https://')):
                url = f"https://arxiv.org/abs/{url}"

            source = url

        elif source_type == 'file':
            # Check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(url_for('index'))

            file = request.files['file']

            # If user does not select file, browser submits an empty file
            if file.filename == '':
                flash('No selected file')
                return redirect(url_for('index'))

            if file and file.filename.lower().endswith('.pdf'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                source = filepath
            else:
                flash('Only PDF files are supported')
                return redirect(url_for('index'))
        else:
            flash('Invalid source type')
            return redirect(url_for('index'))

        # Get summary preferences
        length = request.form.get('length', 'medium')
        format_type = request.form.get('format', 'bullet')

        try:
            # Call our summarizer
            result = summarizer.summarize(source, length, format_type)

            # Clean up temporary file if one was created
            if source_type == 'file' and os.path.exists(source):
                os.remove(source)

            return render_template('summary.html', result=result)

        except Exception as e:
            error_message = str(e)
            if "404" in error_message and "arxiv.org" in source:
                flash(f'Error: The arXiv paper ID or URL appears to be invalid. Please check and try again.')
            else:
                flash(f'Error: {error_message}')
            return redirect(url_for('index'))

    return redirect(url_for('index'))


@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    """API endpoint for summarization"""
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data provided"}), 400

        source = data.get('source')
        if not source:
            return jsonify({"error": "No source URL or file path provided"}), 400

        length = data.get('length', 'medium')
        format_type = data.get('format', 'bullet')

        result = summarizer.summarize(source, length, format_type)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/models')
def list_models():
    """List available models (this would need to be implemented based on your LLM API)"""
    # This is a placeholder - you would need to implement this based on your LLM API
    try:
        # Example for Ollama
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json().get("models", [])
        return jsonify({"models": models})
    except:
        # Default models if API is not available
        return jsonify({"models": ["llama3.2", "mistral"]})


if __name__ == '__main__':
    app.run(debug=True)