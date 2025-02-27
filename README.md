# ARTICLE SUMMARIZER

![project-image.png](Images%20of%20Project/project-image.png)


This project is an article summarizer that uses a locally working llama3.2 model
to summarize articles whether they are PDFs or URL links.

## Features
- Multiple input sources, arxiv.org, web pages, pdf files.
- Summarizations are customizable, there are 3 pre-defined prompt configurations for the summarization operation.
- Local LLMs or API keys to LLMs can be used.

## Example Usage
- In the following example, a random arXiv.org article is chosen and given to the program, the summary of the article is shown below image. Summary length of medium and summary format of structured is selected.


![example-output.png](Images%20of%20Project/example-output.png)


## Installation

### Prerequisites
- python3.7+
- Ollama for local LLM running. (For tests, llama3.2  2GB version is used.)

### Setup
- Clone the repository
    ```bash
    git clone https://github.com/RhinoCoder/Article-Summarizer.git
    cd Article-Summarizer

- Create and activate a virtual environment.
  ```bash
  python -m venv .venv
  source  .venv/bin/activate

- Install dependencies
   ```bash 
   pip install -r requirements.txt
- Set up Ollama (If you will use local LLM)
  Install from https://ollama.com/
  Choose a model and pull, i.e. ollama pull llama3.2  
  To start model locally run,  
  ```bash
  ollama run llama3.2
  
- If you are going to use an API version, update the .env file accordingly.

### Usage
- From terminal in the project directory, run
    python app.py
Then, open your web browser and go to localhost:5000,
Congratulations you have everything needed set up and ready to use it.

### Future Work
A more robust web & app version is under consideration to be implemented.
Any contribution, code format change and suggestion is welcomed.