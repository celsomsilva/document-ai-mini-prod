# document-ai-mini-prod

A minimal **Document AI pipeline** demonstrating how to move from raw documents to a production-ready API that answers questions with **grounded citations**.

This project focuses on the engineering side of AI systems: reproducibility, packaging, testing, and deployment.

This repository is derived from the [mlops-mini-prod project](https://github.com/celsomsilva/mlops-mini-prod), adapting its minimal production structure to a Document AI pipeline.

---

## Live demo

https://document-ai-mini-prod.onrender.com

Documentation and Execution:

https://document-ai-mini-prod.onrender.com/docs

---


## Purpose

Many AI demos stop at notebooks or prompt experiments.

This project shows how to build a **deployable Document AI service** with:

* offline indexing
* retrieval-based inference
* grounded answers
* API service
* Docker container
* automated tests
* CI pipeline

---

## Architecture

```
Document -> Text extraction -> Chunking -> Embeddings / indexing -> Retrieval -> API response with citations
```

The system returns **traceable answers** instead of hallucinated text.

---

## Project structure

```
document-ai-mini-prod/
src/
  mlops_api/
    __init__.py
    api.py              # FastAPI application (/health, /predict)
    train.py            # indexing script that builds document artifacts
    predict.py          # retrieval + answer generation logic

models/                 # saved artifacts (index.json + metadata.json)

tests/                  # automated tests using pytest

Dockerfile              # application container
compose.yaml            # local docker execution
.github/workflows/ci.yaml  # CI pipeline (GitHub Actions)
Makefile                # shortcut commands
requirements.txt        # dependencies
pyproject.toml          # package configuration (src-layout)
README.md
.gitignore
```

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Build index artifacts

```bash
python -m mlops_api.train
```

Artifacts will be created in:

```
models/
  index.json
  metadata.json
```


### Run the API

```bash
uvicorn mlops_api.api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## API usage

Health check:

```bash
GET /health
```

Query documents:

```bash
POST /predict
```

Example request:

```json
{
  "question": "What is the deductible?"
}
```

---

## Docker

Build:

```bash
docker build -t document-ai-mini .
```

Run:

```bash
docker run -p 8000:8000 document-ai-mini
```

---

## Tests

Run:

```bash
pytest -v
```

---

## CI

The repository includes a GitHub Actions workflow that:

* installs dependencies
* installs the package (`pip install -e .`)
* runs tests
* builds the Docker image

---

## Example

Query:

```json
{
  "question": "What is the deductible?"
}
````

Response:

```json
{
  "prediction": "The deductible is $5,000.",
  "model_version": "2026-02-24T00:21:48.730171+00:00",
  "rmse": 0.3849,
  "citations": [
    {
      "doc_id": "sample-policy-001",
      "page": 2,
      "snippet": "Schedule: deductible is $5,000."
    }
  ]
}
```

The system retrieves relevant document chunks and returns both:

* a readable answer
* the source citations

---

## Example queries

{ "question": "What is the deductible?" }
{ "question": "What is the policy number?" }
{ "question": "What coverage does endorsement A1 add?" }
{ "question": "Is flood covered?" }

---

## Notes

This project intentionally avoids complex frameworks and focuses on a clear, reproducible pipeline.

It can easily be extended with:

* LLM answer generation (RAG)
* OCR for scanned documents
* hybrid retrieval
* Azure OpenAI integration

---

## Author

This project was developed by an engineer and data scientist with a background in:

* Postgraduate degree in **Data Science and Analytics (USP)**
* Bachelor of **Science in Electrical and Computer Engineering (UERJ)**
* Special interest in statistical models, interpretability, and applied AI
* Strong interest in algorithmic reasoning, correctness, and performance evaluation

---

## Contact

* [LinkedIn](https://linkedin.com/in/celso-m-silva)
* Or open an [issue](https://github.com/celsomsilva/document-ai-mini-prod/issues)
