"""
FastAPI application.

Exposes health check and prediction endpoints for a Document AI MVP.
Keeps the same endpoint names as the original mlops-mini-prod.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
from fastapi.responses import HTMLResponse

from mlops_api.predict import predict

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document QA API (Mini-Prod)",
    description="""
This API exposes a minimal Document AI pipeline:

- offline indexing step that saves artifacts to `models/`
- retrieval-based answers with page-level citations (traceability)
- production-ready packaging (src-layout), tests, Docker, CI

### Offline step

Run:

`python -m mlops_api.train`

This generates:

- `models/index.json`
- `models/metadata.json`

### Query

Use POST `/predict` with:

{
  "question": "What is the deductible?",
  "top_k": 3
}

The API returns an answer (`prediction`) and citations with page numbers.
""",
    version="1.0.0",
)


class InputSchema(BaseModel):
    question: str = Field(
        description="User question about the indexed documents",
        example="What is the deductible in the schedule?"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="How many supporting chunks to retrieve",
        example=3
    )
    doc_id: str | None = Field(
        default=None,
        description="Optional document id to restrict retrieval (e.g., PDF filename stem)",
        example="sample-policy-001"
    )


@app.on_event("startup")
def startup_event():
    logger.info("API startup completed")


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>Document QA API (Mini-Prod)</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f4f6f8;
                    color: #333;
                }
                .container {
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    max-width: 760px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                }
                h1 { margin-top: 0; }
                a {
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 15px;
                    background-color: #2563eb;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }
                a:hover { background-color: #1e40af; }
                .meta {
                    margin-top: 20px;
                    font-size: 14px;
                    color: #666;
                }
                code { background: #eef2ff; padding: 2px 4px; border-radius: 4px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Document QA API (Mini-Prod)</h1>
                <p>Document AI inference service with grounded answers and citations.</p>
                <a href="/docs">Open interactive documentation/Execution</a>

                <div class="meta">
                    <p><strong>Pipeline:</strong> offline indexing → retrieval → citations</p>
                    <p><strong>Status:</strong> running</p>
                    <p><strong>Tip:</strong> run <code>python -m mlops_api.train</code> to generate artifacts</p>
                </div>
            </div>
        </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(payload: InputSchema):
    try:
        return predict(payload.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # artifacts missing, etc.
        raise HTTPException(status_code=500, detail=str(e))

