"""
FastAPI application.

Exposes health check and prediction endpoints for a Document AI MVP.
Keeps the same endpoint names as the original mlops-production-pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
from fastapi.responses import HTMLResponse
from mlops_api.predict import predict
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document QA API (Mini-Prod)",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

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

