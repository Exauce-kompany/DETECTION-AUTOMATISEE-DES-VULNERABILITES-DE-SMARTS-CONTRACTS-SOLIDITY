"""Assemble the HTTP application; importing it performs no database writes."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.persistence.database import init_database
from webapp.routes import analysis, history, reports

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="SMART BUG",
        description="Analyse de sécurité des Smart Contracts Solidity.",
        version="3.0.0",
        lifespan=lifespan,
    )
    application.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    application.include_router(analysis.router)
    application.include_router(history.router)
    application.include_router(reports.router)

    @application.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse(request=request, name="index.html", context={})

    return application


app = create_app()
