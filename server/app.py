from fastapi import FastAPI

from server.routes import router


def create_app() -> FastAPI:
    application = FastAPI(
        title="LLM Serving",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.include_router(router)
    return application


app = create_app()
