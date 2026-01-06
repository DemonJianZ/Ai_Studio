import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from services.genai_client import init_client
from core.logging import sys_logger

def create_app() -> FastAPI:
    app = FastAPI(title="BananaFlow - 电商智能图像工作台", version="3.3")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request.state.req_id = str(uuid.uuid4())[:8]
        return await call_next(request)

    app.include_router(router)

    # init client on startup
    init_client()
    return app

# bananaflow/app_factory.py (片段)

