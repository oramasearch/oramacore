import asyncio
from functools import wraps, partial
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware

from .middleware import GzipRoute, add_middleware
from .models import create_model_types
from src.embeddings.models import OramaModelInfo


def create_app(service):
    app = FastAPI()
    app.router.route_class = GzipRoute
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

    add_middleware(app, service.config)
    FastApiEmbeddingRequest = create_model_types(service.embeddings_service)

    def route_in_threadpool(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(service.thread_executor, partial(func, *args, **kwargs))

        return wrapper

    @app.get("/health")
    async def health_check():
        return JSONResponse(content={"status": "ok"})

    @app.post("/v1/embeddings")
    @route_in_threadpool
    def post_embedding(request: FastApiEmbeddingRequest):
        embeddings = service.embeddings_service.calculate_embeddings(
            request.input, request.intent.name if request.intent else request.intent, request.model.name
        )
        return JSONResponse(content={"data": [{"object": "embedding", "embedding": e.tolist()} for e in embeddings]})

    @app.post("/v1/embeddings_simple")
    @route_in_threadpool
    def post_embedding_simple(request: FastApiEmbeddingRequest):
        embeddings = service.embeddings_service.calculate_embeddings(
            request.input, request.intent.name, request.model.name
        )
        return JSONResponse(
            content={
                "embeddings": [e.tolist() for e in embeddings],
                "dimensions": OramaModelInfo[request.model.name].value["dimensions"],
            }
        )

    return app
