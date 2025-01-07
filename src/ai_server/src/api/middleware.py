import gzip
from fastapi import Request
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse, Response


class GzipRequest(Request):
    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            body = await super().body()
            if "gzip" in self.headers.getlist("Content-Encoding"):
                body = gzip.decompress(body)
            self._body = body
        return self._body


class GzipRoute(APIRoute):
    def get_route_handler(self):
        original_handler = super().get_route_handler()

        async def custom_handler(request: Request) -> Response:
            request = GzipRequest(request.scope, request.receive)
            return await original_handler(request)

        return custom_handler


def add_middleware(app, config):
    @app.middleware("http")
    async def validate_api_key(request: Request, call_next):
        if config.api_key is None:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if auth_header != f"Bearer {config.api_key}":
            return JSONResponse(status_code=403, content={"error": "Invalid API key"})
        return await call_next(request)

    @app.middleware("http")
    async def handle_errors(request: Request, call_next):
        try:
            return await call_next(request)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            print(f"Fatal error, restarting server: {e}")
            os._exit(129)
