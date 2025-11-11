import json
from typing import Dict, List, Any
from pydantic import BaseModel, Field, ValidationError


class SearchInput(BaseModel):
    """Input parameters for search tool."""

    term: str = Field(description="The search term to look for")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")
    mode: str = Field(default="fulltext", description="Search mode: 'fulltext', 'vector', or 'hybrid'")


class NlpSearchInput(BaseModel):
    """Input parameters for NLP search tool."""

    query: str = Field(description="Natural language query. Example: 'Find highly rated action movies from the 1990s'")


class MCPServer:
    def __init__(self, search_service, collection_description: str):
        self.search_service = search_service
        self.collection_description = collection_description
        self._tools = self._build_tools()

    def _build_tools(self) -> Dict[str, Any]:
        return {
            "search": {
                "name": "search",
                "description": f"Perform a full-text, vector, or hybrid search on {self.collection_description}",
                "inputSchema": SearchInput.model_json_schema(),
                "func": self._search,
            },
            "nlp_search": {
                "name": "nlp_search",
                "description": f"Perform natural language search on {self.collection_description}",
                "inputSchema": NlpSearchInput.model_json_schema(),
                "func": self._nlp_search,
            },
        }

    def _search(self, term: str, limit: int = 10, mode: str = "fulltext") -> Dict[str, Any]:
        inputs = SearchInput(term=term, limit=limit, mode=mode)

        search_params = {
            "mode": inputs.mode,
            "term": inputs.term,
            "limit": inputs.limit,
            "offset": 0,
            "properties": "*",
        }

        if inputs.mode == "vector":
            search_params["similarity"] = 0.6
        elif inputs.mode == "hybrid":
            search_params["similarity"] = 0.6
            search_params["threshold"] = 1.0
            search_params["exact"] = False
        else:
            search_params["exact"] = False

        result_json = self.search_service.search_json(json.dumps(search_params))
        return json.loads(result_json)

    def _nlp_search(self, query: str) -> List[Any]:
        inputs = NlpSearchInput(query=query)
        return self.search_service.nlp_search({"query": inputs.query})

    def handle_jsonrpc_request(self, request_str: str) -> str:
        try:
            request = json.loads(request_str)
        except json.JSONDecodeError:
            return self._error_response(None, -32700, "Parse error")

        if request.get("jsonrpc") != "2.0":
            return self._error_response(request.get("id"), -32600, "Invalid Request. JSON-RPC version must be 2.0")

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "tools/list":
                result = self._handle_tools_list()
            elif method == "tools/call":
                result = self._handle_tools_call(params)
            else:
                return self._error_response(request_id, -32601, "Method not found")

            if request_id is None:
                return json.dumps({"jsonrpc": "2.0", "id": None})

            return self._success_response(request_id, result)

        except Exception as e:
            return self._error_response(request_id, -32603, f"Internal error: {str(e)}")

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "orama-mcp", "version": "1.0.0"},
        }

    def _handle_tools_list(self) -> Dict[str, Any]:
        tools = []
        for tool_info in self._tools.values():
            tools.append(
                {
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "inputSchema": tool_info["inputSchema"],
                }
            )
        return {"tools": tools}

    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tools/call request."""
        tool_name = params.get("name")
        if not tool_name:
            raise ValueError("Missing tool name")

        arguments = params.get("arguments", {})

        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool_func = self._tools[tool_name]["func"]

        try:
            result = tool_func(**arguments)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
        except ValidationError as e:
            raise ValueError(f"Invalid arguments: {str(e)}")
        except TypeError as e:
            raise ValueError(f"Invalid arguments: {str(e)}")

    def _success_response(self, request_id: Any, result: Any) -> str:
        return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _error_response(self, request_id: Any, code: int, message: str) -> str:
        return json.dumps({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


MCP = MCPServer
