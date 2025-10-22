import json
from typing import Literal, List, Any, Dict
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    term: str = Field(description="The search term to look for")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")
    mode: Literal["fulltext", "vector", "hybrid", "auto"] = Field(
        default="fulltext",
        description="Search mode: 'fulltext' for keyword search, 'vector' for semantic search, 'hybrid' for combined, 'auto' to let the system decide based on your input query.",
    )


class NlpSearchInput(BaseModel):
    query: str = Field(description="Natural language query. Example: 'Find highly rated action movies from the 1990s'")


class MCP:
    def __init__(self, search_service, collection_description: str):
        self.search_service = search_service
        self.collection_description = collection_description
        self._tools: Dict[str, Any] = {}
        self._register_tools()

    def _register_tools(self):
        self._tools["search"] = {
            "name": "search",
            "description": f"Perform a full-text, vector, or hybrid search operation on {self.collection_description}",
            "inputSchema": SearchInput.model_json_schema(),
            "func": self._search,
        }

        self._tools["nlp_search"] = {
            "name": "nlp_search",
            "description": f"Perform complex search queries using natural language on {self.collection_description}. Useful for queries with complex filtering, sorting, or faceting.",
            "inputSchema": NlpSearchInput.model_json_schema(),
            "func": self._nlp_search,
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

        params_json = json.dumps(search_params)
        result_json = self.search_service.search_json(params_json)

        return json.loads(result_json)

    def _nlp_search(self, query: str) -> List[Any]:
        inputs = NlpSearchInput(query=query)
        params = {"query": inputs.query}

        result = self.search_service.nlp_search(params)
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        tools_list = []

        for tool_info in self._tools.values():
            tool_def = {
                "name": tool_info["name"],
                "description": tool_info["description"],
            }

            if "inputSchema" in tool_info:
                tool_def["inputSchema"] = tool_info["inputSchema"]

            tools_list.append(tool_def)

        return tools_list

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool_func = self._tools[tool_name].get("func")
        if not tool_func:
            raise ValueError(f"Tool '{tool_name}' has no callable function")

        try:
            result = tool_func(**arguments)
            return result
        except TypeError as e:
            raise ValueError(f"Invalid arguments for tool '{tool_name}': {e}")
