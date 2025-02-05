import json
from textwrap import dedent
from dataclasses import dataclass
from typing import Iterator, List, Dict, Any, Optional
from json_repair import repair_json

from src.utils import OramaAIConfig, json_to_md
from src.actions.main import Actions
from src.service.models import ModelsManager
from src.prompts.party_planner import DEFAULT_PARTY_PLANNER_ACTIONS
from src.prompts.party_planner_actions import (
    DEFAULT_PARTY_PLANNER_ACTIONS_DATA,
    RETURN_TYPE_JSON,
    EXECUTION_SIDE_ORAMACORE,
)

PARTY_PLANNER_SYSTEM_PROMPT = dedent(
    """You're a useful AI assistant. You have access to a vector database and full-text search engine to perform a number of tasks.
       You should follow them one by one to complete the task successfully.
    """
)


@dataclass
class Message:
    action: str | List[Dict[str, Any]]
    result: str

    def to_json(self) -> str:
        return json.dumps({"action": self.action, "result": self.result})


@dataclass
class Step:
    name: str
    description: str
    is_orama_step: bool
    returns_json: bool
    should_stream: bool


def format_action_plan_assistant(action_plan: List[Dict[str, Any]]) -> str:
    as_md = json_to_md(action_plan)
    return dedent(
        f"""
    Alright! Here's the action plan I've come up with based on your request:
    
    {as_md}.

    Ask me to proceed with the next step, one step at a time when you're ready.
    """
    )


def format_orama_search_results_assistant(results: List[Dict[str, Any]]) -> str:
    sources = ""

    try:
        sources = json_to_md(results, 2)
        print("Sources translation to md failed. Falling back to JSON")
    except Exception as e:
        sources = json.dumps(results)

    return dedent(
        f"""
    Here are the search results I found for you:

    {sources}
    """
    )


class PartyPlanner:
    def __init__(
        self,
        config: OramaAIConfig,
        models_service: ModelsManager,
        history: Optional[List[Any]] = None,
    ):
        self.config = config
        self.models_service = models_service
        self.act = Actions(config)
        self.executed_steps: List[Message] = []
        self.history: List[Any] = history if history is not None else []

    def _get_action_plan(self, input: str) -> List[Dict[str, Any]]:
        action_plan = self.models_service.chat(
            model_id="party_planner",
            history=[],
            prompt=input,
            context=json.dumps(DEFAULT_PARTY_PLANNER_ACTIONS),
        )

        repaired_json: str = repair_json(action_plan)  # type: ignore
        json_action_plan = json.loads(repaired_json)

        # LLMs can be unpredictable and they may return slightly different formats.
        # Usually, they return a dictionary with an "actions" key, but sometimes they return a list.
        #
        # When it follows the instructions, it returns the following format:
        # {
        #     "actions": [
        #         {
        #             "step": "GENERATE_QUERIES",
        #             "description": "Generate queries based on the input."
        #         },
        #         {
        #             ...
        #         }
        #     ]
        # }
        #
        # Sometimes it'll just return a list of actions:
        # [
        #     {
        #         "step": "GENERATE_QUERIES",
        #         "description": "Generate queries based on the input."
        #     },
        #     {
        #         ...
        #     }
        # ]
        #
        # We need to handle both cases.

        if isinstance(json_action_plan, dict) and "actions" in json_action_plan:
            return json_action_plan["actions"]
        elif isinstance(json_action_plan, list):
            return json_action_plan
        else:
            raise ValueError("Invalid action plan format")

    def _create_step(self, action: Dict[str, str]) -> Step:
        step_name = action["step"]
        step_config = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]
        return Step(
            name=step_name,
            description=action["description"],
            is_orama_step=step_config["side"] == EXECUTION_SIDE_ORAMACORE,
            returns_json=step_config["returns"] == RETURN_TYPE_JSON,
            should_stream=step_config["stream"],
        )

    def _execute_orama_search(self, collection_id: str, input: str, api_key: str) -> str:
        # Look for a prior step that produced queries; otherwise, use the original, unoptimized input.
        queries = None
        for step in self.executed_steps:
            if step.action == "GENERATE_QUERIES":
                queries = step.result
                break
            elif step.action == "OPTIMIZE_QUERY":
                queries = step.result
                break

        # If no relevant step was found, default to the input.
        if queries is None:
            queries = [input]
        else:
            # Ensure queries is a list; if it's a JSON string representing a list, parse it.
            if isinstance(queries, str):
                try:
                    parsed = json.loads(queries)
                    if isinstance(parsed, list):
                        queries = parsed
                    else:
                        queries = [queries]
                except json.JSONDecodeError:
                    queries = [queries]

        results = []
        limit = 3 if len(queries) > 1 else 5

        for query in queries:
            # Use "vector" mode if no queries were generated; otherwise, use "hybrid".
            # This is to ensure that vector search is used when there is no query optimization.
            mode = "vector" if queries is None else "hybrid"
            full_query = {"term": query, "mode": mode, "limit": limit}
            res = self.act.call_oramacore_search(collection_id=collection_id, query=full_query, api_key=api_key)
            results.append(res)

        return json.dumps(results)

    def _handle_orama_step(self, step: Step, collection_id: str, input: str, api_key: str) -> str:
        if step.name == "PERFORM_ORAMA_SEARCH":
            try:
                result = self._execute_orama_search(collection_id=collection_id, input=input, api_key=api_key)
                self.history.append(
                    {"role": "assistant", "content": format_orama_search_results_assistant(json_to_md(result, 2))}  # type: ignore
                )

                return result
            except Exception as e:
                return json.dumps({"error": str(e)})
        return json.dumps({"message": f"Skipping action {step.name} as it requires a missing OramaCore integration"})

    def run(self, collection_id: str, input: str, api_key: str) -> Iterator[str]:
        # Add a system prompt to the history if the first entry is not a system prompt.
        if len(self.history) > 0 and self.history[0]["role"] != "system":
            self.history.insert(0, {"role": "system", "content": PARTY_PLANNER_SYSTEM_PROMPT})
        elif len(self.history) == 0:
            self.history.append({"role": "system", "content": PARTY_PLANNER_SYSTEM_PROMPT})

        # Use the input as the first history entry.
        self.history.append({"role": "user", "content": input})

        # Create an action plan and store it in the executed steps.
        action_plan = self._get_action_plan(input)
        self.history.append({"role": "assistant", "content": format_action_plan_assistant(action_plan)})

        for action in action_plan:
            self.history.append({"role": "user", "content": action["description"]})
            step = self._create_step(action)

            # Handle Orama-specific steps first. These should never be streamed.
            if step.is_orama_step:
                # History is managed internally for Orama steps.
                result = self._handle_orama_step(step=step, collection_id=collection_id, input=input, api_key=api_key)
                yield result

            # Handle non-streaming and streaming steps.
            elif not step.should_stream:
                result = self.models_service.action(
                    action=step.name,
                    input=input,
                    description=step.description,
                    history=self.history,
                )
                self.history.append({"role": "assistant", "content": result})
                yield result

            # For streaming steps, yield each chunk.
            else:
                acc_result = ""
                for chunk in self.models_service.action_stream(
                    action=step.name,
                    input=input,
                    description=step.description,
                    history=self.history,
                ):
                    yield chunk
                    acc_result += chunk

                self.history.append({"role": "assistant", "content": acc_result})
