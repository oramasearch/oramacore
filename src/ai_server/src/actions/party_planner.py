import json
from dataclasses import dataclass
from typing import Iterator, List, Dict, Any, Optional
from json_repair import repair_json

from src.utils import OramaAIConfig
from src.actions.main import Actions
from src.service.models import ModelsManager
from src.prompts.party_planner import DEFAULT_PARTY_PLANNER_ACTIONS
from src.prompts.party_planner_actions import (
    DEFAULT_PARTY_PLANNER_ACTIONS_DATA,
    RETURN_TYPE_JSON,
    EXECUTION_SIDE_ORAMACORE,
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
        # Avoid mutable default arguments.
        self.history: List[Any] = history if history is not None else []

    def _get_action_plan(self, input: str) -> List[Dict[str, Any]]:
        """Generate and parse the action plan."""
        action_plan = self.models_service.chat(
            model_id="party_planner",
            history=self.history,
            prompt=input,
            context=json.dumps(DEFAULT_PARTY_PLANNER_ACTIONS),
        )
        return json.loads(repair_json(action_plan))["actions"]

    def _create_step(self, action: Dict[str, str]) -> Step:
        """Create a Step object from an action dictionary."""
        step_name = action["step"]
        step_config = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]
        return Step(
            name=step_name,
            description=action["description"],
            is_orama_step=step_config["side"] == EXECUTION_SIDE_ORAMACORE,
            returns_json=step_config["returns"] == RETURN_TYPE_JSON,
            should_stream=step_config["stream"],
        )

    def _execute_orama_search(self, collection_id: str, input: str, api_key: str) -> List[Dict[str, Any]]:
        # Look for a prior step that produced queries; otherwise, use the original input.
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
            full_query = {"term": query, "mode": "hybrid", "limit": limit}
            res = self.act.call_oramacore_search(collection_id=collection_id, query=full_query, api_key=api_key)
            results.append(res)

        return results

    def _handle_orama_step(self, step: Step, collection_id: str, input: str, api_key: str) -> str:
        """Handle Orama-specific steps."""
        if step.name == "PERFORM_ORAMA_SEARCH":
            try:
                result = self._execute_orama_search(collection_id=collection_id, input=input, api_key=api_key)
                # Always return valid JSON.
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e)})
        return json.dumps({"message": f"Skipping action {step.name} as it requires a missing OramaCore integration"})

    def _handle_non_streaming_step(self, step: Step, input: str) -> str:
        """Handle non-streaming model steps."""
        result = self.models_service.action(
            action=step.name,
            input=input,
            description=step.description,
            history=self.history,
        )
        return result

    def _handle_streaming_step(self, step: Step, input: str) -> Iterator[str]:
        """Handle streaming model steps."""
        accumulated_result = ""
        for chunk in self.models_service.action_stream(
            action=step.name,
            input=input,
            description=step.description,
            history=self.history,
        ):
            accumulated_result += chunk
            yield chunk
        # Optionally, yield the final accumulated result if needed.
        yield accumulated_result

    def run(self, collection_id: str, input: str, api_key: str) -> Iterator[str]:
        self.history.append({"role": "user", "content": input})

        action_plan = self._get_action_plan(input)
        message = Message("ACTION_PLAN", action_plan)
        self.executed_steps.append(message)
        yield message.to_json()

        for action in action_plan:
            step = self._create_step(action)

            # Handle Orama-specific steps first
            if step.is_orama_step:
                result = self._handle_orama_step(step=step, collection_id=collection_id, input=input, api_key=api_key)
                message = Message(step.name, result)
                self.executed_steps.append(message)
                yield message.to_json()
                continue

            # Handle non-streaming and streaming steps
            if not step.should_stream:
                result = self._handle_non_streaming_step(step, input)
                message = Message(step.name, result)
                self.executed_steps.append(message)
                yield message.to_json()
            else:
                # For streaming steps, yield each chunk.
                step_result_acc = Message(step.name, "")
                for chunk in self._handle_streaming_step(step, input):
                    # Update the accumulated result
                    step_result_acc.result += chunk
                    yield Message(step.name, chunk).to_json()
                self.executed_steps.append(step_result_acc)

            print("============= history (loop) =============")
            print(json.dumps(self.history, indent=2))
            print("\n\n")

        self.history.append({"role": "assistant", "content": self.executed_steps[-1].result})
        print("============= history =============")
        print(json.dumps(self.history, indent=2))
        print("\n\n")
