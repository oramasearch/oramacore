import json
from dataclasses import dataclass
from typing import Iterator, List
from json_repair import repair_json

from src.utils import OramaAIConfig
from src.actions.main import Actions
from src.service.models import ModelsManager
from src.prompts.party_planner import DEFAULT_PARTY_PLANNER_ACTIONS
from src.prompts.party_planner_actions import (
    DEFAULT_PARTY_PLANNER_ACTIONS_DATA,
    RETURN_TYPE_TEXT,
    EXECUTION_SIDE_ORAMACORE,
)


@dataclass
class Message:
    action: str
    result: str

    def to_json(self) -> str:
        return json.dumps({"action": self.action, "result": self.result})


class PartyPlanner:
    def __init__(self, config: OramaAIConfig, models_service: ModelsManager):
        self.config = config
        self.models_service = models_service
        self.act = Actions(config)

    def run(self, collection_id: str, input: str, history: List[any]) -> Iterator[str]:
        action_plan = self.models_service.chat("party_planner", history, input, DEFAULT_PARTY_PLANNER_ACTIONS)
        action_plan_json = json.loads(repair_json(action_plan))

        yield Message("ACTION_PLAN", json.dumps(action_plan_json)).to_json()

        actions = action_plan_json["actions"]
        history = []
        steps = {}

        for action in actions:
            step_name = action["step"]
            step = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]
            is_orama_step = step["side"] == EXECUTION_SIDE_ORAMACORE
            returns_json = step["returns"] == RETURN_TYPE_TEXT
            should_stream = step["stream"]

            if not is_orama_step:
                steps[step_name] = "" if should_stream else None

                if not should_stream:
                    result = self.models_service.action(
                        action=step_name,
                        input=input,
                        description=action["description"],
                        history=history,
                        stream=should_stream,
                    )

                    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
                        result = "".join(result)

                    steps[step_name] = result
                    history.append({"role": "assistant", "content": result})

                    if returns_json:
                        try:
                            json.loads(result)
                            yield Message(step_name, result).to_json()
                        except json.JSONDecodeError:
                            yield Message(step_name, json.dumps({"error": "Invalid JSON generated"})).to_json()
                    else:
                        yield Message(step_name, result).to_json()

                else:
                    for chunk in self.models_service.action(
                        action=step_name,
                        input=input,
                        description=action["description"],
                        history=history,
                        stream=should_stream,
                    ):
                        steps[step_name] += chunk
                        yield Message(step_name, steps[step_name]).to_json()

                    history.append({"role": "assistant", "content": steps[step_name]})

            else:
                if step_name == "PERFORM_ORAMA_SEARCH":
                    try:
                        result = self.act.call_oramacore_search(collection_id, {"term": input, "mode": "vector"})
                        result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                        steps[step_name] = result_str
                        yield Message(step_name, result_str).to_json()
                    except Exception as e:
                        yield Message(step_name, json.dumps({"error": str(e)})).to_json()
                else:
                    yield Message(
                        step_name,
                        json.dumps(
                            {"message": f"Skipping action {step_name} as it requires a missing OramaCore integration"}
                        ),
                    ).to_json()
