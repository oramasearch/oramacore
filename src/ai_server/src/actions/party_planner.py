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

        yield Message("ACTION_PLAN", repair_json(action_plan)).to_json()

        actions = action_plan_json["actions"]

        history = []
        steps = {}

        for action in actions:
            step_name = action["step"]
            is_orama_step = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]["side"] == EXECUTION_SIDE_ORAMACORE
            returns_json = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]["returns"] == RETURN_TYPE_TEXT

            if not is_orama_step:
                result = self.models_service.action(step_name, input, action["description"], history)

                steps[step_name] = result
                history.append({"role": "assistant", "content": result})

                yield Message(step_name, result).to_json()

            else:
                if step_name == "PERFORM_ORAMA_SEARCH":
                    result = self.act.call_oramacore_search(collection_id, {"term": "shoes", "mode": "vector"})
                    steps[step_name] = result
                    yield Message(step_name, result).to_json()

                else:
                    yield Message(
                        step_name, f"Skipping action {step_name} as it requires a missing OramaCore integration"
                    ).to_json()
