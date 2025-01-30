import json
import time
from typing import Iterator
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


class PartyPlanner:
    def __init__(self, config: OramaAIConfig, models_service: ModelsManager):
        self.config = config
        self.models_service = models_service
        self.act = Actions(config)

    def run(self, collection_id: str, input: str) -> Iterator[str]:
        action_plan = self.models_service.chat("party_planner", [], input, DEFAULT_PARTY_PLANNER_ACTIONS)
        action_plan_json = json.loads(repair_json(action_plan))

        yield repair_json(action_plan)

        actions = action_plan_json["actions"]

        history = []
        steps = {}

        for action in actions:
            step_name = action["step"]
            is_orama_step = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]["side"] == EXECUTION_SIDE_ORAMACORE
            returns_json = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]["returns"] == RETURN_TYPE_TEXT

            if not is_orama_step:
                result = self.models_service.action(step_name, input, action["description"], history)

                if returns_json:
                    result = json.dumps(result)

                steps[step_name] = result
                history.append({"role": "assistant", "content": result})
                yield result

            else:
                if step_name == "PERFORM_ORAMA_SEARCH":
                    result = self.act.call_oramacore_search(collection_id, {"term": "foo"})
                    steps[step_name] = result
                    yield result

                else:
                    yield f"Skipping action {step_name} as it requires a missing OramaCore integration"
