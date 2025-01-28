import json
from json_repair import repair_json
from src.utils import OramaAIConfig
from src.service.models import ModelsManager
from src.prompts.party_planner import DEFAULT_PARTY_PLANNER_ACTIONS
from src.prompts.party_planner_actions import DEFAULT_PARTY_PLANNER_ACTIONS_DATA

config = OramaAIConfig()
models_service = ModelsManager(config)

INPUT = "Is Orama open-source?"


def print_json(data):
    json_data = json.loads(repair_json(data))
    print(json.dumps(json_data, indent=2))
    return json_data


action_plan = print_json(models_service.chat("party_planner", [], INPUT, DEFAULT_PARTY_PLANNER_ACTIONS))

actions = action_plan["actions"]

for action in actions:
    step_name = action["step"]
    is_orama_step = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]["side"] == "ORAMACORE"
    returns_json = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]["returns"] == "TEXT"

    if not is_orama_step:
        if returns_json:
            print_json(models_service.action(action["step"], INPUT, action["description"], []))
        print(models_service.action(action["step"], INPUT, action["description"], []))
    else:
        print(f'Skipping action {action["step"]} as it requires an OramaCore integration')
