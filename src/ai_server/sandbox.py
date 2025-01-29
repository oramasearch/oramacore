import json
import time
from datetime import timedelta
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

config = OramaAIConfig()
models_service = ModelsManager(config)
act = Actions(config)

INPUT = "I can't run Orama on Deno 'cause it gives me an error where JS uses 'assert type' instead of 'with'. How do I fix this?"


def print_json(data):
    json_data = json.loads(repair_json(data))
    print(json.dumps(json_data, indent=2))
    return json_data


action_plan = print_json(models_service.chat("party_planner", [], INPUT, DEFAULT_PARTY_PLANNER_ACTIONS))

actions = action_plan["actions"]

history = []
steps = {}

for action in actions:
    time_start = time.perf_counter()

    step_name = action["step"]
    is_orama_step = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]["side"] == EXECUTION_SIDE_ORAMACORE
    returns_json = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[step_name]["returns"] == RETURN_TYPE_TEXT

    if not is_orama_step:
        result = models_service.action(step_name, INPUT, action["description"], history)

        if returns_json:
            json_data = print_json(result)
            result = json.dumps(result)
        else:
            print(result)

        steps[step_name] = result
        history.append({"role": "assistant", "content": result})
    else:
        if step_name == "PERFORM_ORAMA_SEARCH":
            # result = act.call_oramacore_search("nike-data", { "term": "foo" })
            # steps[step_name] = result
            print("================")
            print(f"Skipping action {step_name} as it requires a missing OramaCore integration")
            print("================")
        else:
            print("================")
            print(f"Skipping action {step_name} as it requires a missing OramaCore integration")
            print("================")

    time_end = time.perf_counter()
    elapsed_ms = (time_end - time_start) * 1000
    print(f"Action {step_name} took {elapsed_ms:.2f} ms")

print("================")
print(" STEPS ")
print("================")
# print(json.dumps(history, indent=2))
print(json.dumps(steps, indent=2))
