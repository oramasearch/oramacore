from typing import Dict, List
from dataclasses import dataclass

DEFAULT_PARTY_PLANNER_ACTIONS = [
    {
        "step": "OPTIMIZE_QUERY",
        "description": "Optimize the input query to get better results. It could split the query into multiple queries or use different keywords.",
    },
    {
        "step": "GENERATE_QUERIES",
        "description": "Generate multiple queries based on the input query to get better results. This is useful when the input query is too broad or contains too much information, ambiguous, or unclear terms.",
    },
    {
        "step": "PERFORM_ORAMA_SEARCH",
        "description": "Perform full-text, vector, or hybrid search on your index to get quality results that are relevant to the inquiry.",
    },
    {
        "step": "DESCRIBE_INPUT_CODE",
        "description": "Describe the input code snippet to understand its purpose and functionality.",
    },
    {
        "step": "IMPROVE_INPUT",
        "description": "Improve the input provided by the user. Correct wording, phrasing, or anything else you find necessary.",
    },
    {
        "step": "CREATE_CODE",
        "description": "Create a code snippet. It can be a solution to a problem, a code example, or a code snippet to test a library.",
    },
    {"step": "SUMMARIZE_FINDINGS", "description": "Summarize the findings from the research."},
    {
        "step": "ASK_FOLLOWUP",
        "description": "Ask follow-up questions to clarify the inquiry or gather more information.",
    },
    {
        "step": "GIVE_REPLY",
        "description": "Reply to the inquiry with the findings, solutions, suggestions, or any other relevant information.",
    },
]


@dataclass
class PartyPlannerAction:
    step: str
    description: str


class PartyPlannerActions:
    def __init__(self):
        self.actions: List[PartyPlannerAction] = [
            PartyPlannerAction(**action) for action in DEFAULT_PARTY_PLANNER_ACTIONS
        ]

    def extend_default_actions(self, new_actions: List[PartyPlannerAction]) -> None:
        """Extend default actions with new ones."""
        self.actions.extend(new_actions)

    def get_actions(self) -> List[PartyPlannerAction]:
        return self.actions
