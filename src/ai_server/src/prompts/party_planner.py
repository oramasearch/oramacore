from typing import List
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
        "description": "Perform full-text, vector, or hybrid search on your index to get quality results that are relevant to the inquiry. Always run this after GENERATE_QUERIES or OPTIMIZE_QUERY.",
    },
    {
        "step": "DESCRIBE_INPUT_CODE",
        "description": "Describe the input code snippet to understand its purpose and functionality. This should only be used when the input includes some code.",
    },
    {
        "step": "IMPROVE_INPUT",
        "description": "If the user requires it, improve the input provided by the user. Correct wording, phrasing, code, or anything else you find necessary. But you should only do this when the user is asking for it specifically.",
    },
    {
        "step": "CREATE_CODE",
        "description": "Create a code snippet. It can be a solution to a problem, a code example, or a code snippet to test a library.",
    },
    {
        "step": "ASK_FOLLOWUP",
        "description": "Ask follow-up questions to clarify the inquiry or gather more information. To be used when the user question is not clear.",
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
