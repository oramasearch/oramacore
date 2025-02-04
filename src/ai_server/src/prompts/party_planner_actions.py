import json
from textwrap import dedent
from typing import Any, Dict

RETURN_TYPE_TEXT = "TEXT"
RETURN_TYPE_JSON = "JSON"

EXECUTION_SIDE_PYTHON = "PYTHON"
EXECUTION_SIDE_ORAMACORE = "ORAMACORE"

COMMON_USER_PROMPT = lambda input, description: dedent(
    f"""
            ### Input
            {input}

            ### Description
            {description}
            """
)


def decode_action_result(action: str, result) -> str:
    as_json = json.loads(result)

    if action == "OPTIMIZE_QUERY":
        return json.dumps(as_json["query"])
    elif action == "GENERATE_QUERIES":
        return json.dumps(as_json["queries"])
    elif action == "CREATE_CODE":
        return json.dumps(as_json["code"])
    else:
        return ""


DEFAULT_PARTY_PLANNER_ACTIONS_DATA = {
    "OPTIMIZE_QUERY": {
        "side": EXECUTION_SIDE_PYTHON,
        "prompt:system": dedent(
            """
            You are an AI assistant. Your job is to optimize a given user input into an optimized query for searching on search engines like Google or similar.
            You'll be given an input (### Input) and a description of what you have to achieve (### Description) when translating the query.

            Reply with an optimized query that could work well on search engines.
            The reply must be a valid JSON in the following format:

            { "query": "<optimized-query>" }

            Reply with a valid JSON only, nothing more.
            """
        ),
        "prompt:user": COMMON_USER_PROMPT,
        "returns": RETURN_TYPE_JSON,
        "stream": False,
    },
    "GENERATE_QUERIES": {
        "side": EXECUTION_SIDE_PYTHON,
        "prompt:system": dedent(
            """
            You are an AI assistant. Your job is to create one or more optimized queries out of a user input.
            These queries must be optimized for searching on search engines like Google or similar.
            You'll be given an input (### Input) and a description of what you have to achieve (### Description) when creating optimized queries.

            Make sure to generate at most three queries. No more than that. Also, these queries must be somewhat different from each other.

            Generate the minimum number of queries you think it's worth generating.

            Reply with a valid JSON that includes optimized queries. It must respect the following format:

            { "queries": ["<optimized-query>", "<optimized-query>"] }

            Reply with a valid JSON only, nothing more.
            """
        ),
        "prompt:user": COMMON_USER_PROMPT,
        "returns": RETURN_TYPE_JSON,
        "stream": False,
    },
    "PERFORM_ORAMA_SEARCH": {
        "side": EXECUTION_SIDE_ORAMACORE,
        "returns": RETURN_TYPE_JSON,
        "stream": False,
    },
    "DESCRIBE_INPUT_CODE": {
        "side": EXECUTION_SIDE_PYTHON,
        "prompt:system": dedent(
            """
            You are an AI assistant. Your job is to describe a given input code in natural language, to extract key features, intent, and errors (if any).

            You'll be given an input (### Input) that represents the user code and a description of what you have to achieve (### Description) when describing the code.

            Reply with a valid JSON that includes the description for the code. It must respect the following format:

            { "description": "<code-description>" }

            For example, if the input is: "What does this error mean? TypeError: 'NoneType' object is not subscriptable", a possible output would be:

            { "description": "The user is facing a TypeError as they're trying to access a None property in a Python dictionary." }

            So your goal is not to provide a solution in this step, but rather give a short, simple, description of the code in the input. 
            
            Reply with a valid JSON only, nothing more.
            """
        ),
        "prompt:user": COMMON_USER_PROMPT,
        "returns": RETURN_TYPE_JSON,
        "stream": False,
    },
    "IMPROVE_INPUT": {
        "side": EXECUTION_SIDE_PYTHON,
        "prompt:system": dedent(
            """
            You're an AI assistant. You'll be given a user input (### Input) and a description (### Description) of the task to execute.

            Your job is to improve the input following the instructions provided in the description (### Description) field.
            
            Reply in plain text.
            """
        ),
        "prompt:user": COMMON_USER_PROMPT,
        "returns": RETURN_TYPE_TEXT,
        "stream": False,
    },
    "CREATE_CODE": {
        "side": EXECUTION_SIDE_PYTHON,
        "prompt:system": dedent(
            """
            You're an AI coding assistant. You'll be given an input (### Input) and a description (### Description), and your job is to follow the instructions in the description to generate some code based on the input.

            You must return a valid JSON object structured the following way:

            {
                "code": "<example-code>"
            }

            Reply with a valid JSON object and nothing more.
            """
        ),
        "prompt:user": COMMON_USER_PROMPT,
        "returns": "JSON",
        "stream": False,
    },
    "ASK_FOLLOWUP": {
        "side": EXECUTION_SIDE_PYTHON,
        "prompt:system": dedent(
            """
            You're an AI assistant. The user has asked a question (### Input) that you may not have understood completely.
            Follow the instructions provided (### Description) to ask a follow-up question to prompt the user to clarify their inquiry.
            """
        ),
        "prompt:user": COMMON_USER_PROMPT,
        "returns": RETURN_TYPE_TEXT,
        "stream": True,
    },
    "GIVE_REPLY": {
        "side": EXECUTION_SIDE_PYTHON,
        "prompt:system": dedent(
            """
            You are a AI support agent. You are helping a user with his question around the product.
		    Your task is to provide a solution to the user's question.
		    You'll be provided a context (### Context) and a question (### Question).

		    RULES TO FOLLOW STRICTLY:

		    You should provide a solution to the user's question based on the context and question.
		    You should provide code snippets, quotes, or any other resource that can help the user, only when you can derive them from the context.
		    You should separate content into paragraphs.
		    You shouldn't put the returning text between quotes.
		    You shouldn't use headers.
		    You shouldn't mention "context" or "question" in your response, just provide the answer. That's very important.

		    You MUST include the language name when providing code snippets.
		    You MUST reply with valid markdown code.
		    You MUST only use the information provided in the context and the question to generate the answer. External information or your own knowledge should be avoided.
		    You MUST say one the following sentences if the context or the conversation history is not enough to provide a solution. Be aware that past messages are considered context:
                - "I'm sorry. Could you clarify your question? I'm not sure I fully understood it.", if the user question is not clear or seems to be incomplete.
            You MUST read the user prompt carefully. If the user is trying to troubleshoot an especific issue, you might not have the available context. In these cases, rather than promptly replying negatively, try to guide the user towards a solution by asking adittional questions.
            """
        ),
        "prompt:user": lambda input, context: f"### Question\n{input}\n\n### Context\n{context}",
        "returns": RETURN_TYPE_TEXT,
        "stream": True,
    },
}
