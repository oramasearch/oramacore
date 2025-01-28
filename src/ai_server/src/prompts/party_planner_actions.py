from textwrap import dedent

COMMON_USER_PROMPT = lambda input, description: dedent(
    f"""
            ### Input
            {input}

            ### Description
            {description}
            """
)

DEFAULT_PARTY_PLANNER_ACTIONS_DATA = {
    "OPTIMIZE_QUERY": {
        "side": "PYTHON",
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
        "returns": "JSON",
    },
    "GENERATE_QUERIES": {
        "side": "PYTHON",
        "prompt:system": dedent(
            """
            You are an AI assistant. Your job is to create one or more optimized queries out of a user input.
            These queries must be optimized for searching on search engines like Google or similar.
            You'll be given an input (### Input) and a description of what you have to achieve (### Description) when creating optimized queries.

            Reply with a valid JSON that includes optimized queries. It must respect the following format:

            { "queries": ["<optimized-query>", "<optimized-query>"] }

            Reply with a valid JSON only, nothing more.
            """
        ),
        "prompt:user": COMMON_USER_PROMPT,
        "returns": "JSON",
    },
    "PERFORM_ORAMA_SEARCH": {"side": "ORAMACORE", "returns": "JSON"},
    "DESCRIBE_INPUT_CODE": {
        "side": "PYTHON",
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
        "returns": "JSON",
    },
    "IMPROVE_INPUT": {
        "side": "PYTHON",
        "prompt:system": dedent(
            """
            You're an AI assistant. You'll be given a user input (### Input) and a description (### Description) of the task to execute.

            Your job is to improve the input following the instructions provided in the description (### Description) field.
            
            Reply in plain text.
            """
        ),
        "prompt:user": COMMON_USER_PROMPT,
        "returns": "TEXT",
    },
    "CREATE_CODE": {
        "side": "PYTHON",
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
    },
    "SUMMARIZE_FINDINGS": {
        "side": "PYTHON",
        "prompt:system": dedent(
            """
            You're an AI assistant. Your job is to summarize the findings you'll be given (### Input) following a description (### Description) that will give your direction on how to summarize them.

            Be brief but exhaustive. Reply in plain text.
            """
        ),
        "prompt:user": COMMON_USER_PROMPT,
        "returns": "TEXT",
    },
    "ASK_FOLLOWUP": {
        "side": "PYTHON",
        "prompt:system": dedent(
            """
            You're an AI assistant. The user has asked a question (### Input) that you may not have understood completely.
            Follow the instructions provided (### Description) to ask a follow-up question to prompt the user to clarify their inquiry.
            """
        ),
        "prompt:user": COMMON_USER_PROMPT,
        "returns": "TEXT",
    },
    "GIVE_REPLY": {
        "side": "PYTHON",
        "prompt:system": dedent(
            """
            You're an AI assistant. You'll be given a markdown text with two fields, input (### Input) and context (### Context).

            The input identifies a user inquiry. The context provides all you need to know in order to provide a correct answer.

            Using the context only, provide an answer to the user.

            Reply in plain text.
            """
        ),
        "prompt:user": lambda input, context: f"### Input\n{input}\n\n### Context\n{context}",
        "returns": "TEXT",
    },
}
