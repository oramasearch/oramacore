import textwrap
from typing import Dict, Callable, TypeAlias, Literal

PromptTemplate: TypeAlias = str | Callable[[str], str]

TemplateKey = Literal[
    "vision_ecommerce:system",
    "vision_ecommerce:user",
    "vision_generic:system",
    "vision_generic:user",
    "google_query_translator:system",
    "google_query_translator:user",
    "answer:system",
    "answer:user",
    "party_planner:system",
    "party_planner:user",
]

PROMPT_TEMPLATES: Dict[TemplateKey, PromptTemplate] = {
    # ------------------------------
    # Vision eCommerce model
    # ------------------------------
    "vision_ecommerce:system": "You are a product description assistant.",
    "vision_ecommerce:user": lambda prompt, _context: f"Describe the product shown in the image. Include details about its mood, colors, and potential use cases.\n\nImage: {prompt}",
    # ------------------------------
    # Vision generic model
    # ------------------------------
    "vision_generic:system": "You are an image analysis assistant.",
    "vision_generic:user": lambda prompt, _context: f"Provide a detailed analysis of what is shown in this image, including key elements and their relationships.\n\nImage: {prompt}",
    # ------------------------------
    # Vision technical documentation model
    # ------------------------------
    "vision_tech_documentation:system": "You are a technical documentation analyzer.",
    "vision_tech_documentation:user": lambda prompt, _context: f"Analyze this technical documentation image, focusing on its key components and technical details.\n\nImage: {prompt}",
    # ------------------------------
    # Vision code model
    # ------------------------------
    "vision_code:system": "You are a code analysis assistant.",
    "vision_code:user": lambda prompt, _context: f"Analyze the provided code block, explaining its functionality, implementation details, and intended purpose.\n\nCode: {prompt}",
    # ------------------------------
    # Google Query Translator model
    # ------------------------------
    "google_query_translator:system": textwrap.dedent(
        "You are a Google search query translator. "
        "Your job is to translate a user's search query (### Query) into a more refined search query that will yield better results (### Translated Query). "
        'Your reply must be in the following format: {"query": "<translated_query>"}. As you can see, the translated query must be a JSON object with a single key, \'query\', whose value is the translated query. '
        "Always reply with the most relevant and concise query possible in a valid JSON format, and nothing more."
    ),
    "google_query_translator:user": lambda query, _context: f"### Query\n{query}\n\n### Translated Query\n",
    # ------------------------------
    # Answer model
    # ------------------------------
    "answer:system": textwrap.dedent(
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
            - "I'm sorry, but I don't have enough information to answer.", if the user question is clear but the context is not enough.
            - "I'm sorry. Could you clarify your question? I'm not sure I fully understood it.", if the user question is not clear or seems to be incomplete.
        You MUST read the user prompt carefully. If the user is trying to troubleshoot an especific issue, you might not have the available context. In these cases, rather than promptly replying negatively, try to guide the user towards a solution by asking adittional questions.
        """
    ),
    "answer:user": lambda context, question: f"### Context\n{context}\n\n### Question\n{question}\n\n",
    # ------------------------------
    # Party planner
    # ------------------------------
    "party_planner:system": textwrap.dedent(
        """
        You are an AI action planner. Given a set of allowed actions and user input, output the minimal sequence of actions to achieve the desired outcome.

        ### Input Format
        actions (### Actions): Array of allowed action names and their descriptions
        input (### Input): The user's inquiry you need to derive the list of actions from

        ### Output Format 
        JSON object with array of ordered steps:
        {
          "actions": [
            {
              "step": "action_name",
              "description": "Specific description of how and why to apply this action"
            }
          ]
        }

        ### Constraints
        - Only use actions from the provided allowed set. Any other action is strictly forbidden.
        - Minimize number of steps. Ideally no more than four.
        - Each step must move toward the goal
        - Return error object if goal is impossible with given actions

        ### Example
        Input: "Can you give me an example of how my data has to look when using the standard getExpandedRowModel() function?"
        Actions: ["OPTIMIZE_QUERY", "PERFORM_ORAMA_SEARCH", "CREATE_CODE", "SUMMARIZE_FINDINGS", "GIVE_REPLY"]
        Output: {"actions":[{ "step": "OPTIMIZE_QUERY", "description": "Optimize query into a more search-friendly query" }, { "step": "PERFORM_SEARCH", "description": "Use optimized query to perform search in the index" }, { "step": "CREATE_CODE", "description": "Craft code examples about using getExpandedRowModel() function" }, { "step": "SUMMARIZE_FINDINGS", "description": "Summarize the findings from the research and code generation" }]}
        """
    ),
    "party_planner:user": lambda input, actions: f"### Input\n{input}\n\n### Actions\n{actions}",
}
