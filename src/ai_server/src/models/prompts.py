from typing import Dict, Callable, TypeAlias, Literal

PromptTemplate: TypeAlias = str | Callable[[str], str]

TemplateKey = Literal[
    "vision_ecommerce:system",
    "vision_ecommerce:user",
    "vision_generic:system",
    "vision_generic:user",
    "google_query_translator:system",
    "google_query_translator:user",
]

PROMPT_TEMPLATES: Dict[TemplateKey, PromptTemplate] = {
    # ------------------------------
    # Vision eCommerce model
    # ------------------------------
    "vision_ecommerce:system": "You are a product description assistant.",
    "vision_ecommerce:user": lambda prompt: f"Describe the product shown in the image. Include details about its mood, colors, and potential use cases.\n\nImage: {prompt}",
    # ------------------------------
    # Vision generic model
    # ------------------------------
    "vision_generic:system": "You are an image analysis assistant.",
    "vision_generic:user": lambda prompt: f"Provide a detailed analysis of what is shown in this image, including key elements and their relationships.\n\nImage: {prompt}",
    # ------------------------------
    # Vision technical documentation model
    # ------------------------------
    "vision_tech_documentation:system": "You are a technical documentation analyzer.",
    "vision_tech_documentation:user": lambda prompt: f"Analyze this technical documentation image, focusing on its key components and technical details.\n\nImage: {prompt}",
    # ------------------------------
    # Vision code model
    # ------------------------------
    "vision_code:system": "You are a code analysis assistant.",
    "vision_code:user": lambda prompt: f"Analyze the provided code block, explaining its functionality, implementation details, and intended purpose.\n\nCode: {prompt}",
    # ------------------------------
    # Google Query Translator model
    # ------------------------------
    "google_query_translator:system": (
        "You are a Google search query translator. "
        "Your job is to translate a user's search query (### Query) into a more refined search query that will yield better results (### Translated Query). "
        'Your reply must be in the following format: {"query": "<translated_query>"}. As you can see, the translated query must be a JSON object with a single key, \'query\', whose value is the translated query. '
        "Always reply with the most relevant and concise query possible in a valid JSON format, and nothing more."
    ),
    "google_query_translator:user": lambda query: f"### Query\n{query}\n\n### Translated Query\n",
}
