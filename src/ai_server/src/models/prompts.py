from typing import Literal, TypedDict


class PromptContext(TypedDict):
    """Defines structured prompts for different analysis contexts.

    Each key represents a specific analysis context (e.g., ecommerce, documentation),
    and its value contains the corresponding prompt template.
    """

    vision_ecommerce: str
    vision_generic: str
    vision_tech_documentation: str
    vision_code: str


ContextType = Literal["vision_ecommerce", "vision_generic", "vision_tech_documentation", "vision_code"]

PROMPT_TEMPLATES: PromptContext = {
    "vision_ecommerce": (
        "Describe the product shown in the image. " "Include details about its mood, colors, and potential use cases."
    ),
    "vision_generic": (
        "Provide a detailed analysis of what is shown in this image, " "including key elements and their relationships."
    ),
    "vision_tech_documentation": (
        "Analyze this technical documentation image, " "focusing on its key components and technical details."
    ),
    "vision_code": (
        "Analyze the provided code block, explaining its functionality, "
        "implementation details, and intended purpose."
    ),
}
