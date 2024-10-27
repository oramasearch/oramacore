use textwrap::dedent;

#[derive(Clone)]
pub enum Prompts {
    VisionECommerce,
    VisionGeneric,
    VisionTechDocumentation
}


pub fn get_prompt(prompt: Prompts) -> String {
    match prompt {
        Prompts::VisionECommerce => {
            "Describe the product shown in the image. Describe its mood, colors, when would you use/wear it.".to_string()
        },
        Prompts::VisionGeneric => {
            "What is shown in this image? Write a detailed response analyzing the scene.".to_string()
        },
        Prompts::VisionTechDocumentation => {
            "Describe what is shown in this image. For context, it's coming from a technical documentation.".to_string()
        }
    }
}
