use textwrap::dedent;

pub enum Prompts {
    VisionECommerce,
    VisionGeneric,
}


pub fn get_prompt(prompt: Prompts) -> String {
    match prompt {
        Prompts::VisionECommerce => {
            dedent("\
                Describe the product shown in the image. Describe its mood, colors, when would you use/wear it.\
            ")
        },
        Prompts::VisionGeneric => {
            dedent("\
                What is shown in this image? Write a detailed response analyzing the scene.\
            ")
        }
    }
}
