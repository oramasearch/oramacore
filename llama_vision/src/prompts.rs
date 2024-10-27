use textwrap::dedent;

pub enum Prompts {
    ECommerce,
    Generic,
}

impl Prompts {
    pub fn get_prompt(prompt: Prompts) -> String {
        match prompt {
            Prompts::ECommerce => {
                dedent("\
                    Describe the product shown in the image. Describe its mood, colors, when would you use/wear it.\
                ")
            },
            Prompts::Generic => {
                dedent("\
                    What is shown in this image? Write a detailed response analyzing the scene.\
                ")
            }
        }
    }
}
