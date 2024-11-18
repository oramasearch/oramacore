pub const QUESTIONS_GENERATION_SYSTEM_PROMPT: &str = r#"
        Pretend you're a user searching on Google, a forum, or a blog. Your task is to generate a list of questions that relates to the the context (### Context).

        For example, if the context was the following:

        ```
        At Orama, we specialize in edge-application development. This allows us to build high-performance, low-latency applications distributed via global CDNs. In other words, we prioritize performance and security when developing software.
        ```

        Valid questions would look like the following:

        ```json
        ["What does Orama specialize on?", "Is Orama a low-latency edge application?", "Do Orama prioritize security when developing software?"]
        ```

        Reply with a valid array of strings in a JSON format and nothing more.
    "#;

pub fn get_questions_generation_prompt(context: String) -> String {
    format!("### Context\n\n{}", context)
}
