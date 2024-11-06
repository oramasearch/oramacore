use futures::executor::block_on;
use llm::questions_generation::generator::generate_questions;
use textwrap::dedent;

fn main() {
    let context = dedent(
        r"
            Introduction
            When we say that Orama Cloud is batteries-included, we mean that it gives you everything you need to start searching and generating answers (conversations) without any complex configuration. Out of the box, Orama Cloud also includes:
            
            🧩 Native and Custom integrations to easily import your data.
            🚀 Web Components to easily integrate a full-featured Searchbox on your website in no time.
            📊 Quality checks, analytics and quality control tools to fine-tune your users experience.
            🔐 Secure proxy configuration and advanced security options.
            and much more…
            
            Basic concepts
            At the core of Orama Cloud, there are three simple concepts:
            
            📖 Index: a collection of documents that you can search through.
            📄 Schema: a set of rules that define how the documents are structured.
            🗿 Immutability: once you’ve created an index and populated it with documents, it will remain immutable. To change the content of an index, you have to perform a re-deployment.
            With your index, you can perform full-text, vector, and hybrid search queries, as well as generative conversations. Add your data, define the schema, and you’re ready to go!
        ",
    );

    let questions = block_on(generate_questions(context)).unwrap();

    dbg!(questions);
}
