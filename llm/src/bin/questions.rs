use futures::executor::block_on;
use llm::questions_generation::lib::generate_questions;
use textwrap::dedent;

fn main() {
    let context = dedent(
        r"
        One important concept to understand when working with Orama Cloud, is the concept of data source.
        A data source, as the name suggests, refers to the original source from which data is derived.
        This could be a database, a web service, a JSON file, or any other platform that handles and stores your data.
        Connecting your data source to Orama Cloud is the first step in creating an index. An index is a collection of documents that you can search through.",
    );

    let questions = block_on(generate_questions(context)).unwrap();

    dbg!(questions);
}
