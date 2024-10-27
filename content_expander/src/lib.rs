mod prompts;
mod vision;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let example_json: &str = r"
    # Using Hooks
    Functions starting with use are called Hooks. useState is a built-in Hook provided by React. You can find other built-in Hooks in the API reference. You can also write your own Hooks by combining the existing ones.

    Hooks are more restrictive than other functions. You can only call Hooks at the top of your components (or other Hooks). If you want to use useState in a condition or a loop, extract a new component and put it there.

    # Sharing data between components
    In the previous example, each MyButton had its own independent count, and when each button was clicked, only the count for the button clicked changed:

    ![Initially, each MyButton’s count state is 0](https://react.dev/_next/image?url=%2Fimages%2Fdocs%2Fdiagrams%2Fsharing_data_child.dark.png&w=640&q=75)
    ![On click, MyApp updates its count state to 1 and passes it down to both children](https://react.dev/_next/image?url=%2Fimages%2Fdocs%2Fdiagrams%2Fsharing_data_child_clicked.dark.png&w=640&q=75)

    However, often you’ll need components to share data and always update together.
    To make both MyButton components display the same count and update together, you need to move the state from the individual buttons “upwards” to the closest component containing all of them.

    ```jsx
        export default function MyApp() {
          const [count, setCount] = useState(0);

          function handleClick() {
            setCount(count + 1);
          }

          return (
            <div>
              <h1>Counters that update separately</h1>
              <MyButton />
              <MyButton />
            </div>
          );
        }

        function MyButton() {
          // ... we're moving code from here ...
        }
    ```

    Then, pass the state down from MyApp to each MyButton, together with the shared click handler. You can pass information to MyButton using the JSX curly braces, just like you previously did with built-in tags like <img>:
    ";

    let results = vision::describe_images(example_json.to_string(), prompts::Prompts::VisionTechDocumentation).await?;
    dbg!(results);

    Ok(())
}
