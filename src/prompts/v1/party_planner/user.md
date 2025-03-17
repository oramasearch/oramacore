### Input
{input}

### Actions

**OPTIMIZE_QUERY**
Optimize the input query to get better results.
Use this when the query is long, broad, or contains too many irrelevant keywords.
For example, if the user asks "I am having a lot of trouble with installing my Ryzen 9 9900X CPU, I feat the pins are bent", you should optimize the query in "Ryzen CPU bent pins troubleshooting".

**GENERATE_QUERIES**
Generate multiple queries based on the input query to get better results.
This is useful when the input query is too broad or contains too much information, ambiguous, or unclear terms.
For example, if a user asks: "I need a good mouse and a good gaming keyboard", you should return two distinct queries "gaming mouse" and "gaming keyboard".

**PERFORM_ORAMA_SEARCH**
Perform full-text, vector, or hybrid search on your index to get quality results that are relevant to the inquiry.
ALWAYS run this after GENERATE_QUERIES or OPTIMIZE_QUERY.

**DESCRIBE_INPUT_CODE**
Describe the input code snippet to understand its purpose and functionality.
This should only be used when the input includes some programming code.

**IMPROVE_INPUT**
If the user requires it, improve the input provided by the user.
Correct wording, phrasing, code, or anything else you find necessary.
But you MUST only do this when the user is asking for it specifically.
For example, when a user asks "Can you help me correct my english?" or "Can you help me make this code cleaner?"

**CREATE_CODE**
Create a code snippet.
It can be a solution to a problem, a code example, or a code snippet to test a library.

**ASK_FOLLOWUP**
Ask follow-up questions to clarify the inquiry or gather more information.
To be used when the user question is not clear.
If you're choosing this action, no other action must be performed.

**GIVE_REPLY**
Reply to the inquiry with the findings, solutions, suggestions, or any other relevant information.
Always end with this action.