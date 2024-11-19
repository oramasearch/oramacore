#![allow(unused_variables)]

use nlp::chunker::{Chunker, ChunkerConfig};
use types::CodeLanguage;

fn main() {
    let text = r#"
        The 1921 Centre vs. Harvard football game was a regular-season collegiate American football game played on October 29, 1921, at Harvard Stadium in Boston, Massachusetts.
        The contest featured the undefeated Centre Praying Colonels, representing Centre College, and the undefeated Harvard Crimson, representing Harvard University.
        Centre won the game 6–0, despite entering as heavy underdogs, and the game is widely viewed as one of the largest upsets in college football history.
        The game is often referred to by the shorthand C6H0, after a Centre professor's remark that Harvard had been poisoned by this "impossible" chemical formula.

        The teams had met for the first time in the previous year.
        Centre, led by Charley Moran, shocked many by taking a tie into halftime but ultimately Bob Fisher's Harvard squad took control in the second half and won the game.
        Centre played well enough to warrant a rematch the following year, and the Colonels, led by quarterback Bo McMillin and halfback Norris Armstrong, again found themselves tied with the Crimson at halftime.
        Less than two minutes into the game's third quarter, McMillin rushed for a touchdown, giving the visitors a 6–0 lead. The conversion failed but the Centre defense held for the remainder of the game.

        Harvard threatened and even reached the Centre 3-yard line at one point but were unable to score.
        Regaining possession with several minutes remaining in the game, the Praying Colonels ran out the clock to secure a six-point victory and maintain their perfect record.

        The teams first met on October 23, 1920,[6] at Harvard Stadium in the Boston neighborhood of Allston.[19] Entering the game, both teams were undefeated and untied and neither Centre, in three games, nor Harvard, in four, had been scored on.[6][20] Centre's team was called "the scoring machine of the football universe" by The Dayton Herald after totaling 241 points in their first three games combined.[20][21] Attendance was estimated by The Dayton Herald to be at least 37,000 people (and was reported to have been as high as 45,000[22]). Ticket sales were stopped the night before when the contest sold out and as many as 10,000 potential attendees were turned away at the gates as a result.[21][23] In contrast, Centre had never played before a crowd exceeding 8,000 fans.[24] Harvard was favored to win the game with 8-to-5 odds and had, on average, a 22-pound weight advantage over Centre's squad.[21] Centre president William Arthur Ganfield traveled to attend the game and led the team in prayer before they took the field.[21]

        The Praying Colonels surprised many by taking a 14–14 tie into halftime.[19] Harvard scored one touchdown in each quarter, adding a field goal in the third quarter, and held Centre scoreless in the second half to win 31–14.[19] McMillin, Centre's quarterback, finished the game having tallied 151 rushing yards and 131 passing yards.[25] During the game, Harvard used nine of its substitutes while Centre used three.[26] Following the game, Harvard captain Arnold Horween offered the game ball to McMillin, who declined the ball and promised "We'll be back next year to take it home with us."[14] The Boston Globe described the game as the most interesting to watch in Harvard Stadium's history.[27] Centre was praised by The Boston Globe for its resiliency and unwillingness to give up. After the game, the Harvard team hosted Centre's team, coaches, and president for dinner.[27] The visitors earned $6,000 (equivalent to $91,000 in 2023) from the game.[28] Despite this loss, Centre was still seen as a strong team by the sportswriter Fuzzy Woodruff, who said that they entered their next game against Georgia Tech as an "unbeatable team".[29] Despite this, Centre ultimately lost that game 24–0.[30]

        McMillin and captain Norris Armstrong played basketball for Centre in the offseason, during which the Colonels defeated Harvard by five points.[31] McMillin was made a Kentucky Colonel by governor Edwin P. Morrow around the same time.[32]
    "#;

    let code = r#"
        /* eslint-disable @typescript-eslint/no-this-alias */
        import { syncBoundedLevenshtein } from '../components/levenshtein.js'
        import { InternalDocumentID } from '../components/internal-document-id-store.js'
        import { getOwnProperty } from '../utils.js'

        interface FindParams {
          term: string
          exact?: boolean
          tolerance?: number
        }

        export type FindResult = Record<string, InternalDocumentID[]>

        export class RadixNode {
          // Node key
          public k: string
          // Node subword
          public s: string
          // Node children
          public c: Map<string, RadixNode> = new Map()
          // Node documents
          public d: Set<InternalDocumentID> = new Set()
          // Node end
          public e: boolean
          // Node word
          public w = ''

          constructor(key: string, subWord: string, end: boolean) {
            this.k = key
            this.s = subWord
            this.e = end
          }

          public updateParent(parent: RadixNode): void {
            this.w = parent.w + this.s
          }

          public addDocument(docID: InternalDocumentID): void {
            this.d.add(docID)
          }

          public removeDocument(docID: InternalDocumentID): boolean {
            return this.d.delete(docID)
          }

          public findAllWords(output: FindResult, term: string, exact?: boolean, tolerance?: number): FindResult {
            const stack: RadixNode[] = [this]
            while (stack.length > 0) {
              const node = stack.pop()!

              if (node.e) {
                const { w, d: docIDs } = node

                if (exact && w !== term) {
                  continue
                }

                // check if _output[w] exists and then add the doc to it
                // always check in own property to prevent access to inherited properties
                // fix https://github.com/askorama/orama/issues/137
                if (getOwnProperty(output, w) !== null) {
                  if (tolerance) {
                    const difference = Math.abs(term.length - w.length)

                    if (difference <= tolerance && syncBoundedLevenshtein(term, w, tolerance).isBounded) {
                      output[w] = []
                    } else {
                      continue
                    }
                  } else {
                    output[w] = []
                  }
                }

                // check if _output[w] exists and then add the doc to it
                // always check in own property to prevent access to inherited properties
                // fix https://github.com/askorama/orama/issues/137
                if (getOwnProperty(output, w) != null && docIDs.size > 0) {
                  const docs = output[w]
                  for (const docID of docIDs) {
                    if (!docs.includes(docID)) {
                      docs.push(docID)
                    }
                  }
                }
              }

              if (node.c.size > 0) {
                stack.push(...node.c.values())
              }
            }
            return output
          }

          public insert(word: string, docId: InternalDocumentID): void {
            let node: RadixNode = this
            let i = 0
            const wordLength = word.length

            while (i < wordLength) {
              const currentCharacter = word[i]
              const childNode = node.c.get(currentCharacter)

              if (childNode) {
                const edgeLabel = childNode.s
                const edgeLabelLength = edgeLabel.length
                let j = 0

                // Find the common prefix length between edgeLabel and the remaining word
                while (j < edgeLabelLength && i + j < wordLength && edgeLabel[j] === word[i + j]) {
                  j++
                }

                if (j === edgeLabelLength) {
                  // Edge label fully matches; proceed to the child node
                  node = childNode
                  i += j
                  if (i === wordLength) {
                    // The word is a prefix of an existing word
                    if (!childNode.e) {
                      childNode.e = true
                    }
                    childNode.addDocument(docId)
                    return
                  }
                  continue
                }

                // Split the edgeLabel at the common prefix
                const commonPrefix = edgeLabel.slice(0, j)
                const newEdgeLabel = edgeLabel.slice(j)
                const newWordLabel = word.slice(i + j)

                // Create an intermediate node for the common prefix
                const inbetweenNode = new RadixNode(commonPrefix[0], commonPrefix, false)
                node.c.set(commonPrefix[0], inbetweenNode)
                inbetweenNode.updateParent(node)

                // Update the existing childNode
                childNode.s = newEdgeLabel
                childNode.k = newEdgeLabel[0]
                inbetweenNode.c.set(newEdgeLabel[0], childNode)
                childNode.updateParent(inbetweenNode)

                if (newWordLabel) {
                  // Create a new node for the remaining part of the word
                  const newNode = new RadixNode(newWordLabel[0], newWordLabel, true)
                  newNode.addDocument(docId)
                  inbetweenNode.c.set(newWordLabel[0], newNode)
                  newNode.updateParent(inbetweenNode)
                } else {
                  // The word ends at the inbetweenNode
                  inbetweenNode.e = true
                  inbetweenNode.addDocument(docId)
                }
                return
              } else {
                // No matching child; create a new node
                const newNode = new RadixNode(currentCharacter, word.slice(i), true)
                newNode.addDocument(docId)
                node.c.set(currentCharacter, newNode)
                newNode.updateParent(node)
                return
              }
            }

            // If we reach here, the word already exists in the tree
            if (!node.e) {
              node.e = true
            }
            node.addDocument(docId)
          }

          private _findLevenshtein(
            term: string,
            index: number,
            tolerance: number,
            originalTolerance: number,
            output: FindResult
          ) {
            const stack: Array<{ node: RadixNode; index: number; tolerance: number }> = [{ node: this, index, tolerance }]

            while (stack.length > 0) {
              const { node, index, tolerance } = stack.pop()!

              if (node.w.startsWith(term)) {
                node.findAllWords(output, term, false, 0)
                continue
              }

              if (tolerance < 0) {
                continue
              }

              if (node.e) {
                const { w, d: docIDs } = node
                if (w) {
                  if (syncBoundedLevenshtein(term, w, originalTolerance).isBounded) {
                    output[w] = []
                  }
                  if (getOwnProperty(output, w) !== undefined && docIDs.size > 0) {
                    const docs = new Set(output[w])

                    for (const docID of docIDs) {
                      docs.add(docID)
                    }
                    output[w] = Array.from(docs)
                  }
                }
              }

              if (index >= term.length) {
                continue
              }

              const currentChar = term[index]

              // 1. If node has child matching term[index], push { node: childNode, index +1, tolerance }
              if (node.c.has(currentChar)) {
                const childNode = node.c.get(currentChar)!
                stack.push({ node: childNode, index: index + 1, tolerance })
              }

              // 2. Push { node, index +1, tolerance -1 } (Delete operation)
              stack.push({ node: node, index: index + 1, tolerance: tolerance - 1 })

              // 3. For each child:
              for (const [character, childNode] of node.c) {
                // a) Insert operation
                stack.push({ node: childNode, index: index, tolerance: tolerance - 1 })

                // b) Substitute operation
                if (character !== currentChar) {
                  stack.push({ node: childNode, index: index + 1, tolerance: tolerance - 1 })
                }
              }
            }
          }

          public find(params: FindParams): FindResult {
            const { term, exact, tolerance } = params
            if (tolerance && !exact) {
              const output: FindResult = {}
              this._findLevenshtein(term, 0, tolerance, tolerance, output)
              return output
            } else {
              let node: RadixNode = this
              let i = 0
              const termLength = term.length

              while (i < termLength) {
                const character = term[i]
                const childNode = node.c.get(character)

                if (childNode) {
                  const edgeLabel = childNode.s
                  const edgeLabelLength = edgeLabel.length
                  let j = 0

                  // Compare edge label with the term starting from position i
                  while (j < edgeLabelLength && i + j < termLength && edgeLabel[j] === term[i + j]) {
                    j++
                  }

                  if (j === edgeLabelLength) {
                    // Full match of edge label; proceed to the child node
                    node = childNode
                    i += j
                  } else if (i + j === termLength) {
                    // The term ends in the middle of the edge label
                    if (exact) {
                      // Exact match required but term doesn't end at a node
                      return {}
                    } else {
                      // Partial match; collect words starting from this node
                      const output: FindResult = {}
                      childNode.findAllWords(output, term, exact, tolerance)
                      return output
                    }
                  } else {
                    // Mismatch found
                    return {}
                  }
                } else {
                  // No matching child node
                  return {}
                }
              }

              // Term fully matched; collect words starting from this node
              const output: FindResult = {}
              node.findAllWords(output, term, exact, tolerance)
              return output
            }
          }

          public contains(term: string): boolean {
            let node: RadixNode = this
            let i = 0
            const termLength = term.length

            while (i < termLength) {
              const character = term[i]
              const childNode = node.c.get(character)

              if (childNode) {
                const edgeLabel = childNode.s
                const edgeLabelLength = edgeLabel.length
                let j = 0

                while (j < edgeLabelLength && i + j < termLength && edgeLabel[j] === term[i + j]) {
                  j++
                }

                if (j < edgeLabelLength) {
                  return false
                }

                i += edgeLabelLength
                node = childNode
              } else {
                return false
              }
            }
            return true
          }

          public removeWord(term: string): boolean {
            if (!term) {
              return false
            }

            let node: RadixNode = this
            const termLength = term.length
            const stack: { parent: RadixNode; character: string }[] = []
            for (let i = 0; i < termLength; i++) {
              const character = term[i]
              if (node.c.has(character)) {
                const childNode = node.c.get(character)!
                stack.push({ parent: node, character })
                i += childNode.s.length - 1
                node = childNode
              } else {
                return false
              }
            }

            // Remove documents from the node
            node.d.clear()
            node.e = false

            // Clean up any nodes that no longer lead to a word
            while (stack.length > 0 && node.c.size === 0 && !node.e && node.d.size === 0) {
              const { parent, character } = stack.pop()!
              parent.c.delete(character)
              node = parent
            }

            return true
          }

          public removeDocumentByWord(term: string, docID: InternalDocumentID, exact = true): boolean {
            if (!term) {
              return true
            }

            let node: RadixNode = this
            const termLength = term.length
            for (let i = 0; i < termLength; i++) {
              const character = term[i]
              if (node.c.has(character)) {
                const childNode = node.c.get(character)!
                i += childNode.s.length - 1
                node = childNode

                if (exact && node.w !== term) {
                  // Do nothing if the exact condition is not met.
                } else {
                  node.removeDocument(docID)
                }
              } else {
                return false
              }
            }
            return true
          }

          private static getCommonPrefix(a: string, b: string): string {
            const len = Math.min(a.length, b.length)
            let i = 0
            while (i < len && a.charCodeAt(i) === b.charCodeAt(i)) {
              i++
            }
            return a.slice(0, i)
          }

          public toJSON(): object {
            return {
              w: this.w,
              s: this.s,
              e: this.e,
              k: this.k,
              d: Array.from(this.d),
              c: Array.from(this.c?.entries())?.map(([key, node]) => [key, node.toJSON()])
            }
          }

          public static fromJSON(json: any): RadixNode {
            const node = new RadixNode(json.k, json.s, json.e)
            node.w = json.w
            node.d = new Set(json.d)
            node.c = new Map(json?.c?.map(([key, nodeJson]: [string, any]) => [key, RadixNode.fromJSON(nodeJson)]))
            return node
          }
        }

        export class RadixTree extends RadixNode {
          constructor() {
            super('', '', false)
          }

          public static fromJSON(json: any): RadixTree {
            const tree = new RadixTree()
            tree.w = json.w
            tree.s = json.s
            tree.e = json.e
            tree.k = json.k
            tree.d = new Set(json.d)
            tree.c = new Map(json.c?.map(([key, nodeJson]: [string, any]) => [key, RadixNode.fromJSON(nodeJson)]))
            return tree
          }

          public toJSON(): object {
            return super.toJSON()
          }
        }
    "#;

    let mdx = r#"
        ---
        title: Answers Customization
        description: Learn how to customize the answer engine behavior in your Orama index.
        next: false
        ---
        import { Steps } from '@astrojs/starlight/components';

        Orama allows you to define custom instructions (**System Prompts**) that shape how the AI engine responds to users. These prompts can be tailored for various use cases, such as **altering the tone** of the AI, **introducing information** that isn’t in the data sources, or **directing users** toward specific actions.

        For example, if a user asks about pricing, a system prompt could instruct the AI to suggest visiting a website for details, instead of providing a direct answer. You could also include **more context**, like software versions or other specialized information to ensure more accurate responses.

        #### Best Practices and Limitations

        While System Prompts give you powerful control over the AI’s behavior, they are subject to certain validation rules to maintain safety and efficiency. We implemented **jailbreak protection** to guard against malicious or harmful actions. There are also limits on the length and complexity of prompts to ensure optimal performance.

        These safeguards ensure that while you have the freedom to shape the AI’s responses, the assistant will always behave in a reliable, secure, and efficient manner.

        ## Customizing AI Answers

        This guide will help you understand how to customize the answers generated by the AI answer engine in your Orama index.

        First of all, let's create a new index. If this is your first time creating a new index, follow [this guide](/cloud/data-sources/static-files/json-file) to set up your first index.

        <Steps>
        <ol>
            <li>
              <p class="pl-10">Once you have your index set up, you will find yourself in the index overview page. Click on the "**System Prompts**" tab on the left menu to access the system prompts settings.</p>

              <img src="/cloud/guides/custom-system-prompts/new-index.png" alt="New index" class="mx-10" />

              <p class="pl-10">If you're creating your first system prompt, click on "**Create system prompt**" to get started.</p>

              <img src="/cloud/guides/custom-system-prompts/new-prompt.png" alt="New system prompt" class="mx-10" />
            </li>
            <li>
              <p class="pl-10">You can finally customize the system prompt by performing a number of operations, let's break them down:</p>
              <img src="/cloud/guides/custom-system-prompts/customizing-the-prompt.png" alt="Customizing the system prompt" class="mx-10" />
            </li>
            <li>
              <strong class="pl-10">Giving a name to the system prompt</strong>
              <p class="pl-10">Every system prompt should have a name. It should be unique and descriptive so that you can easily identify it later.</p>
            </li>
            <li>
              <strong class="pl-10">Usage method</strong>
              <p class="pl-10">Select how the system prompt should be activated: <strong>Automatic</strong> or <strong>Manual</strong>. See [Usage Methods](#usage-methods) for more details.</p>
            </li>
          </ol>
        </Steps>

        ## Usage Methods

        Orama allows you to use the custom system prompts in a few different ways:

        ### Automatic

        By default, when giving an answer, Orama will automatically choose the system prompt to use. If you create multiple system prompts, Orama will randomly choose one of them.

        This is useful when you want to A/B test which system prompt works best for your users.

        ### Manual via SDK

        When set to "manual via SDK", the system prompt will not be used automatically. Instead, you will need to specify the system prompt ID when making a request to the Orama Answer Engine API via the SDK.

        This can be useful when you know the user that's logged in, and you want to give them a specific system prompt (e.g., to reply in a specific language or avoid giving certain information).

        ```js
        import { OramaClient } from '@oramacloud/client'

        const client = new OramaClient({
          endpoint: 'your-endpoint',
          api_key: 'your-api-key'
        })

        const session = client
          .createAnswerSession({
            events: { ... },
            // Orama will randomly choose one of the system prompts.
            // Set just one prompt if you want to force Orama to use it.
            systemPrompts: [
              'sp_italian-prompt-chc4o0',
              'sp_italian-prompt-with-greetings-2bx7d3'
            ]
          })

        await session.ask({
          term: 'what is Orama?'
        })
        ```

        You can also change the system prompt configuration at any time by updating the system prompt ID in the SDK:

        ```js
        import { OramaClient } from '@oramacloud/client'

        const client = new OramaClient({
          endpoint: 'your-endpoint',
          api_key: 'your-api-key'
        })

        const session = client
          .createAnswerSession({
            events: { ... },
            systemPrompts: ['sp_italian-prompt-with-greetings-2bx7d3']
          })

        session.setSystemPromptConfiguration({
          systemPrompts: ['sp_italian-prompt-with-greetings-2bx7d3'] // Overrides the previous configuration
        })

        await session.ask({
          term: 'what is Orama?'
        })
        ```

        ## Using the editor

        You can instruct Orama to behave in a specific way when giving answers to the user. For example, you could say to always reply in Italian, or to reply in a very specific format (like JSON), etc.

        Before you'll be able to test and use your system prompt, Orama will validate it against three security metrics:

        1. **Jailbreak.** Content that includes impersonation, hacking, illegal activities, or harmful actions is not allowed. Please also avoid trying to bypass or ignore any previous instructions to keep interactions safe and compliant.
        2. **Length.** Keep the content under 2000 characters to ensure clarity and readability. A long prompt can be confusing and difficult for LLMs to follow.
        3. **Number of instructions.** Limit the content to 10 instructions or fewer to keep it clear and easy for the LLMs to follow.

        <img src="/cloud/guides/custom-system-prompts/customized-prompt.png" alt="Customized prompt" />

        Once you've customized the system prompt, you can test it with the demo searchbox on the right side of the page:

        <img src="/cloud/guides/custom-system-prompts/test-prompt.png" alt="Testing prompt" />

        ## What's next

        In the next release of Orama, we will introduce a **scoring mechanism** for your answer sessions. This will allow you to track the performance of your system prompts and understand which one works best for your users.
    "#;

    let chunker = Chunker::try_new(ChunkerConfig {
        max_tokens: 100,
        overlap: None,
    })
    .unwrap();

    let text_result = chunker.chunk_text(text);
    let code_result = chunker.chunk_code(code, CodeLanguage::TypeScript);
    let mdx_result = chunker.chunk_markdown(mdx);

    dbg!(text_result);
}
