use assert_approx_eq::assert_approx_eq;
use futures::FutureExt;
use serde_json::json;

use crate::collection_manager::sides::read::IndexFieldStatsType;
use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread")]
async fn test_vector_search_basic() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                json!({
                    "id": "1",
                    "text": "The cat is sleeping on the table.",
                }),
                json!({
                    "id": "2",
                    "text": "A cat rests peacefully on the sofa.",
                }),
                json!({
                    "id": "3",
                    "text": "The dog is barking loudly in the yard.",
                }),
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = wait_for(&collection_client, |collection_client| {
        async {
            let output = collection_client
                .search(
                    json!({
                        "term": "A cat sleeps",
                        "mode": "vector"
                    })
                    .try_into()
                    .unwrap(),
                )
                .await
                .unwrap();
            if output.count == 2 {
                Ok(output)
            } else {
                Err(anyhow::anyhow!("Expected 2 result, got {}", output.count))
            }
        }
        .boxed()
    })
    .await
    .unwrap();
    assert_eq!(output.count, 2);
    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );
    assert!(output.hits[0].score > 0.);
    assert!(output.hits[1].score > 0.);

    let output = wait_for(&collection_client, |collection_client| {
        async {
            let output = collection_client
                .search(
                    json!({
                        "term": "A cat sleeps",
                        "mode": "vector",
                        "similarity": 0.0001
                    })
                    .try_into()
                    .unwrap(),
                )
                .await
                .unwrap();
            if output.count == 3 {
                Ok(output)
            } else {
                Err(anyhow::anyhow!("Expected 3 results, got {}", output.count))
            }
        }
        .boxed()
    })
    .await
    .unwrap();
    assert_eq!(output.count, 3);
    assert_eq!(output.hits.len(), 3);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );
    assert_eq!(
        output.hits[2].id,
        format!("{}:{}", index_client.index_id, "3")
    );
    assert!(output.hits[0].score > 0.);
    assert!(output.hits[1].score > 0.);
    assert!(output.hits[2].score > 0.);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vector_search_should_work_after_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                json!({
                    "id": "1",
                    "text": "The process of photosynthesis in plants converts carbon dioxide and water into glucose and oxygen using sunlight energy, primarily occurring in chloroplasts through light-dependent and light-independent reactions.",
                }),
                json!({
                    "id": "2",
                    "text": "Machine learning models require extensive training data and computational resources, with deep neural networks often needing millions of parameters to achieve state-of-the-art performance on complex tasks.",
                }),
                json!({
                    "id": "3",
                    "text": "The Renaissance period in Italy saw unprecedented artistic innovation, with masters like Leonardo da Vinci and Michelangelo combining scientific observation with artistic technique to create revolutionary works.",
                }),
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    wait_for(&collection_client, |collection_client| {
        async {
            let mut result = collection_client
                .reader_stats()
                .await
                .unwrap()
                .indexes_stats
                .into_iter()
                .find(|i| i.id == index_client.index_id)
                .unwrap();
            let IndexFieldStatsType::UncommittedVector(stats) = result.fields_stats.remove(0).stats
            else {
                return Err(anyhow::anyhow!("Expected committed vector field stats"));
            };

            if stats.vector_count > 0 {
                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "Expected 3 vectors, found {}",
                    stats.vector_count
                ))
            }
        }
        .boxed()
    })
    .await
    .unwrap();

    let output1 = collection_client
        .search(
            json!({
                "term": "How do plants make food from sunlight?",
                "mode": "vector",
                "similarity": 0.7
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output1.count, 1);
    assert_eq!(output1.hits.len(), 1);
    assert_eq!(
        output1.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert!(output1.hits[0].score > 0.);

    test_context.commit_all().await.unwrap();

    let output2 = collection_client
        .search(
            json!({
                "term": "How do plants make food from sunlight?",
                "mode": "vector",
                "similarity": 0.7
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output2.count, 1);
    assert_eq!(output2.hits.len(), 1);
    assert_eq!(
        output2.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert!(output2.hits[0].score > 0.);

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();
    let index_id = index_client.index_id;

    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let output3 = collection_client
        .search(
            json!({
                "term": "How do plants make food from sunlight?",
                "mode": "vector",
                "similarity": 0.7
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output3.count, 1);
    assert_eq!(output3.hits.len(), 1);
    assert_eq!(output3.hits[0].id, format!("{}:{}", index_id, "1"));
    assert!(output3.hits[0].score > 0.);

    assert_approx_eq!(output1.hits[0].score, output2.hits[0].score);
    assert_approx_eq!(output2.hits[0].score, output3.hits[0].score);

    drop(test_context);
}

/*
#[tokio::test(flavor = "multi_thread")]
async fn test_commit_hooks() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let code = r#"
function selectEmbeddingsProperties() {
    return "The pen is on the table.";
}
export default {
    selectEmbeddingsProperties
}
"#;

    test_context
        .writer
        .get_hooks_runtime(
            collection_client.write_api_key,
            collection_client.collection_id,
        )
        .await
        .unwrap()
        .insert_javascript_hook(
            index_client.index_id,
            HookName::SelectEmbeddingsProperties,
            code.to_string(),
        )
        .await
        .unwrap();

    index_client
        .insert_documents(
            json!([json!({
                "title": "Today I want to listen only Max Pezzali.",
            })])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Generage embeddings keeps time
    sleep(std::time::Duration::from_millis(500)).await;

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "The pen is on the table.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    // Hook change the meaning of the text, so the exact match should not work
    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "Today I want to listen only Max Pezzali.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    test_context.commit_all().await.unwrap();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key.clone(),
        )
        .unwrap();
    let index_client = collection_client
        .get_test_index_client(index_client.index_id)
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "The pen is on the table.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "My dog is barking.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    index_client
        .insert_documents(
            json!([json!({
                "title": "My dog is barking.",
            })])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Generage embeddings keeps time
    sleep(std::time::Duration::from_millis(500)).await;

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "The pen is on the table.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 2);

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "My dog is barking.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    test_context.commit_all().await.unwrap();
    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key.clone(),
        )
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "The pen is on the table.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 2);

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "My dog is barking.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    drop(test_context);
}

*/

#[tokio::test(flavor = "multi_thread")]
async fn test_document_chunk_long_text_for_embedding_calculation() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([json!({
            "id": "1",
            "text": r#"
                The mind’s first step is to distinguish what is true from what is
                false. However, as soon as thought reflects on itself, what it first
                discovers is a contradiction. Useless to strive to be convincing in
                this case. Over the centuries no one has furnished a clearer and
                more elegant demonstration of the business than Aristotle: “The
                often ridiculed consequence of these opinions is that they destroy
                themselves. For by asserting that all is true we assert the truth of
                the contrary assertion and consequently the falsity of our own
                thesis (for the contrary assertion does not admit that it can be true).
                And if one says that all is false, that assertion is itself false. If we
                declare that solely the assertion opposed to ours is false or else that
                solely ours is not false, we are nevertheless forced to admit an
                infinite number of true or false judgments. For the one who
                expresses a true assertion proclaims simultaneously that it is true,
                and so on ad infinitum.”
                This vicious circle is but the first of a series in which the mind
                that studies itself gets lost in a giddy whirling. The very simplicity
                of these paradoxes makes them irreducible. Whatever may be the
                plays on words and the acrobatics of logic, to understand is, above
                all, to unify. The mind’s deepest desire, even in its most elaborate
                operations, parallels man’s unconscious feeling in the face of his
                universe: it is an insistence upon familiarity, an appetite for clarity.
                Understanding the world for a man is reducing it to the human,
                stamping it with his seal. The cat’s universe is not the universe of
                the anthill. The truism “All thought is anthropomorphic” has no
                other meaning. Likewise, the mind that aims to understand reality
                can consider itself satisfied only by reducing it to terms of thought.
                If man realized that the universe like him can love and suffer, he
                would be reconciled. If thought discovered in the shimmering
                mirrors of phenomena eternal relations capable of summing them
                up and summing themselves up in a single principle, then would be
                seen an intellectual joy of which the myth of the blessed would be
                but a ridiculous imitation. That nostalgia for unity, that appetite for
                the absolute illustrates the essential impulse of the human drama.
                But the fact of that nostalgia’s existence does not imply that it is to
                be immediately satisfied. For if, bridging the gulf that separates
                desire from conquest, we assert with Parmenides the reality of the
                One (whatever it may be), we fall into the ridiculous contradiction
                of a mind that asserts total unity and proves by its very assertion its
                own difference and the diversity it claimed to resolve. This other
                vicious circle is enough to stifle our hopes.
                These are again truisms. I shall again repeat that they are not
                interesting in themselves but in the consequences that can be
                deduced from them. I know another truism: it tells me that man is
                mortal. One can nevertheless count the minds that have deduced
                the extreme conclusions from it. It is essential to consider as a
                constant point of reference in this essay the regular hiatus between
                what we fancy we know and what we really know, practical assent
                and simulated ignorance which allows us to live with ideas which,
                if we truly put them to the test, ought to upset our whole life. Faced
                with this inextricable contradiction of the mind, we shall fully
                grasp the divorce separating us from our own creations. So long as
                the mind keeps silent in the motionless world of its hopes,
                everything is reflected and arranged in the unity of its nostalgia.
                But with its first move this world cracks and tumbles: an infinite
                number of shimmering fragments is offered to the understanding.
                We must despair of ever reconstructing the familiar, calm surface
                which would give us peace of heart. After so many centuries of
                inquiries, so many abdications among thinkers, we are well aware
                that this is true for all our knowledge. With the exception of
                professional rationalists, today people despair of true knowledge. If
                the only significant history of human thought were to be written, it
                would have to be the history of its successive regrets and its
                impotences.
                Of whom and of what indeed can I say: “I know that!” This
                heart within me I can feel, and I judge that it exists. This world I
                can touch, and I likewise judge that it exists. There ends all my
                knowledge, and the rest is construction. For if I try to seize this self
                of which I feel sure, if I try to define and to summarize it, it is
                nothing but water slipping through my fingers. I can sketch one by
                one all the aspects it is able to assume, all those likewise that have
                been attributed to it, this upbringing, this origin, this ardor or these
                silences, this nobility or this vileness. But aspects cannot be added
                up. This very heart which is mine will forever remain indefinable
                to me. Between the certainty I have of my existence and the
                content I try to give to that assurance, the gap will never be filled.
                Forever I shall be a stranger to myself. In psychology as in logic,
                there are truths but no truth. Socrates’”Know thyself” has as much
                value as the “Be virtuous” of our confessionals. They reveal a
                nostalgia at the same time as an ignorance. They are sterile
                exercises on great subjects. They are legitimate only in precisely so
                far as they are approximate.
                And here are trees and I know their gnarled surface, water and
                I feel its taste. These scents of grass and stars at night, certain
                evenings when the heart relaxes—how shall I negate this world
                whose power and strength I feel? Yet all the knowledge on earth
                will give me nothing to assure me that this world is mine. You
                describe it to me and you teach me to classify it. You enumerate its
                laws and in my thirst for knowledge I admit that they are true. You
                take apart its mechanism and my hope increases. At the final stage
                you teach me that this wondrous and multicolored universe can be
                reduced to the atom and that the atom itself can be reduced to the
                electron. All this is good and I wait for you to continue. But you
                tell me of an invisible planetary system in which electrons
                gravitate around a nucleus. You explain this world to me with an
                image. I realize then that you have been reduced to poetry: I shall
                never know. Have I the time to become indignant? You have
                already changed theories. So that science that was to teach me
                everything ends up in a hypothesis, that lucidity founders in
                metaphor, that uncertainty is resolved in a work of art. What need
                had I of so many efforts? The soft lines of these hills and the hand
                of evening on this troubled heart teach me much more. I have
                returned to my beginning. I realize that if through science I can
                seize phenomena and enumerate them, I cannot, for all that,
                apprehend the world. Were I to trace its entire relief with my
                finger, I should not know any more. And you give me the choice
                between a description that is sure but that teaches me nothing and
                hypotheses that claim to teach me but that are not sure. A stranger
                to myself and to the world, armed solely with a thought that
                negates itself as soon as it asserts, what is this condition in which I
                can have peace only by refusing to know and to live, in which the
                appetite for conquest bumps into walls that defy its assaults? To
                will is to stir up paradoxes. Everything is ordered in such a way as
                to bring into being that poisoned peace produced by
                thoughtlessness, lack of heart, or fatal renunciations.
                Hence the intelligence, too, tells me in its way that this world is
                absurd. Its contrary, blind reason, may well claim that all is clear; I
                was waiting for proof and longing for it to be right. But despite so
                many pretentious centuries and over the heads of so many eloquent
                and persuasive men, I know that is false. On this plane, at least,
                there is no happiness if I cannot know. That universal reason,
                practical or ethical, that determinism, those categories that explain
                everything are enough to make a decent man laugh. They have
                nothing to do with the mind. They negate its profound truth, which
                is to be enchained. In this unintelligible and limited universe,
                man’s fate henceforth assumes its meaning. A horde of irrationals
                has sprung up and surrounds him until his ultimate end. In his
                recovered and now studied lucidity, the feeling of the absurd
                becomes clear and definite. I said that the world is absurd, but I
                was too hasty. This world in itself is not reasonable, that is all that
                can be said. But what is absurd is the confrontation of this
                irrational and the wild longing for clarity whose call echoes in the
                human heart. The absurd depends as much on man as on the world.
                For the moment it is all that links them together. It binds them one
                to the other as only hatred can weld two creatures together. This is
                all I can discern clearly in this measureless universe where my
                adventure takes place. Let us pause here. If I hold to be true that
                absurdity that determines my relationship with life, if I become
                thoroughly imbued with that sentiment that seizes me in face of the
                world’s scenes, with that lucidity imposed on me by the pursuit of
                a science, I must sacrifice everything to these certainties and I must
                see them squarely to be able to maintain them. Above all, I must
                adapt my behavior to them and pursue them in all their
                consequences. I am speaking here of decency. But I want to know
                beforehand if thought can live in those deserts.
            "#
            })])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    wait_for(&collection_client, |collection_client| {
        async {
            let result = collection_client.reader_stats().await.unwrap();

            let mut result = result
                .indexes_stats
                .into_iter()
                .find(|i| i.id == index_client.index_id)
                .unwrap();
            let IndexFieldStatsType::UncommittedVector(stats) = result.fields_stats.remove(0).stats
            else {
                panic!("Expected vector field stats")
            };

            // Even if we insert only one document, we have 5 vectors because the text is chunked
            // (2230 tokens with BGESmall, chunk size is 512 tokens + overlap)
            if stats.vector_count == 5 {
                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "Expected 5 vectors, found {}",
                    stats.vector_count
                ))
            }
        }
        .boxed()
    })
    .await
    .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": r#"
                    Throughout history, philosophers and scientists have grappled with a troubling realization: the very tools we use to understand reality seem to undermine themselves when examined closely. When human reason attempts to grasp absolute truth, it encounters a fundamental paradox that cannot be resolved through more reasoning. This self-defeating nature of thought becomes apparent the moment we try to establish any foundation for certain knowledge.
                    Consider how every attempt to prove something ultimately relies on assumptions that themselves require proof, leading to an infinite regress. If someone claims "all statements are false," they create an immediate contradiction, as their own statement would then be false. Similarly, if we assert that we can know truth with certainty, we must first prove that our capacity for knowing is itself trustworthy—but this proof would require the very faculty we're trying to validate. We find ourselves trapped in circular reasoning, unable to escape the prison of our own consciousness.
                    The human mind desperately seeks unity and coherence, yearning to reduce the chaotic multiplicity of experience into comprehensible patterns and principles. We want to believe that underneath the bewildering diversity of phenomena lies some fundamental order that our intellect can grasp. This drive toward unification and simplification appears to be hardwired into human cognition—we cannot help but search for underlying patterns, causal relationships, and organizing principles that would render the universe intelligible.
                    Yet this very desire for clarity and unity collides with the irreducible complexity and apparent irrationality of existence. The world presents itself to us as a collection of fragments that resist our attempts at systematization. Even our most sophisticated scientific theories ultimately rest on metaphors and models rather than direct apprehension of reality. Physics tells us about mathematical relationships between quantities, but cannot tell us what matter or energy actually are in themselves. We can predict and manipulate phenomena with extraordinary precision, yet the essential nature of things remains as mysterious as ever.
                    The problem extends to self-knowledge as well. When I attempt to understand my own consciousness, to grasp the "I" that is doing the thinking, it slips away like water through fingers. I can observe my thoughts, emotions, and sensations, but the observer itself remains elusive. Every attribute I might assign to myself—my history, personality, beliefs—fails to capture the immediate reality of subjective experience. The gap between the certainty of my existence and any content I might give to that existence remains unbridgeable.
                    This epistemological crisis isn't merely an abstract philosophical puzzle but strikes at the heart of human existence. We are beings who need meaning, purpose, and understanding to function, yet we find ourselves in a universe that offers no inherent meaning, no given purpose, no ultimate understanding. The collision between our psychological needs and the silence of the cosmos generates a tension that defines the human condition.
                    Science, which promised to deliver certain knowledge and dispel mystery, instead reveals its own limitations. Each advancement in understanding opens new questions, each solution creates new problems. Theories that once seemed absolutely certain are later overturned. The atom, once thought indivisible, splits into particles, which themselves dissolve into probability waves and quantum uncertainties. The solid ground of material reality evaporates into mathematical abstractions that no one truly comprehends.
                    We might hope that pure logic or mathematics could provide a foundation for certain knowledge, but even these most abstract and rigorous disciplines rest on axioms that must be accepted without proof. Gödel's incompleteness theorems demonstrated that even arithmetic contains statements that cannot be proven within the system itself. Every system of thought, no matter how carefully constructed, contains its own blind spots and limitations.
                    Faced with this situation, we have choices about how to respond. We can embrace a kind of philosophical despair, abandoning the search for truth as futile. We can retreat into dogmatism, arbitrarily choosing beliefs and defending them against all challenges. We can lose ourselves in distraction and routine, avoiding the uncomfortable questions entirely. Or we can accept the tension between our need for understanding and the limits of our comprehension as the fundamental condition within which we must operate, neither surrendering the search for truth nor expecting to find final answers, living in the productive tension between question and uncertainty, between the human need for meaning and the cosmic silence that greets our inquiries.
                "#,
                "mode": "vector",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    // In search result, the document is returned only once
    // because the id is the same
    // even it the document matches twice
    assert_eq!(result.count, 1);
    assert_eq!(result.hits[0].id, format!("{}:1", index_client.index_id));

    drop(test_context);
}

/// Test that vector search works correctly after multiple commits with incremental updates.
/// This tests the incremental commit path for vector fields where:
/// 1. Initial documents are inserted and committed
/// 2. Additional documents are inserted and committed (incremental update)
/// 3. Documents are deleted and committed
/// 4. Search results correctly reflect all changes
#[tokio::test(flavor = "multi_thread")]
async fn test_vector_search_incremental_commits() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Phase 1: Insert initial documents
    index_client
        .insert_documents(
            json!([
                {
                    "id": "plant1",
                    "text": "Photosynthesis is the process by which plants convert sunlight into energy."
                },
                {
                    "id": "plant2",
                    "text": "Chlorophyll gives plants their green color and helps absorb light."
                },
                {
                    "id": "animal1",
                    "text": "Mammals are warm-blooded animals that nurse their young with milk."
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Wait for vectors to be generated
    wait_for(&collection_client, |collection_client| {
        async {
            let output = collection_client
                .search(
                    json!({
                        "term": "How do plants make food?",
                        "mode": "vector",
                        "similarity": 0.5
                    })
                    .try_into()
                    .unwrap(),
                )
                .await
                .unwrap();
            if output.count >= 1 {
                Ok(output)
            } else {
                Err(anyhow::anyhow!("Waiting for vectors to be generated"))
            }
        }
        .boxed()
    })
    .await
    .unwrap();

    // First commit
    test_context.commit_all().await.unwrap();

    // Verify search works after commit
    let output = collection_client
        .search(
            json!({
                "term": "How do plants make food?",
                "mode": "vector",
                "similarity": 0.5
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(output.count >= 1, "Should find plant-related documents");

    // Phase 2: Insert additional documents (incremental update)
    index_client
        .insert_documents(
            json!([
                {
                    "id": "plant3",
                    "text": "Roots absorb water and nutrients from the soil for plant growth."
                },
                {
                    "id": "tech1",
                    "text": "Machine learning algorithms learn patterns from large datasets."
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Wait for new vectors
    wait_for(&collection_client, |collection_client| {
        async {
            let output = collection_client
                .search(
                    json!({
                        "term": "Machine learning and artificial intelligence",
                        "mode": "vector",
                        "similarity": 0.4
                    })
                    .try_into()
                    .unwrap(),
                )
                .await
                .unwrap();
            if output.count >= 1 {
                Ok(output)
            } else {
                Err(anyhow::anyhow!("Waiting for new vectors"))
            }
        }
        .boxed()
    })
    .await
    .unwrap();

    // Second commit (incremental update path)
    test_context.commit_all().await.unwrap();

    // Verify both old and new documents are searchable
    let output = collection_client
        .search(
            json!({
                "term": "How do plants absorb water and nutrients?",
                "mode": "vector",
                "similarity": 0.5
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(output.count >= 1, "Should find plant-related documents including new one");

    let output = collection_client
        .search(
            json!({
                "term": "Machine learning and AI",
                "mode": "vector",
                "similarity": 0.4
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(output.count >= 1, "Should find machine learning document");

    // Phase 3: Delete a document
    index_client
        .delete_documents(vec!["plant1".to_string()])
        .await
        .unwrap();

    // Third commit (deletion)
    test_context.commit_all().await.unwrap();

    // Verify deleted document is not returned in search
    let output = collection_client
        .search(
            json!({
                "term": "Photosynthesis converts sunlight",
                "mode": "vector",
                "similarity": 0.7
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // The deleted document (plant1) should not appear in results
    for hit in &output.hits {
        assert!(
            !hit.id.ends_with(":plant1"),
            "Deleted document plant1 should not appear in search results"
        );
    }

    // But other plant documents should still be searchable
    let output = collection_client
        .search(
            json!({
                "term": "Chlorophyll and plant color",
                "mode": "vector",
                "similarity": 0.5
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(output.count >= 1, "Non-deleted plant documents should still be searchable");

    // Phase 4: Verify data persists after reload
    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    // Verify search still works after reload
    let output = collection_client
        .search(
            json!({
                "term": "Machine learning AI",
                "mode": "vector",
                "similarity": 0.4
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(output.count >= 1, "Should still find tech document after reload");

    // Verify deleted document is still not returned after reload
    let output = collection_client
        .search(
            json!({
                "term": "Photosynthesis sunlight energy plants",
                "mode": "vector",
                "similarity": 0.7
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    for hit in &output.hits {
        assert!(
            !hit.id.ends_with(":plant1"),
            "Deleted document should not appear after reload: found {}",
            hit.id
        );
    }

    drop(test_context);
}
