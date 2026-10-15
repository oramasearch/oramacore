# Write side

Regular and temporary index creation resolve the collection locale through the
shared NLP service before publishing the operation. All Locale variants are
supported; JP, KO and ZH keep their identity and use the shared Unicode word
fallback. Invalid language values fail API deserialization before publication.
The reader must use the same tokenizer dependency as the writer.
