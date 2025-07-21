# @orama/oramacore-events-parser

A JavaScript/TypeScript library for parsing OramaCore Server-Sent Events (SSE) streams, with a simple, strongly-typed event subscription API.

## Features

- Parse SSE streams from OramaCore's `/generate/answer` and `/generate/nlp_query` endpoints
- Subscribe to specific event types with `.on('event_name', handler)`
- Statically-typed helpers: `.onStateChange(handler)` and `.onProgress(handler)`

## Installation

```sh
npm i @orama/oramacore-events-parser
```

## Usage

### Parse Answer Stream

```ts
const emitter = parseAnswerStream(response.body!);

emitter.on('answer_token', (event) => {
  console.log('Token:', event.token);
});

emitter.onStateChange((event) => {
  console.log('State changed:', event.state, event.message);
});

emitter.onProgress((event) => {
  console.log('Progress:', event.current_step, '/', event.total_steps);
});
```

### Parse NLP Query Stream

```ts
const emitter = parseNLPQueryStream(response.body!);

emitter.on('search_results', (event) => {
  console.log('Results:', event.results);
});

emitter.onStateChange((event) => {
  console.log('State changed:', event.state, event.message);
});
```

## API

- `.on(eventName, handler)` — Subscribe to any event type
- `.onStateChange(handler)` — Subscribe to `state_changed` events (typed)
- `.onProgress(handler)` — Subscribe to `progress` events (typed)

## License

[AGPLv3](/LICENSE.md)
