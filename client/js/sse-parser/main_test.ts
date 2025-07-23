import { assert, assertEquals, assertExists } from 'https://deno.land/std@0.204.0/assert/mod.ts';
import type { AdvancedAutoqueryEvent, AnswerEvent } from './main.ts';

import { EventsStreamTransformer, parseAnswerStream, parseNLPQueryStream } from './main.ts';

function encodeSSE(lines: string[]): Uint8Array {
  return new TextEncoder().encode(lines.join('\n'));
}

function encodeSSEString(data: string): Uint8Array {
  return new TextEncoder().encode(data);
}

Deno.test('EventsStreamTransformer parses valid SSE JSON event', async () => {
  const eventObj = {
    type: 'state_changed',
    state: 'foo',
    message: 'bar',
  } as const;
  const sseData = `data: ${JSON.stringify(eventObj)}\n\n`;
  const input = encodeSSE([sseData]);
  const transformer = new EventsStreamTransformer();
  const reader = new Response(input).body!.pipeThrough(transformer).getReader();
  const { value, done } = await reader.read();
  assert(!done);
  assertEquals(value, eventObj);
});

Deno.test('EventsStreamTransformer parses new nested format', async () => {
  const nestedSSEData = `data: {"state_changed":{"state":"initializing","message":"Starting answer generation","data":null}}\n\n`;
  const transformer = new EventsStreamTransformer();
  const input = encodeSSEString(nestedSSEData);
  const reader = new Response(input).body!.pipeThrough(transformer).getReader();
  const { value, done } = await reader.read();
  assert(!done);
  assertEquals(value.type, 'state_changed');

  // Type guard: check that it's a state_changed event before accessing properties
  if (value.type === 'state_changed') {
    assertEquals(value.state, 'initializing');
    assertEquals(value.message, 'Starting answer generation');
    assertEquals(value.data, null);
  }
});

Deno.test('EventsStreamTransformer parses string events', async () => {
  const stringSSEData = `data: "acknowledged"\n\n`;
  const transformer = new EventsStreamTransformer();
  const input = encodeSSEString(stringSSEData);
  const reader = new Response(input).body!.pipeThrough(transformer).getReader();
  const { value, done } = await reader.read();
  assert(!done);
  assertEquals(value.type, 'acknowledged');
});

Deno.test('EventsStreamTransformer handles complex nested progress events', async () => {
  const complexProgressSSE =
    `data: {"progress":{"current_step":{"type":"Initialize","interaction_id":"123"},"total_steps":9,"message":"Processing step 1/9"}}\n\n`;
  const transformer = new EventsStreamTransformer();
  const input = encodeSSEString(complexProgressSSE);
  const reader = new Response(input).body!.pipeThrough(transformer).getReader();
  const { value, done } = await reader.read();
  assert(!done);
  assertEquals(value.type, 'progress');

  // Type guard: check that it's a progress event before accessing properties
  if (value.type === 'progress') {
    assertEquals(value.total_steps, 9);
    assertEquals(value.message, 'Processing step 1/9');
    assertExists(value.current_step);
    assertEquals((value.current_step as any).type, 'Initialize');
    assertEquals((value.current_step as any).interaction_id, '123');
  }
});

Deno.test('EventsStreamTransformer emits error event on invalid JSON', async () => {
  const sseData = `data: {invalid json}\n\n`;
  const input = encodeSSE([sseData]);
  const transformer = new EventsStreamTransformer();
  const reader = new Response(input).body!.pipeThrough(transformer).getReader();
  const { value, done } = await reader.read();
  assert(!done);
  assert(value !== undefined);
  if (value && value.type === 'error') {
    assertEquals(value.state, 'parse_error');
  } else {
    throw new Error('Expected error event');
  }
});

Deno.test('parseAnswerStream emits correct events to handlers', async () => {
  const events: AnswerEvent[] = [
    { type: 'acknowledged' },
    { type: 'selected_llm', provider: 'openai', model: 'gpt-4' },
    { type: 'answer_token', token: 'Hello' },
    { type: 'state_changed', state: 'foo', message: 'bar' },
    { type: 'error', error: 'fail', state: 'fail_state' },
  ];
  const sseLines = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('');
  const input = encodeSSE([sseLines]);
  const stream = new Response(input).body!;
  const emitter = parseAnswerStream(stream);

  const received: Record<string, any[]> = {};
  for (const e of events) {
    emitter.on(e.type, (ev) => {
      received[e.type] ||= [];
      received[e.type].push(ev);
    });
  }

  // Wait for stream processing to complete
  await emitter.done;

  for (const e of events) {
    assert(received[e.type]?.length);
    assertEquals(received[e.type][0], e);
  }
});

Deno.test('parseAnswerStream handles new nested format', async () => {
  const nestedSSEData = `data: {"state_changed":{"state":"initializing","message":"Starting answer generation","data":null}}

data: "acknowledged"

data: {"selected_llm":{"provider":"OpenAI","model":"gpt-4.1"}}

data: {"optimizing_query":{"original_query":"test","optimized_query":"test optimized"}}

data: {"answer_token":{"token":"Hello"}}

data: {"state_changed":{"state":"completed","message":"Done"}}

`;

  const stream = new Response(encodeSSEString(nestedSSEData)).body!;
  const emitter = parseAnswerStream(stream);

  const events: AnswerEvent[] = [];
  let completedReceived = false;
  let endCalled = false;

  emitter.on('state_changed', (event) => {
    events.push(event);
    if (event.state === 'completed') completedReceived = true;
  });
  emitter.on('acknowledged', (event) => events.push(event));
  emitter.on('selected_llm', (event) => events.push(event));
  emitter.on('optimizing_query', (event) => events.push(event));
  emitter.on('answer_token', (event) => events.push(event));

  emitter.onEnd(() => {
    endCalled = true;
  });

  await emitter.done;

  assertEquals(events.length, 6);
  assert(completedReceived);
  assert(endCalled); // Should end on completion

  // Verify specific events
  const initializing = events.find(
    (e) => e.type === 'state_changed' && 'state' in e && e.state === 'initializing',
  );
  assertExists(initializing);

  // Type guard: ensure it's a state_changed event before accessing message
  if (initializing && initializing.type === 'state_changed') {
    assertEquals(initializing.message, 'Starting answer generation');
  }

  const selectedLLM = events.find((e) => e.type === 'selected_llm') as any;
  assertExists(selectedLLM);
  assertEquals(selectedLLM.provider, 'OpenAI');
  assertEquals(selectedLLM.model, 'gpt-4.1');
});

Deno.test('parseNLPQueryStream emits correct events to handlers', async () => {
  const events: AdvancedAutoqueryEvent[] = [
    { type: 'state_changed', state: 'init', message: 'Initializing' },
    {
      type: 'progress',
      current_step: { step: 1 },
      total_steps: 2,
      message: 'Step 1',
    },
    { type: 'search_results', results: [{ id: 1 }] },
    { type: 'error', error: 'fail', state: 'fail_state' },
  ];
  const sseLines = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('');
  const input = encodeSSE([sseLines]);
  const stream = new Response(input).body!;
  const emitter = parseNLPQueryStream(stream);

  const received: Record<string, any[]> = {};
  for (const e of events) {
    emitter.on(e.type, (ev) => {
      received[e.type] ||= [];
      received[e.type].push(ev);
    });
  }

  // Wait for all events to be processed
  await new Promise((resolve) => setTimeout(resolve, 50));

  for (const e of events) {
    assert(received[e.type]?.length);
    assertEquals(received[e.type][0], e);
  }
});

Deno.test('parseNLPQueryStream handles advanced autoquery nested format', async () => {
  const advancedSSEData = `data: {"state_changed":{"state":"advanced_autoquery_initializing","message":"Advanced Autoquery: Starting","data":null}}

data: {"progress":{"current_step":{"type":"advanced_autoquery","step":{"type":"Initialize","conversation_messages":1}},"total_steps":8,"message":"Processing step 1/8"}}

data: {"search_results":{"results":[{"id":"test:123","score":0.95}]}}

`;

  const stream = new Response(encodeSSEString(advancedSSEData)).body!;
  const emitter = parseNLPQueryStream(stream);

  const stateChanges: any[] = [];
  const progressEvents: any[] = [];
  const searchResults: any[] = [];

  emitter.onStateChange((event) => stateChanges.push(event));
  emitter.onProgress((event) => progressEvents.push(event));
  emitter.on('search_results', (event) => searchResults.push(event));

  await new Promise((resolve) => setTimeout(resolve, 100));

  assertEquals(stateChanges.length, 1);
  assertEquals(stateChanges[0].state, 'advanced_autoquery_initializing');

  assertEquals(progressEvents.length, 1);
  assertEquals(progressEvents[0].total_steps, 8);
  assertExists(progressEvents[0].current_step);
  assertEquals(
    (progressEvents[0].current_step as any).type,
    'advanced_autoquery',
  );

  assertEquals(searchResults.length, 1);
  assertEquals(searchResults[0].results[0].id, 'test:123');
});

Deno.test('handlers for unrelated events are not called', async () => {
  const events: AnswerEvent[] = [
    { type: 'acknowledged' },
    { type: 'answer_token', token: 'Hello' },
  ];
  const sseLines = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('');
  const input = encodeSSE([sseLines]);
  const stream = new Response(input).body!;
  const emitter = parseAnswerStream(stream);

  let unrelatedCalled = false;
  emitter.on('selected_llm', () => {
    unrelatedCalled = true;
  });

  // Wait for all events to be processed
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert(!unrelatedCalled);
});

Deno.test('completion only triggers on completed state or terminal error', async () => {
  const nonTerminalSSE = `data: {"search_results":{"results":[{"id":"123"}]}}

data: {"state_changed":{"state":"processing","message":"Still working"}}

data: {"error":{"error":"Non-terminal error","state":"retry_state","is_terminal":false}}

`;

  const emitter = parseAnswerStream(
    new Response(encodeSSEString(nonTerminalSSE)).body!,
  );

  let endCalled = false;
  emitter.onEnd(() => {
    endCalled = true;
  });

  await new Promise((resolve) => setTimeout(resolve, 100));

  // Should NOT trigger end since no completion or terminal error
  assert(!endCalled);
});

Deno.test('completion triggers on terminal error', async () => {
  const terminalErrorSSE = `data: {"state_changed":{"state":"processing","message":"Working"}}

data: {"error":{"error":"Fatal error","state":"failed","is_terminal":true}}

`;

  const emitter = parseAnswerStream(
    new Response(encodeSSEString(terminalErrorSSE)).body!,
  );

  let endCalled = false;
  emitter.onEnd(() => {
    endCalled = true;
  });

  await emitter.done;

  // Should trigger end on terminal error
  assert(endCalled);
});

Deno.test('onStateChange and onProgress work for parseAnswerStream', async () => {
  const events: AnswerEvent[] = [
    { type: 'state_changed', state: 'foo', message: 'bar' },
    { type: 'progress', current_step: 1, total_steps: 2, message: 'Step 1' },
  ];
  const sseLines = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('');
  const input = encodeSSE([sseLines]);
  const stream = new Response(input).body!;
  const emitter = parseAnswerStream(stream);

  let stateChangeCalled = 0;
  let progressCalled = 0;
  let lastState: string | undefined;
  let lastStep: number | undefined;

  emitter.onStateChange((ev) => {
    stateChangeCalled++;
    lastState = ev.state;
  });
  emitter.onProgress((ev) => {
    progressCalled++;
    if (typeof ev.current_step === 'number') {
      lastStep = ev.current_step;
    } else if (
      ev.current_step &&
      typeof ev.current_step === 'object' &&
      'step' in ev.current_step &&
      typeof (ev.current_step as any).step === 'number'
    ) {
      lastStep = (ev.current_step as any).step;
    } else {
      lastStep = undefined;
    }
  });

  await new Promise((resolve) => setTimeout(resolve, 50));

  assertEquals(stateChangeCalled, 1);
  assertEquals(progressCalled, 1);
  assertEquals(lastState, 'foo');
  assertEquals(lastStep, 1);
});

Deno.test('onStateChange and onProgress work with new nested format', async () => {
  const nestedSSE = `data: {"state_changed":{"state":"initializing","message":"Starting"}}

data: {"progress":{"current_step":{"type":"Initialize","interaction_id":"123"},"total_steps":9,"message":"Processing step 1/9"}}

`;

  const emitter = parseAnswerStream(
    new Response(encodeSSEString(nestedSSE)).body!,
  );

  let stateChangeCalled = 0;
  let progressCalled = 0;
  let lastState: string | undefined;
  let lastMessage: string | undefined;

  emitter.onStateChange((ev) => {
    stateChangeCalled++;
    lastState = ev.state;
  });
  emitter.onProgress((ev) => {
    progressCalled++;
    lastMessage = ev.message;
  });

  await new Promise((resolve) => setTimeout(resolve, 50));

  assertEquals(stateChangeCalled, 1);
  assertEquals(progressCalled, 1);
  assertEquals(lastState, 'initializing');
  assertEquals(lastMessage, 'Processing step 1/9');
});

Deno.test('onStateChange and onProgress work for parseNLPQueryStream', async () => {
  const events: AdvancedAutoqueryEvent[] = [
    { type: 'state_changed', state: 'init', message: 'Initializing' },
    {
      type: 'progress',
      current_step: { step: 2 },
      total_steps: 3,
      message: 'Step 2',
    },
  ];
  const sseLines = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('');
  const input = encodeSSE([sseLines]);
  const stream = new Response(input).body!;
  const emitter = parseNLPQueryStream(stream);

  let stateChangeCalled = 0;
  let progressCalled = 0;
  let lastState: string | undefined;
  let lastStep: number | undefined;

  emitter.onStateChange((ev) => {
    stateChangeCalled++;
    lastState = ev.state;
  });
  emitter.onProgress((ev) => {
    progressCalled++;
    const step = typeof ev.current_step === 'object' &&
        ev.current_step !== null &&
        'step' in ev.current_step &&
        typeof (ev.current_step as any).step === 'number'
      ? (ev.current_step as any).step
      : undefined;
    lastStep = step;
  });

  await new Promise((resolve) => setTimeout(resolve, 50));

  assertEquals(stateChangeCalled, 1);
  assertEquals(progressCalled, 1);
  assertEquals(lastState, 'init');
  assertEquals(lastStep, 2);
});

Deno.test('handles mixed line endings \\n and \\r\\n', async () => {
  const mixedSSE =
    `data: {"state_changed":{"state":"init","message":"Starting"}}\r\n\r\ndata: "acknowledged"\n\ndata: {"state_changed":{"state":"completed","message":"Done"}}\r\n\r\n`;

  const transformer = new EventsStreamTransformer();
  const input = encodeSSEString(mixedSSE);
  const reader = new Response(input).body!.pipeThrough(transformer).getReader();

  const events: any[] = [];
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    events.push(value);
  }

  assertEquals(events.length, 3);
  assertEquals(events[0].type, 'state_changed');
  assertEquals(events[0].state, 'init');
  assertEquals(events[1].type, 'acknowledged');
  assertEquals(events[2].type, 'state_changed');
  assertEquals(events[2].state, 'completed');
});
