import { assert, assertEquals } from 'https://deno.land/std@0.204.0/assert/mod.ts';
import type { AdvancedAutoqueryEvent, AnswerEvent } from './main.ts';

import { EventsStreamTransformer, parseAnswerStream, parseNLPQueryStream } from './main.ts';

function encodeSSE(lines: string[]): Uint8Array {
  return new TextEncoder().encode(lines.join('\n'));
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
