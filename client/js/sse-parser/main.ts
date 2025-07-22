type Handler<T> = (event: T) => void;

export type SSEEvent = {
  data: string;
};

export type AdvancedAutoqueryEvent =
  | {
    type: 'state_changed';
    state: string;
    message: string;
    data?: unknown;
    is_terminal?: boolean;
  }
  | { type: 'error'; error: string; state: string; is_terminal?: boolean }
  | {
    type: 'progress';
    current_step: unknown;
    total_steps: number;
    message: string;
  }
  | { type: 'search_results'; results: unknown[] };

export type AnswerEvent =
  | {
    type: 'state_changed';
    state: string;
    message: string;
    data?: unknown;
    is_terminal?: boolean;
  }
  | { type: 'error'; error: string; state: string; is_terminal?: boolean }
  | {
    type: 'progress';
    current_step: unknown;
    total_steps: number;
    message: string;
  }
  | { type: 'acknowledged' }
  | { type: 'selected_llm'; provider: string; model: string }
  | {
    type: 'optimizing_query';
    original_query: string;
    optimized_query: string;
  }
  | { type: 'search_results'; results: unknown[] }
  | { type: 'answer_token'; token: string }
  | { type: 'related_queries'; queries: string }
  | { type: 'result_action'; action: string; result: string };

export type OramaSSEEvent = AdvancedAutoqueryEvent | AnswerEvent;

export class EventsStreamTransformer extends TransformStream<
  Uint8Array,
  OramaSSEEvent
> {
  constructor() {
    const decoder = new TextDecoder('utf-8', { ignoreBOM: false });
    let buffer = '';
    let currentEvent: Record<string, string> = {};

    super({
      start() {
        buffer = '';
        currentEvent = {};
      },
      transform(chunk, controller) {
        const chunkText = decoder.decode(chunk);
        buffer += chunkText;

        // Split on double newlines (end of SSE event)
        let eventEnd = buffer.indexOf('\n\n');
        while (eventEnd === -1 && buffer.indexOf('\r\n\r\n') !== -1) {
          eventEnd = buffer.indexOf('\r\n\r\n');
        }
        while (eventEnd !== -1) {
          // Support both \n\n and \r\n\r\n as event delimiters
          let delimiterLength = 2;
          if (buffer.slice(eventEnd, eventEnd + 4) === '\r\n\r\n') {
            delimiterLength = 4;
          }
          const eventBlock = buffer.slice(0, eventEnd);
          buffer = buffer.slice(eventEnd + delimiterLength);

          // Find the data line(s)
          const dataLines = eventBlock
            .split(/\r?\n/)
            .filter((line) => line.startsWith('data:'));
          for (const dataLine of dataLines) {
            const jsonStr = dataLine.replace(/^data:\s*/, '');
            try {
              let parsed = JSON.parse(jsonStr);
              if (
                typeof parsed === 'object' &&
                parsed !== null &&
                Object.keys(parsed).length === 1 &&
                !('type' in parsed)
              ) {
                const [key] = Object.keys(parsed);
                parsed = { type: key, ...parsed[key] };
              }
              controller.enqueue(parsed);
            } catch (e) {
              controller.enqueue({
                type: 'error',
                error: 'Invalid JSON in SSE data',
                state: 'parse_error',
              });
            }
          }
          // Find the next event
          eventEnd = buffer.indexOf('\n\n');
          if (eventEnd === -1 && buffer.indexOf('\r\n\r\n') !== -1) {
            eventEnd = buffer.indexOf('\r\n\r\n');
          }
        }
      },
    });
  }
}

class OramaEventEmitter<T extends { type: string }> {
  private handlers: { [key: string]: Handler<T>[] } = {};
  private endHandlers: Handler<void>[] = [];
  public done: Promise<void>;
  private resolveDone: () => void = () => {};

  constructor() {
    this.done = new Promise<void>((resolve) => {
      this.resolveDone = resolve;
    });
  }

  on<K extends T['type']>(event: K, handler: Handler<Extract<T, { type: K }>>) {
    if (!this.handlers[event]) {
      this.handlers[event] = [];
    }
    (this.handlers[event] as Handler<Extract<T, { type: K }>>[]).push(handler);
    return this;
  }

  onStateChange(handler: Handler<Extract<T, { type: 'state_changed' }>>) {
    return this.on('state_changed' as T['type'], handler as any);
  }

  onProgress(handler: Handler<Extract<T, { type: 'progress' }>>) {
    return this.on('progress' as T['type'], handler as any);
  }

  onEnd(handler: Handler<void>) {
    this.endHandlers.push(handler);
    return this;
  }

  emit(event: T) {
    const hs = this.handlers[event.type];
    if (hs) {
      for (const h of hs) h(event as any);
    }

    // Check for completion
    const shouldEnd =
      // Success completion
      (event.type === 'state_changed' &&
        'state' in event &&
        event.state === 'completed') ||
      // Search results
      event.type === 'search_results' ||
      // Terminal errors only
      (event.type === 'error' &&
        'is_terminal' in event &&
        event.is_terminal === true);

    if (shouldEnd) {
      this._triggerEnd();
    }
  }

  private _triggerEnd() {
    for (const handler of this.endHandlers) {
      handler();
    }
  }

  _markDone() {
    this.resolveDone();
  }
}

function isAnswerEvent(event: OramaSSEEvent): event is AnswerEvent {
  return (
    event.type === 'acknowledged' ||
    event.type === 'selected_llm' ||
    event.type === 'optimizing_query' ||
    event.type === 'search_results' ||
    event.type === 'answer_token' ||
    event.type === 'related_queries' ||
    event.type === 'result_action' ||
    event.type === 'state_changed' ||
    event.type === 'error' ||
    event.type === 'progress'
  );
}

function isAdvancedAutoqueryEvent(
  event: OramaSSEEvent,
): event is AdvancedAutoqueryEvent {
  return (
    event.type === 'state_changed' ||
    event.type === 'error' ||
    event.type === 'progress' ||
    event.type === 'search_results'
  );
}

export function parseAnswerStream(
  stream: ReadableStream<Uint8Array>,
): OramaEventEmitter<AnswerEvent> {
  const emitter = new OramaEventEmitter<AnswerEvent>();
  const transformer = new EventsStreamTransformer();
  (async () => {
    const reader = stream.pipeThrough(transformer).getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (isAnswerEvent(value)) emitter.emit(value);
    }
    // Give a small delay to ensure all events are emitted
    await new Promise((resolve) => setTimeout(resolve, 0));
    emitter._markDone();
  })();
  return emitter;
}

export function parseNLPQueryStream(
  stream: ReadableStream<Uint8Array>,
): OramaEventEmitter<AdvancedAutoqueryEvent> {
  const emitter = new OramaEventEmitter<AdvancedAutoqueryEvent>();
  const transformer = new EventsStreamTransformer();
  (async () => {
    const reader = stream.pipeThrough(transformer).getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (isAdvancedAutoqueryEvent(value)) emitter.emit(value);
    }
    // Give a small delay to ensure all events are emitted
    await new Promise((resolve) => setTimeout(resolve, 0));
    emitter._markDone();
  })();
  return emitter;
}
