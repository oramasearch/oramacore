type Handler<T> = (event: T) => void;

export type SSEEvent = {
  data: string;
};

export type AdvancedAutoqueryEvent =
  | { type: 'state_changed'; state: string; message: string; data?: unknown }
  | { type: 'error'; error: string; state: string }
  | {
    type: 'progress';
    current_step: unknown;
    total_steps: number;
    message: string;
  }
  | { type: 'search_results'; results: unknown[] };

export type AnswerEvent =
  | { type: 'state_changed'; state: string; message: string; data?: unknown }
  | { type: 'error'; error: string; state: string }
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

        let lineEnd: RegExpExecArray | null;
        while (true) {
          lineEnd = /\r\n|\n|\r/.exec(buffer);
          if (lineEnd === null) break;
          const line = buffer.substring(0, lineEnd.index);
          buffer = buffer.substring(lineEnd.index + lineEnd[0].length);
          if (line.length === 0) {
            if (currentEvent.data) {
              try {
                const parsed = JSON.parse(currentEvent.data) as OramaSSEEvent;
                controller.enqueue(parsed);
              } catch (_) {
                controller.enqueue({
                  type: 'error',
                  error: 'Invalid JSON in SSE data',
                  state: 'parse_error',
                } as AnswerEvent);
              }
            }
            currentEvent = {};
          } else if (!line.startsWith(':')) {
            const firstColon = line.indexOf(':');
            if (firstColon === -1) {
              currentEvent[line] = '';
              continue;
            }
            const key = line.substring(0, firstColon);
            const value = line.substring(firstColon + 1).replace(/^\u0020/, '');
            currentEvent[key] = value;
          }
        }
      },
    });
  }
}

class OramaEventEmitter<T extends { type: string }> {
  private handlers: { [key: string]: Handler<T>[] } = {};

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

  emit(event: T) {
    const hs = this.handlers[event.type];
    if (hs) { for (const h of hs) h(event as any); }
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
  })();
  return emitter;
}
