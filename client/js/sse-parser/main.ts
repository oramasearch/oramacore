type Handler<T> = (event: T) => void;

export type SSEEvent = {
  data: string;
};

// Answer flow specific states
export type AnswerFlowState =
  | 'initializing'
  | 'handle_gpu_overload'
  | 'get_llm_config'
  | 'determine_query_strategy'
  | 'simple_rag'
  | 'advanced_autoquery'
  | 'handle_system_prompt'
  | 'optimize_query'
  | 'execute_search'
  | 'execute_before_answer_hook'
  | 'generate_answer'
  | 'generate_related_queries'
  | 'completed'
  | 'error';

// Advanced autoquery specific states (with prefix)
export type AdvancedAutoqueryFlowState =
  | 'advanced_autoquery_initializing'
  | 'advanced_autoquery_analyzing_input'
  | 'advanced_autoquery_query_optimized'
  | 'advanced_autoquery_select_properties'
  | 'advanced_autoquery_properties_selected'
  | 'advanced_autoquery_combine_queries'
  | 'advanced_autoquery_queries_combined'
  | 'advanced_autoquery_generate_tracked_queries'
  | 'advanced_autoquery_tracked_queries_generated'
  | 'advanced_autoquery_execute_before_retrieval_hook'
  | 'advanced_autoquery_hooks_executed'
  | 'advanced_autoquery_execute_searches'
  | 'advanced_autoquery_search_results'
  | 'advanced_autoquery_completed';

// Combined type
export type AllPossibleStates = AnswerFlowState | AdvancedAutoqueryFlowState;

// Create union types from the arrays
export type StateStep = (typeof STATES_STEPS)[number];
export type ProgressStep = (typeof PROGRESS_STEPS)[number];

// More specific event types
export type AdvancedAutoqueryEvent =
  | {
    type: 'state_changed';
    state: StateStep; // Now type-safe instead of string
    message: string;
    data?: unknown;
    is_terminal?: boolean;
  }
  | {
    type: 'error';
    error: string;
    state: StateStep; // Now type-safe
    is_terminal?: boolean;
  }
  | {
    type: 'progress';
    current_step:
      | ProgressStep
      | { type: 'advanced_autoquery'; step: unknown }; // Handle nested format
    total_steps: number;
    message: string;
  }
  | { type: 'search_results'; results: unknown[] };

export type AnswerEvent =
  | {
    type: 'state_changed';
    state: StateStep; // Now type-safe instead of string
    message: string;
    data?: unknown;
    is_terminal?: boolean;
  }
  | {
    type: 'error';
    error: string;
    state: StateStep; // Now type-safe
    is_terminal?: boolean;
  }
  | {
    type: 'progress';
    current_step:
      | ProgressStep
      | { type: 'advanced_autoquery'; step: unknown }; // Handle nested format
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

export const STATES_STEPS = [
  // Main answer flow states
  'initializing',
  'handle_gpu_overload',
  'get_llm_config',
  'determine_query_strategy',
  'simple_rag', // When simple RAG is selected
  'advanced_autoquery', // When advanced autoquery is selected
  'handle_system_prompt',
  'optimize_query',
  'execute_search',
  'execute_before_answer_hook',
  'generate_answer',
  'generate_related_queries',
  'completed',
  'error',

  // Advanced autoquery states (forwarded with prefix)
  'advanced_autoquery_initializing',
  'advanced_autoquery_analyzing_input',
  'advanced_autoquery_query_optimized',
  'advanced_autoquery_select_properties',
  'advanced_autoquery_properties_selected',
  'advanced_autoquery_combine_queries',
  'advanced_autoquery_queries_combined',
  'advanced_autoquery_generate_tracked_queries',
  'advanced_autoquery_tracked_queries_generated',
  'advanced_autoquery_execute_before_retrieval_hook',
  'advanced_autoquery_hooks_executed',
  'advanced_autoquery_execute_searches',
  'advanced_autoquery_search_results',
  'advanced_autoquery_completed',
];

export const PROGRESS_STEPS = [
  // Main answer flow progress steps (enum variant names)
  'Initialize',
  'HandleGPUOverload',
  'GetLLMConfig',
  'DetermineQueryStrategy',
  'HandleSystemPrompt',
  'OptimizeQuery',
  'ExecuteSearch',
  'ExecuteBeforeAnswerHook',
  'GenerateAnswer',
  'GenerateRelatedQueries',
  'Completed',
  'Error',

  // Advanced autoquery progress steps
  'AnalyzeInput',
  'QueryOptimized',
  'SelectProperties',
  'PropertiesSelected',
  'CombineQueriesAndProperties',
  'QueriesCombined',
  'GenerateTrackedQueries',
  'TrackedQueriesGenerated',
  'ExecuteBeforeRetrievalHook',
  'HooksExecuted',
  'ExecuteSearches',
  'SearchResults',
];

export class EventsStreamTransformer extends TransformStream<
  Uint8Array,
  OramaSSEEvent
> {
  constructor() {
    const decoder = new TextDecoder('utf-8', { ignoreBOM: false });
    let buffer = '';

    super({
      start() {
        buffer = '';
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
              // Handle string events like "acknowledged"
              if (jsonStr.startsWith('"') && jsonStr.endsWith('"')) {
                const stringEvent = JSON.parse(jsonStr);
                controller.enqueue({ type: stringEvent });
                continue;
              }

              let parsed = JSON.parse(jsonStr);

              // Handle new nested structure
              if (
                typeof parsed === 'object' &&
                parsed !== null &&
                !('type' in parsed)
              ) {
                // Convert nested format like {"state_changed": {...}} to {type: "state_changed", ...}
                const keys = Object.keys(parsed);
                if (keys.length === 1) {
                  const [eventType] = keys;
                  const eventData = parsed[eventType];

                  if (typeof eventData === 'object' && eventData !== null) {
                    parsed = { type: eventType, ...eventData };
                  } else {
                    parsed = { type: eventType, data: eventData };
                  }
                }
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

    // Check for completion - updated for new event format
    const shouldEnd =
      // Success completion
      (event.type === 'state_changed' &&
        'state' in event &&
        (event.state === 'completed' ||
          event.state === 'advanced_autoquery_completed')) ||
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

// Define separate types for pure advanced autoquery vs forwarded events
export type PureAdvancedAutoqueryEvent =
  | (AdvancedAutoqueryEvent & {
    type: 'state_changed';
    state: Exclude<StateStep, AnswerFlowState>; // Only advanced_autoquery_ states
  })
  | Exclude<AdvancedAutoqueryEvent, { type: 'state_changed' }>;

function isAnswerEvent(event: OramaSSEEvent): event is AnswerEvent {
  // First check for Answer-specific event types
  if (
    event.type === 'acknowledged' ||
    event.type === 'selected_llm' ||
    event.type === 'optimizing_query' ||
    event.type === 'answer_token' ||
    event.type === 'related_queries' ||
    event.type === 'result_action'
  ) {
    return true;
  }

  // For overlapping types, include all since AnswerEvent is a superset
  // The distinction is made at the parser level, not the type guard level
  if (
    event.type === 'state_changed' ||
    event.type === 'error' ||
    event.type === 'progress' ||
    event.type === 'search_results'
  ) {
    return true;
  }

  return false;
}

function isAdvancedAutoqueryEvent(
  event: OramaSSEEvent,
): event is AdvancedAutoqueryEvent {
  // Advanced autoquery events can have any state - the prefix logic only applies
  // when they're forwarded through the answer stream
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

export function isAnswerFlowState(state: StateStep): state is AnswerFlowState {
  return !state.startsWith('advanced_autoquery_');
}

export function isAdvancedAutoqueryState(
  state: StateStep,
): state is AdvancedAutoqueryFlowState {
  return state.startsWith('advanced_autoquery_');
}
