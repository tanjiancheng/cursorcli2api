/**
 * OpenAI Responses API streaming compatibility.
 * Converts internal chat.completion.chunk SSE into Responses API SSE events
 * expected by Codex CLI (wire_api = "responses").
 */

import { randomUUID } from "crypto";

export type ResponsesStreamState = {
  responseId: string;
  messageItemId: string;
  model: string;
  created: number;
  outputIndex: number;
  contentIndex: number;
  sequenceNumber: number;
  textBuffer: string;
  started: boolean;
};

export function createResponsesStreamState(model: string): ResponsesStreamState {
  return {
    responseId: `resp_${randomUUID().replace(/-/g, "").slice(0, 24)}`,
    messageItemId: `msg_${randomUUID().replace(/-/g, "").slice(0, 24)}`,
    model: model || "auto",
    created: Math.floor(Date.now() / 1000),
    outputIndex: 0,
    contentIndex: 0,
    sequenceNumber: 0,
    textBuffer: "",
    started: false,
  };
}

function nextSequence(state: ResponsesStreamState): number {
  const current = state.sequenceNumber;
  state.sequenceNumber += 1;
  return current;
}

function formatResponsesEvent(eventType: string, data: Record<string, unknown>): string {
  return `event: ${eventType}\ndata: ${JSON.stringify({ type: eventType, ...data })}\n\n`;
}

function baseResponse(state: ResponsesStreamState, status: string): Record<string, unknown> {
  return {
    id: state.responseId,
    object: "response",
    created_at: state.created,
    status,
    model: state.model,
    output: [],
    parallel_tool_calls: true,
  };
}

export function responsesStreamInitEvents(state: ResponsesStreamState): string[] {
  const inProgress = baseResponse(state, "in_progress");
  return [
    formatResponsesEvent("response.created", {
      response: inProgress,
      sequence_number: nextSequence(state),
    }),
    formatResponsesEvent("response.in_progress", {
      response: inProgress,
      sequence_number: nextSequence(state),
    }),
    formatResponsesEvent("response.output_item.added", {
      output_index: state.outputIndex,
      item: {
        id: state.messageItemId,
        type: "message",
        role: "assistant",
        status: "in_progress",
        content: [],
      },
      sequence_number: nextSequence(state),
    }),
    formatResponsesEvent("response.content_part.added", {
      item_id: state.messageItemId,
      output_index: state.outputIndex,
      content_index: state.contentIndex,
      part: { type: "output_text", text: "" },
      sequence_number: nextSequence(state),
    }),
  ];
}

export function responsesStreamTextDelta(state: ResponsesStreamState, delta: string): string {
  state.textBuffer += delta;
  return formatResponsesEvent("response.output_text.delta", {
    item_id: state.messageItemId,
    output_index: state.outputIndex,
    content_index: state.contentIndex,
    delta,
    sequence_number: nextSequence(state),
  });
}

export function responsesStreamFinalize(state: ResponsesStreamState): string[] {
  const fullText = state.textBuffer;
  const messageItem = {
    id: state.messageItemId,
    type: "message",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text: fullText }],
  };

  const outputTokens = fullText.length > 0 ? Math.max(1, Math.ceil(fullText.length / 4)) : 0;
  const usage = {
    input_tokens: 0,
    output_tokens: outputTokens,
    total_tokens: outputTokens,
  };

  return [
    formatResponsesEvent("response.output_text.done", {
      item_id: state.messageItemId,
      output_index: state.outputIndex,
      content_index: state.contentIndex,
      text: fullText,
      sequence_number: nextSequence(state),
    }),
    formatResponsesEvent("response.content_part.done", {
      item_id: state.messageItemId,
      output_index: state.outputIndex,
      content_index: state.contentIndex,
      part: { type: "output_text", text: fullText },
      sequence_number: nextSequence(state),
    }),
    formatResponsesEvent("response.output_item.done", {
      output_index: state.outputIndex,
      item: messageItem,
      sequence_number: nextSequence(state),
    }),
    formatResponsesEvent("response.completed", {
      response: {
        ...baseResponse(state, "completed"),
        output: [messageItem],
        usage,
      },
      sequence_number: nextSequence(state),
    }),
  ];
}

export function responsesStreamError(message: string, code: string | number = "stream_error"): string {
  return formatResponsesEvent("error", {
    code,
    message,
    sequence_number: 0,
  });
}

export function createResponsesStreamFromChatCompletion(
  upstreamBody: ReadableStream<Uint8Array>,
  model: string
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const state = createResponsesStreamState(model);

  return new ReadableStream({
    async start(controller) {
      for (const event of responsesStreamInitEvents(state)) {
        controller.enqueue(encoder.encode(event));
      }

      const reader = upstreamBody.getReader();
      let buffer = "";
      let finishReason: string | null = null;
      let streamError: string | null = null;
      let sentContent = false;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            if (trimmed.startsWith(":")) {
              controller.enqueue(encoder.encode(`${trimmed}\n\n`));
              continue;
            }

            if (trimmed.startsWith("event:")) {
              const eventName = trimmed.slice(6).trim();
              if (eventName === "error") continue;
              continue;
            }

            if (!trimmed.startsWith("data:")) continue;

            const payload = trimmed.slice(5).trim();
            if (!payload || payload === "[DONE]") continue;

            let chunk: Record<string, unknown>;
            try {
              chunk = JSON.parse(payload) as Record<string, unknown>;
            } catch {
              continue;
            }

            if (chunk.error && typeof chunk.error === "object") {
              const errObj = chunk.error as Record<string, unknown>;
              streamError = String(errObj.message ?? chunk.error);
              break;
            }

            const choices = chunk.choices as Array<Record<string, unknown>> | undefined;
            if (!choices?.length) continue;

            const choice = choices[0];
            const delta = choice.delta as Record<string, unknown> | undefined;
            const content = delta?.content;

            if (typeof content === "string" && content) {
              sentContent = true;
              controller.enqueue(encoder.encode(responsesStreamTextDelta(state, content)));
            }

            if (choice.finish_reason) {
              finishReason = String(choice.finish_reason);
              break;
            }
          }

          if (streamError || finishReason) break;
        }
      } catch (e) {
        streamError = String(e);
      } finally {
        try {
          reader.releaseLock();
        } catch {
          /* ignore */
        }
      }

      if (streamError) {
        controller.enqueue(encoder.encode(responsesStreamError(streamError)));
      } else if (sentContent || finishReason) {
        for (const event of responsesStreamFinalize(state)) {
          controller.enqueue(encoder.encode(event));
        }
      } else {
        controller.enqueue(
          encoder.encode(responsesStreamError("Upstream stream ended without content"))
        );
      }

      controller.close();
    },
  });
}
