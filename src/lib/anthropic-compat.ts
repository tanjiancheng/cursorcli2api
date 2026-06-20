/**
 * Anthropic Messages API compatibility module.
 * Converts between Anthropic Messages API format and internal ChatCompletionRequest.
 *
 * Inbound: Anthropic content blocks (text, image, tool_use, tool_result, thinking)
 *          → OpenAI ChatMessage format (with tool_calls and tool role messages).
 * Outbound: OpenAI ChatCompletion response → Anthropic Messages API response
 *           (text + tool_use content blocks).
 */

import { z } from "zod";
import { randomUUID } from "crypto";
import type { ChatCompletionRequest, ChatMessage } from "./openai-compat.js";
import { normalizeMessageContent } from "./openai-compat.js";

// ─────────────────────────────────────────────────────────────────────────────
// Zod schemas
// ─────────────────────────────────────────────────────────────────────────────

const AnthropicContentBlockSchema = z.union([
  z.string(),
  z.array(
    z
      .object({
        type: z.string(),
      })
      .passthrough()
  ),
]);

const AnthropicMessageSchema = z
  .object({
    role: z.enum(["user", "assistant", "system"]),
    content: AnthropicContentBlockSchema,
  })
  .passthrough();

export const AnthropicMessagesRequestSchema = z
  .object({
    model: z.string(),
    messages: z.array(AnthropicMessageSchema).min(1),
    max_tokens: z.number().int().positive(),
    system: z.unknown().optional(),
    stream: z.boolean().optional().default(false),
    temperature: z.number().optional(),
    top_p: z.number().optional(),
    top_k: z.number().optional(),
    stop_sequences: z.array(z.string()).optional(),
    metadata: z.unknown().optional(),
  })
  .passthrough();

export type AnthropicMessagesRequest = z.infer<typeof AnthropicMessagesRequestSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// Request conversion: Anthropic -> internal ChatCompletionRequest
// ─────────────────────────────────────────────────────────────────────────────

function anthropicSystemToString(system: unknown): string {
  if (!system) return "";
  if (typeof system === "string") return system;
  if (Array.isArray(system)) {
    return system
      .filter(
        (block): block is Record<string, unknown> =>
          typeof block === "object" && block !== null && (block as Record<string, unknown>).type === "text"
      )
      .map((block) => String(block.text ?? ""))
      .join("\n");
  }
  return String(system);
}

interface RawContentBlock {
  type: string;
  [key: string]: unknown;
}

interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface ToolResultItem {
  tool_call_id: string;
  content: string;
}

/** Extract OpenAI-compatible tool_calls from Anthropic tool_use content blocks. */
function extractToolCalls(content: unknown): OpenAIToolCall[] | undefined {
  if (!Array.isArray(content)) return undefined;
  const calls: OpenAIToolCall[] = [];
  for (const block of content) {
    if (typeof block !== "object" || block === null) continue;
    const b = block as Record<string, unknown>;
    if (b.type !== "tool_use") continue;
    calls.push({
      id: typeof b.id === "string" ? b.id : `toolu_${calls.length}`,
      type: "function",
      function: {
        name: typeof b.name === "string" ? b.name : "",
        arguments: typeof b.input === "object" && b.input !== null
          ? JSON.stringify(b.input)
          : String(b.input ?? "{}"),
      },
    });
  }
  return calls.length > 0 ? calls : undefined;
}

/** Extract tool_result items from user content blocks. */
function extractToolResults(content: unknown): ToolResultItem[] | undefined {
  if (!Array.isArray(content)) return undefined;
  const results: ToolResultItem[] = [];
  for (const block of content) {
    if (typeof block !== "object" || block === null) continue;
    const b = block as Record<string, unknown>;
    if (b.type !== "tool_result") continue;
    const tcId = typeof b.tool_use_id === "string" ? b.tool_use_id : "unknown";
    const resultContent =
      typeof b.content === "string"
        ? b.content
        : JSON.stringify(b.content ?? "");
    results.push({ tool_call_id: tcId, content: resultContent });
  }
  return results.length > 0 ? results : undefined;
}

function anthropicContentToChatContent(content: unknown): unknown {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return String(content ?? "");

  const parts: Array<Record<string, unknown>> = [];
  for (const block of content) {
    if (typeof block !== "object" || block === null) continue;
    const b = block as Record<string, unknown>;
    if (b.type === "text") {
      parts.push({ type: "text", text: String(b.text ?? "") });
    } else if (b.type === "image") {
      const source = b.source as Record<string, unknown> | undefined;
      if (source?.type === "base64") {
        const mediaType = String(source.media_type ?? "image/png");
        const data = String(source.data ?? "");
        parts.push({
          type: "image_url",
          image_url: { url: `data:${mediaType};base64,${data}` },
        });
      }
    } else if (b.type === "thinking") {
      parts.push({ type: "text", text: String(b.thinking ?? "") });
    } else if (b.type === "tool_use" || b.type === "tool_result") {
      // extracted separately via extractToolCalls / extractToolResults
    }
  }

  if (parts.length === 1 && parts[0].type === "text") {
    return parts[0].text as string;
  }
  return parts.length > 0 ? parts : "";
}

function anthropicMessageContentToString(content: unknown): string {
  if (typeof content === "string") return content;
  const chatContent = anthropicContentToChatContent(content);
  if (typeof chatContent === "string") return chatContent;
  if (Array.isArray(chatContent)) {
    return chatContent
      .filter((part): part is Record<string, unknown> => typeof part === "object" && part !== null)
      .map((part) => (part.type === "text" ? String(part.text ?? "") : ""))
      .filter(Boolean)
      .join("\n");
  }
  return String(chatContent ?? "");
}

export function anthropicRequestToChatRequest(req: AnthropicMessagesRequest): ChatCompletionRequest {
  const messages: ChatMessage[] = [];
  const systemParts: string[] = [];

  const systemText = anthropicSystemToString(req.system);
  if (systemText.trim()) systemParts.push(systemText);

  for (const msg of req.messages) {
    if (msg.role === "system") {
      const text = anthropicMessageContentToString(msg.content);
      if (text.trim()) systemParts.push(text);
      continue;
    }

    if (msg.role === "assistant") {
      const toolCalls = extractToolCalls(msg.content);
      const chatMsg: ChatMessage = {
        role: "assistant",
        content: anthropicContentToChatContent(msg.content),
      };
      if (toolCalls) {
        (chatMsg as Record<string, unknown>).tool_calls = toolCalls;
      }
      messages.push(chatMsg);
    } else {
      // user role: extract tool_results as separate tool messages
      const toolResults = extractToolResults(msg.content);
      messages.push({
        role: "user",
        content: anthropicContentToChatContent(msg.content),
      });
      if (toolResults) {
        for (const tr of toolResults) {
          messages.push({
            role: "tool",
            content: tr.content,
          } as ChatMessage & { tool_call_id?: string });
          const last = messages[messages.length - 1] as Record<string, unknown>;
          last.tool_call_id = tr.tool_call_id;
        }
      }
    }
  }

  if (systemParts.length > 0) {
    messages.unshift({ role: "system", content: systemParts.join("\n\n") });
  }

  return {
    model: req.model,
    messages,
    stream: req.stream ?? false,
    max_tokens: req.max_tokens ?? null,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Response conversion: internal -> Anthropic Messages format
// ─────────────────────────────────────────────────────────────────────────────

export interface AnthropicContentBlock {
  type: "text" | "tool_use" | "thinking";
  text?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  thinking?: string;
  signature?: string;
}

export interface AnthropicMessagesResponse {
  id: string;
  type: "message";
  role: "assistant";
  content: AnthropicContentBlock[];
  model: string;
  stop_reason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" | null;
  stop_sequence: string | null;
  usage: { input_tokens: number; output_tokens: number };
}

export function chatCompletionToAnthropicResponse(
  chat: Record<string, unknown>,
  requestModel: string
): AnthropicMessagesResponse {
  const content: AnthropicContentBlock[] = [];
  const choices = (chat.choices as unknown[]) ?? [];
  let message: Record<string, unknown> = {};

  if (choices.length > 0) {
    const first = choices[0] as Record<string, unknown>;
    message = (first.message as Record<string, unknown>) ?? {};
    const text = normalizeMessageContent(message.content);
    if (text) {
      content.push({ type: "text", text });
    }
  }

  // Extract tool_calls and convert to tool_use blocks
  const toolCalls = message.tool_calls as Array<Record<string, unknown>> | undefined;
  if (toolCalls && Array.isArray(toolCalls)) {
    for (const tc of toolCalls) {
      const fn = tc.function as Record<string, unknown> | undefined;
      let input: Record<string, unknown> = {};
      if (typeof fn?.arguments === "string") {
        try { input = JSON.parse(fn.arguments) as Record<string, unknown>; } catch { /* keep empty */ }
      } else if (typeof fn?.arguments === "object" && fn?.arguments !== null) {
        input = fn.arguments as Record<string, unknown>;
      }
      content.push({
        type: "tool_use",
        id: typeof tc.id === "string" ? tc.id : `toolu_${content.length}`,
        name: typeof fn?.name === "string" ? fn.name : "",
        input,
      });
    }
  }

  let inputTokens = 0;
  let outputTokens = 0;
  const usage = chat.usage as Record<string, unknown> | undefined;
  if (usage && typeof usage === "object") {
    inputTokens = Math.floor(Number(usage.prompt_tokens) || 0);
    outputTokens = Math.floor(Number(usage.completion_tokens) || 0);
  }

  const finishReason = (choices[0] as Record<string, unknown>)?.finish_reason;
  let stopReason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" | null = "end_turn";
  if (finishReason === "tool_calls") stopReason = "tool_use";
  else if (finishReason === "length") stopReason = "max_tokens";
  else if (finishReason === "stop") stopReason = "end_turn";

  return {
    id: `msg_${randomUUID().replace(/-/g, "").slice(0, 24)}`,
    type: "message",
    role: "assistant",
    content: content.length > 0 ? content : [{ type: "text", text: "" }],
    model: requestModel,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming helpers: Anthropic SSE format
// ─────────────────────────────────────────────────────────────────────────────

export function anthropicStreamMessageStart(
  messageId: string,
  model: string,
  inputTokens: number
): string {
  const msg = {
    id: messageId,
    type: "message",
    role: "assistant",
    content: [],
    model,
    stop_reason: null,
    stop_sequence: null,
    usage: { input_tokens: inputTokens, output_tokens: 0 },
  };
  return `event: message_start\ndata: ${JSON.stringify({ type: "message_start", message: msg })}\n\n`;
}

export function anthropicStreamContentBlockStart(index: number): string {
  return `event: content_block_start\ndata: ${JSON.stringify({
    type: "content_block_start",
    index,
    content_block: { type: "text", text: "" },
  })}\n\n`;
}

export function anthropicStreamContentBlockStartForToolUse(
  index: number,
  id: string,
  name: string
): string {
  return `event: content_block_start\ndata: ${JSON.stringify({
    type: "content_block_start",
    index,
    content_block: { type: "tool_use", id, name, input: {} },
  })}\n\n`;
}

export function anthropicStreamContentBlockDelta(index: number, text: string): string {
  return `event: content_block_delta\ndata: ${JSON.stringify({
    type: "content_block_delta",
    index,
    delta: { type: "text_delta", text },
  })}\n\n`;
}

export function anthropicStreamContentBlockDeltaForToolUse(
  index: number,
  partialJson: string
): string {
  return `event: content_block_delta\ndata: ${JSON.stringify({
    type: "content_block_delta",
    index,
    delta: { type: "input_json_delta", partial_json: partialJson },
  })}\n\n`;
}

export function anthropicStreamContentBlockStop(index: number): string {
  return `event: content_block_stop\ndata: ${JSON.stringify({
    type: "content_block_stop",
    index,
  })}\n\n`;
}

export function anthropicStreamMessageDelta(
  stopReason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use",
  outputTokens: number
): string {
  return `event: message_delta\ndata: ${JSON.stringify({
    type: "message_delta",
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: outputTokens },
  })}\n\n`;
}

export function anthropicStreamMessageStop(): string {
  return `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`;
}

export function anthropicStreamPing(): string {
  return `event: ping\ndata: ${JSON.stringify({ type: "ping" })}\n\n`;
}

export function anthropicErrorResponse(message: string, statusCode: number): Response {
  const typeMap: Record<number, string> = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
  };
  return new Response(
    JSON.stringify({
      type: "error",
      error: {
        type: typeMap[statusCode] ?? "api_error",
        message,
      },
    }),
    { status: statusCode, headers: { "Content-Type": "application/json" } }
  );
}
