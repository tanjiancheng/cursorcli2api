/**
 * Google Gemini API compatibility module.
 * Converts between Gemini generateContent API format and internal ChatCompletionRequest.
 *
 * Inbound: Gemini contents (parts with text, functionCall, functionResponse, inlineData)
 *          → OpenAI ChatMessage format.
 * Outbound: OpenAI ChatCompletion response → Gemini generateContent response.
 */

import { z } from "zod";
import { randomUUID } from "crypto";
import type { ChatCompletionRequest, ChatMessage } from "./openai-compat.js";
import { normalizeMessageContent } from "./openai-compat.js";

// ─────────────────────────────────────────────────────────────────────────────
// Zod schemas
// ─────────────────────────────────────────────────────────────────────────────

const GeminiPartSchema = z.object({}).passthrough();

const GeminiContentSchema = z
  .object({
    role: z.string().optional(),
    parts: z.array(GeminiPartSchema),
  })
  .passthrough();

export const GeminiGenerateContentRequestSchema = z
  .object({
    contents: z.array(GeminiContentSchema),
    systemInstruction: GeminiContentSchema.nullable().optional(),
    generationConfig: z.object({}).passthrough().optional(),
    tools: z.array(z.object({}).passthrough()).optional(),
    toolConfig: z.object({}).passthrough().optional(),
    safetySettings: z.array(z.object({}).passthrough()).optional(),
  })
  .passthrough();

export type GeminiGenerateContentRequest = z.infer<typeof GeminiGenerateContentRequestSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// Request conversion: Gemini → internal ChatCompletionRequest
// ─────────────────────────────────────────────────────────────────────────────

function geminiToolsToOpenai(
  tools: Array<Record<string, unknown>>
): Array<Record<string, unknown>> {
  const result: Array<Record<string, unknown>> = [];
  for (const tool of tools) {
    const declarations = (tool.functionDeclarations ?? tool.function_declarations) as
      | Array<Record<string, unknown>>
      | undefined;
    if (!Array.isArray(declarations)) continue;
    for (const decl of declarations) {
      if (typeof decl !== "object" || decl === null) continue;
      const fn: Record<string, unknown> = { name: decl.name };
      if (decl.description) fn.description = decl.description;
      if (decl.parameters) fn.parameters = decl.parameters;
      result.push({ type: "function", function: fn });
    }
  }
  return result;
}

export function geminiRequestToChatRequest(
  req: GeminiGenerateContentRequest,
  model: string,
  stream: boolean
): ChatCompletionRequest {
  const messages: ChatMessage[] = [];

  if (req.systemInstruction) {
    const parts = (req.systemInstruction.parts ?? []) as Array<Record<string, unknown>>;
    const texts = parts
      .filter((p) => typeof p.text === "string")
      .map((p) => p.text as string);
    if (texts.length > 0) {
      messages.push({ role: "system", content: texts.join("\n") });
    }
  }

  for (const content of req.contents) {
    const geminiRole = ((content.role as string) ?? "user").toLowerCase();
    const parts = (content.parts ?? []) as Array<Record<string, unknown>>;

    const textParts: string[] = [];
    const imageParts: Array<Record<string, unknown>> = [];
    const toolCalls: Array<Record<string, unknown>> = [];
    const toolResults: Array<{ name: string; content: string }> = [];

    for (const p of parts) {
      if (typeof p.text === "string") {
        textParts.push(p.text);
      } else if (p.inlineData && typeof p.inlineData === "object") {
        const inline = p.inlineData as Record<string, unknown>;
        const mime = String(inline.mimeType ?? inline.mime_type ?? "image/png");
        const data = String(inline.data ?? "");
        imageParts.push({
          type: "image_url",
          image_url: { url: `data:${mime};base64,${data}` },
        });
      } else if (p.functionCall && typeof p.functionCall === "object") {
        const fc = p.functionCall as Record<string, unknown>;
        toolCalls.push({
          id: `call_${randomUUID().replace(/-/g, "").slice(0, 24)}`,
          type: "function",
          function: {
            name: String(fc.name ?? ""),
            arguments:
              typeof fc.args === "object" && fc.args !== null
                ? JSON.stringify(fc.args)
                : String(fc.args ?? "{}"),
          },
        });
      } else if (p.functionResponse && typeof p.functionResponse === "object") {
        const fr = p.functionResponse as Record<string, unknown>;
        const name = String(fr.name ?? "tool");
        const resp = fr.response as Record<string, unknown> | undefined;
        const rc =
          resp?.content != null ? String(resp.content) : JSON.stringify(resp ?? {});
        toolResults.push({ name, content: rc });
      }
    }

    if (toolResults.length > 0) {
      for (const tr of toolResults) {
        const msg = { role: "tool" as const, content: tr.content } as ChatMessage;
        (msg as Record<string, unknown>).tool_call_id = `call_${tr.name}`;
        messages.push(msg);
      }
      continue;
    }

    const role: "user" | "assistant" = geminiRole === "model" ? "assistant" : "user";
    let msgContent: unknown;
    if (imageParts.length > 0) {
      const arr: Array<Record<string, unknown>> = textParts.map((t) => ({
        type: "text",
        text: t,
      }));
      arr.push(...imageParts);
      msgContent = arr;
    } else {
      msgContent = textParts.join("");
    }

    const msg: ChatMessage = { role, content: msgContent };
    if (toolCalls.length > 0) {
      (msg as Record<string, unknown>).tool_calls = toolCalls;
    }
    messages.push(msg);
  }

  const genConfig = (req.generationConfig ?? {}) as Record<string, unknown>;
  const chatReq: ChatCompletionRequest = {
    model,
    messages,
    stream,
    max_tokens:
      (typeof genConfig.maxOutputTokens === "number"
        ? genConfig.maxOutputTokens
        : null) as number | null,
  };

  if (req.tools && Array.isArray(req.tools) && req.tools.length > 0) {
    const converted = geminiToolsToOpenai(
      req.tools as Array<Record<string, unknown>>
    );
    if (converted.length > 0) {
      (chatReq as Record<string, unknown>).tools = converted;
    }
  }

  if (typeof genConfig.temperature === "number") {
    (chatReq as Record<string, unknown>).temperature = genConfig.temperature;
  }
  if (typeof genConfig.topP === "number") {
    (chatReq as Record<string, unknown>).top_p = genConfig.topP;
  }

  return chatReq;
}

// ─────────────────────────────────────────────────────────────────────────────
// Response conversion: internal → Gemini generateContent format
// ─────────────────────────────────────────────────────────────────────────────

function finishReasonToGemini(reason: string | null | undefined): string {
  switch (reason) {
    case "stop":
      return "STOP";
    case "length":
      return "MAX_TOKENS";
    case "tool_calls":
      return "STOP";
    case "content_filter":
      return "SAFETY";
    default:
      return "STOP";
  }
}

function parseToolCallArgs(fn: Record<string, unknown> | undefined): Record<string, unknown> {
  if (!fn) return {};
  if (typeof fn.arguments === "string") {
    try {
      return JSON.parse(fn.arguments) as Record<string, unknown>;
    } catch {
      return {};
    }
  }
  if (typeof fn.arguments === "object" && fn.arguments !== null) {
    return fn.arguments as Record<string, unknown>;
  }
  return {};
}

export function chatCompletionToGeminiResponse(
  chat: Record<string, unknown>,
  modelVersion: string
): Record<string, unknown> {
  const choices = (chat.choices as Array<Record<string, unknown>>) ?? [];
  const parts: Array<Record<string, unknown>> = [];
  let finishReason = "STOP";

  if (choices.length > 0) {
    const first = choices[0];
    const message = (first.message as Record<string, unknown>) ?? {};
    const text = normalizeMessageContent(message.content);
    if (text) parts.push({ text });

    const toolCalls = message.tool_calls as Array<Record<string, unknown>> | undefined;
    if (toolCalls && Array.isArray(toolCalls)) {
      for (const tc of toolCalls) {
        const fn = tc.function as Record<string, unknown> | undefined;
        parts.push({
          functionCall: {
            name: typeof fn?.name === "string" ? fn.name : "",
            args: parseToolCallArgs(fn),
          },
        });
      }
    }

    finishReason = finishReasonToGemini(first.finish_reason as string);
  }

  const resp: Record<string, unknown> = {
    candidates: [
      {
        content: {
          role: "model",
          parts: parts.length > 0 ? parts : [{ text: "" }],
        },
        finishReason,
        index: 0,
      },
    ],
    modelVersion,
  };

  const usage = chat.usage as Record<string, unknown> | undefined;
  if (usage && typeof usage === "object") {
    resp.usageMetadata = {
      promptTokenCount: Math.floor(Number(usage.prompt_tokens) || 0),
      candidatesTokenCount: Math.floor(Number(usage.completion_tokens) || 0),
      totalTokenCount: Math.floor(Number(usage.total_tokens) || 0),
    };
  }

  return resp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming helpers: Gemini SSE format
// ─────────────────────────────────────────────────────────────────────────────

export function geminiStreamTextChunk(text: string, modelVersion: string): string {
  const data = {
    candidates: [
      { content: { role: "model", parts: [{ text }] }, index: 0 },
    ],
    modelVersion,
  };
  return `data: ${JSON.stringify(data)}\n\n`;
}

export function geminiStreamToolCallChunk(
  toolCalls: Array<Record<string, unknown>>,
  modelVersion: string
): string {
  const parts: Array<Record<string, unknown>> = [];
  for (const tc of toolCalls) {
    const fn = tc.function as Record<string, unknown> | undefined;
    parts.push({
      functionCall: {
        name: typeof fn?.name === "string" ? fn.name : "",
        args: parseToolCallArgs(fn),
      },
    });
  }
  const data = {
    candidates: [
      { content: { role: "model", parts }, index: 0 },
    ],
    modelVersion,
  };
  return `data: ${JSON.stringify(data)}\n\n`;
}

export function geminiStreamFinalChunk(
  finishReason: string,
  usage: {
    promptTokenCount: number;
    candidatesTokenCount: number;
    totalTokenCount: number;
  } | null,
  modelVersion: string
): string {
  const data: Record<string, unknown> = {
    candidates: [
      {
        content: { role: "model", parts: [] },
        finishReason,
        index: 0,
      },
    ],
    modelVersion,
  };
  if (usage) data.usageMetadata = usage;
  return `data: ${JSON.stringify(data)}\n\n`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Error response
// ─────────────────────────────────────────────────────────────────────────────

const GEMINI_ERROR_STATUS_MAP: Record<number, string> = {
  400: "INVALID_ARGUMENT",
  401: "UNAUTHENTICATED",
  403: "PERMISSION_DENIED",
  404: "NOT_FOUND",
  429: "RESOURCE_EXHAUSTED",
  500: "INTERNAL",
  503: "UNAVAILABLE",
};

export function geminiErrorResponse(message: string, statusCode: number): Response {
  return new Response(
    JSON.stringify({
      error: {
        code: statusCode,
        message,
        status: GEMINI_ERROR_STATUS_MAP[statusCode] ?? "INTERNAL",
      },
    }),
    { status: statusCode, headers: { "Content-Type": "application/json" } }
  );
}
