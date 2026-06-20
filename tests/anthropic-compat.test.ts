import test from "node:test";
import assert from "node:assert/strict";

import {
  anthropicRequestToChatRequest,
  chatCompletionToAnthropicResponse,
} from "../src/lib/anthropic-compat.js";
import type { AnthropicMessagesRequest } from "../src/lib/anthropic-compat.js";

test("inbound tool_use blocks → assistant message with tool_calls", () => {
  const req: AnthropicMessagesRequest = {
    model: "claude:sonnet",
    messages: [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me check" },
          {
            type: "tool_use",
            id: "toolu_abc",
            name: "get_weather",
            input: { city: "Beijing" },
          },
        ],
      },
      {
        role: "user",
        content: "next",
      },
    ],
    max_tokens: 100,
  };

  const chat = anthropicRequestToChatRequest(req);

  assert.equal(chat.messages.length, 2);
  assert.equal(chat.messages[0].role, "assistant");

  const toolCalls = (chat.messages[0] as Record<string, unknown>).tool_calls as Array<Record<string, unknown>>;
  assert.ok(toolCalls, "should have tool_calls");
  assert.equal(toolCalls.length, 1);
  assert.equal(toolCalls[0].id, "toolu_abc");
  assert.equal((toolCalls[0].function as Record<string, unknown>).name, "get_weather");
});

test("inbound tool_result blocks → tool role messages", () => {
  const req: AnthropicMessagesRequest = {
    model: "claude:sonnet",
    messages: [
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_xyz", content: "sunny" },
          { type: "text", text: "what next?" },
        ],
      },
    ],
    max_tokens: 100,
  };

  const chat = anthropicRequestToChatRequest(req);

  assert.equal(chat.messages.length, 2);
  assert.equal(chat.messages[0].role, "user");
  assert.equal(chat.messages[1].role, "tool");

  const toolMsg = chat.messages[1] as Record<string, unknown>;
  assert.equal(toolMsg.tool_call_id, "toolu_xyz");
  assert.equal(toolMsg.content, "sunny");
});

test("inbound thinking blocks → text content", () => {
  const req: AnthropicMessagesRequest = {
    model: "claude:sonnet",
    messages: [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "inner monologue" },
          { type: "text", text: "actual reply" },
        ],
      },
    ],
    max_tokens: 100,
  };

  const chat = anthropicRequestToChatRequest(req);

  assert.equal(chat.messages.length, 1);
  assert.equal(chat.messages[0].role, "assistant");
  const content = chat.messages[0].content as Array<Record<string, unknown>>;
  assert.ok(Array.isArray(content));
  const texts = content.filter((c) => c.type === "text");
  assert.equal(texts.length, 2);
});

test("outbound tool_calls → tool_use content blocks", () => {
  const openaiResponse = {
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: "Let me check",
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: {
                name: "get_weather",
                arguments: '{"city":"Beijing"}',
              },
            },
          ],
        },
        finish_reason: "tool_calls",
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: 20 },
  };

  const anthropic = chatCompletionToAnthropicResponse(openaiResponse, "claude:sonnet");

  assert.equal(anthropic.stop_reason, "tool_use");
  assert.equal(anthropic.content.length, 2);

  const textBlock = anthropic.content.find((c) => c.type === "text");
  assert.ok(textBlock);
  assert.equal(textBlock!.text, "Let me check");

  const toolBlock = anthropic.content.find((c) => c.type === "tool_use");
  assert.ok(toolBlock);
  assert.equal(toolBlock!.name, "get_weather");
  assert.deepEqual(toolBlock!.input, { city: "Beijing" });
});

test("outbound text-only response", () => {
  const openaiResponse = {
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: "Hello" },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 5, completion_tokens: 3 },
  };

  const anthropic = chatCompletionToAnthropicResponse(openaiResponse, "claude:sonnet");

  assert.equal(anthropic.stop_reason, "end_turn");
  assert.equal(anthropic.content.length, 1);
  assert.equal(anthropic.content[0].type, "text");
  assert.equal(anthropic.content[0].text, "Hello");
});

test("inbound plain text messages pass through unchanged", () => {
  const req: AnthropicMessagesRequest = {
    model: "claude:sonnet",
    messages: [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi there" },
    ],
    max_tokens: 100,
  };

  const chat = anthropicRequestToChatRequest(req);

  assert.equal(chat.messages.length, 2);
  assert.equal(chat.messages[0].role, "user");
  assert.equal(chat.messages[0].content, "hello");
  assert.equal(chat.messages[1].role, "assistant");
  assert.equal(chat.messages[1].content, "hi there");
});

test("inbound system role in messages array", () => {
  const req: AnthropicMessagesRequest = {
    model: "auto",
    messages: [
      { role: "system", content: "You are helpful" },
      { role: "user", content: "hello" },
    ],
    max_tokens: 100,
  };

  const chat = anthropicRequestToChatRequest(req);

  assert.equal(chat.messages.length, 2);
  assert.equal(chat.messages[0].role, "system");
  assert.equal(chat.messages[0].content, "You are helpful");
  assert.equal(chat.messages[1].role, "user");
});

test("inbound with system prompt", () => {
  const req: AnthropicMessagesRequest = {
    model: "claude:sonnet",
    messages: [{ role: "user", content: "hello" }],
    max_tokens: 100,
    system: "You are helpful",
  };

  const chat = anthropicRequestToChatRequest(req);

  assert.equal(chat.messages.length, 2);
  assert.equal(chat.messages[0].role, "system");
  assert.equal(chat.messages[0].content, "You are helpful");
});

test("inbound tool_use without text content → empty text + tool_calls", () => {
  const req: AnthropicMessagesRequest = {
    model: "claude:sonnet",
    messages: [
      {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "toolu_1",
            name: "search",
            input: { query: "rust" },
          },
        ],
      },
    ],
    max_tokens: 100,
  };

  const chat = anthropicRequestToChatRequest(req);

  const toolCalls = (chat.messages[0] as Record<string, unknown>).tool_calls as Array<Record<string, unknown>>;
  assert.ok(toolCalls);
  assert.equal(toolCalls.length, 1);
  assert.equal((toolCalls[0].function as Record<string, unknown>).name, "search");
});
