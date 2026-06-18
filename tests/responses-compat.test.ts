import test from "node:test";
import assert from "node:assert/strict";

import {
  createResponsesStreamFromChatCompletion,
  createResponsesStreamState,
  responsesStreamInitEvents,
  responsesStreamTextDelta,
  responsesStreamFinalize,
} from "../src/lib/responses-compat.js";

test("responses stream init emits Codex-compatible lifecycle events", () => {
  const state = createResponsesStreamState("composer-2.5-fast");
  const events = responsesStreamInitEvents(state);

  assert.equal(events.length, 4);
  assert.match(events[0], /event: response\.created/);
  assert.match(events[1], /event: response\.in_progress/);
  assert.match(events[2], /event: response\.output_item\.added/);
  assert.match(events[3], /event: response\.content_part\.added/);

  const created = JSON.parse(events[0].split("data: ")[1].trim());
  assert.equal(created.type, "response.created");
  assert.equal(created.response.model, "composer-2.5-fast");
});

test("responses stream delta and finalize produce output_text and completed events", () => {
  const state = createResponsesStreamState("auto");
  const delta = responsesStreamTextDelta(state, "Hello");
  assert.match(delta, /response\.output_text\.delta/);
  assert.match(delta, /"delta":"Hello"/);

  const done = responsesStreamFinalize(state);
  assert.equal(done.length, 4);
  assert.match(done[0], /response\.output_text\.done/);
  assert.match(done[3], /response\.completed/);

  const completed = JSON.parse(done[3].split("data: ")[1].trim());
  assert.equal(completed.response.status, "completed");
  assert.equal(completed.response.output[0].content[0].text, "Hello");
});

test("createResponsesStreamFromChatCompletion converts chat.completion.chunk SSE", async () => {
  const upstream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({
            id: "chatcmpl-test",
            object: "chat.completion.chunk",
            choices: [{ index: 0, delta: { content: "Hi" }, finish_reason: null }],
          })}\n\n`
        )
      );
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({
            id: "chatcmpl-test",
            object: "chat.completion.chunk",
            choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          })}\n\n`
        )
      );
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  const reader = createResponsesStreamFromChatCompletion(upstream, "auto").getReader();
  const decoder = new TextDecoder();
  let output = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    output += decoder.decode(value);
  }

  assert.match(output, /response\.created/);
  assert.match(output, /response\.output_text\.delta/);
  assert.match(output, /"delta":"Hi"/);
  assert.match(output, /response\.completed/);
  assert.match(output, /"text":"Hi"/);
});
