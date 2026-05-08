/**
 * Tests for stream-json-cli.ts — subprocess lifecycle management.
 *
 * Uses real child_process.spawn (echo + node -e scripts) to produce
 * controllable NDJSON output without mocking.
 */

import test from "node:test";
import assert from "node:assert/strict";

import { iterStreamJsonEvents, TextAssembler } from "../src/providers/stream-json-cli.js";

test("iterStreamJsonEvents yields parsed NDJSON events from subprocess stdout", async () => {
  const events: Record<string, unknown>[] = [];
  // Use setTimeout to separate writes into different ticks, avoiding
  // the readline once('line') race when both lines arrive in one chunk.
  const lines = [
    `process.stdout.write('{"type":"test","n":1}\\n');`,
    `setTimeout(() => { process.stdout.write('{"type":"test","n":2}\\n'); }, 10);`,
  ].join("");
  for await (const evt of iterStreamJsonEvents({
    cmd: ["node", "-e", lines],
    timeoutMs: 5000,
  })) {
    events.push(evt);
  }
  assert.equal(events.length, 2);
  assert.equal(events[0].type, "test");
  assert.equal(events[0].n, 1);
  assert.equal(events[1].type, "test");
  assert.equal(events[1].n, 2);
});

test("iterStreamJsonEvents handles killOnResult", async () => {
  const events: Record<string, unknown>[] = [];
  const script = [
    `process.stdout.write('{"type":"assistant","message":{"content":"hello"}}\\n');`,
    `setTimeout(() => { process.stdout.write('{"type":"result","result":"done"}\\n'); }, 10);`,
    // Would output after result but process should be killed before this
    `setTimeout(() => process.stdout.write('{"type":"after","x":1}\\n'), 15000);`,
  ].join(" ");
  for await (const evt of iterStreamJsonEvents({
    cmd: ["node", "-e", script],
    timeoutMs: 5000,
    killOnResult: true,
  })) {
    events.push(evt);
  }
  assert.equal(events.length, 2);
  assert.equal(events[0].type, "assistant");
  assert.equal(events[1].type, "result");
});

test("iterStreamJsonEvents propagates subprocess non-zero exit as error", async () => {
  try {
    for await (const _evt of iterStreamJsonEvents({
      cmd: ["node", "-e", `process.stderr.write('something broke'); process.exit(1);`],
      timeoutMs: 5000,
    })) {
      // should throw before yielding anything useful
    }
    assert.fail("Expected error was not thrown");
  } catch (e) {
    const msg = String(e);
    assert.ok(
      msg.includes("something broke") || msg.includes("subprocess failed") || msg.includes("1"),
      `error message should indicate subprocess failure, got: ${msg}`,
    );
  }
});

test("iterStreamJsonEvents with AbortSignal kills subprocess early", async () => {
  const ac = new AbortController();
  const events: Record<string, unknown>[] = [];

  const script = [
    `process.stdout.write('{"type":"start"}\\n');`,
    // Sleep 30 seconds — abort should kill before this
    `setTimeout(() => process.stdout.write('{"type":"never"}\\n'), 30000);`,
  ].join(" ");

  const iter = iterStreamJsonEvents({
    cmd: ["node", "-e", script],
    timeoutMs: 3000,
    signal: ac.signal,
  });

  // Abort after a short delay
  setTimeout(() => ac.abort(), 50);

  try {
    for await (const evt of iter) {
      events.push(evt);
    }
  } catch (_) {
    // Expected — generator should throw after abort kills the process
  }

  const types = events.map((e) => e.type);
  assert.ok(types.includes("start"), "should have received start event");
  assert.ok(!types.includes("never"), "should NOT have received never event");
});

test("TextAssembler produces clean deltas from partial/full text", () => {
  const a = new TextAssembler();
  assert.equal(a.feed("Hello"), "Hello");
  assert.equal(a.text, "Hello");
  assert.equal(a.feed("Hello World"), " World");
  assert.equal(a.text, "Hello World");
  assert.equal(a.feed("New"), "New");
  assert.equal(a.text, "Hello WorldNew");
  assert.equal(a.feed(""), "");
  assert.equal(a.text, "Hello WorldNew");
});
