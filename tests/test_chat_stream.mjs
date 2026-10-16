import assert from "node:assert/strict";
import test from "node:test";
import { ChatStream } from "../tools/vrambridge/web/chat-stream.mjs";

const encode = (data) => new TextEncoder().encode(data);
const event = (data) => `data: ${JSON.stringify(data)}\n\n`;
const delta = (data) => ({ choices: [{ delta: data }] });

test("reasoning is retained separately while waiting for a final answer", () => {
  const stream = new ChatStream();
  stream.push(encode(event(delta({ reasoning_content: "Working on it" }))));
  assert.equal(stream.content, "");
  assert.equal(stream.reasoning, "Working on it");
  assert.equal(stream.done, false);
  stream.push(encode(event(delta({ content: "323" })) + "data: [DONE]\n\n"));
  assert.equal(stream.content, "323");
  assert.equal(stream.warning(8192), "");
});

test("fragmented UTF-8, CRLF, heartbeats, and batched events are parsed", () => {
  const stream = new ChatStream();
  const wire = ": heartbeat\r\n\r\n" + event(delta({ reasoning_content: "\u2713" }))
    + event(delta({ content: "\u4f60\u597d" })) + "data: [DONE]\r\n\r\n";
  for (const byte of encode(wire)) stream.push(new Uint8Array([byte]));
  assert.equal(stream.reasoning, "\u2713");
  assert.equal(stream.content, "\u4f60\u597d");
  assert.equal(stream.done, true);
});

test("live speed and token counts use backend metrics, never SSE chunk counts", () => {
  const stream = new ChatStream();
  stream.push(encode(event({
    ...delta({ reasoning_content: "One large chunk" }),
    timings: { predicted_n: 256, predicted_per_second: 123.456 },
  })));
  const text = stream.metrics(2.5, "Thinking");
  assert.match(text, /Thinking/);
  assert.match(text, /123\.5 tok\/s/);
  assert.match(text, /256 output tokens/);
  assert.match(text, /2\.5s total/);
});

test("final usage-only events preserve speed and include reasoning tokens", () => {
  const stream = new ChatStream();
  stream.push(encode(event({
    ...delta({ content: "Answer" }),
    timings: { predicted_n: 729, predicted_per_second: 123.5 },
  })));
  stream.push(encode(event({ choices: [{ delta: {}, finish_reason: "stop" }] })
    + event({ choices: [], usage: { completion_tokens: 730, prompt_tokens: 18 } })
    + "data: [DONE]\n\n"));
  assert.equal(stream.finishReason, "stop");
  assert.match(stream.metrics(6, "Complete"), /730 output tokens/);
  assert.match(stream.metrics(6, "Complete"), /123\.5 tok\/s/);
});

test("reasoning-only truncation explains the token limit instead of showing a blank answer", () => {
  const stream = new ChatStream();
  stream.push(encode(event(delta({ reasoning_content: "Still thinking" }))
    + event({ choices: [{ delta: {}, finish_reason: "length" }] }) + "data: [DONE]\n\n"));
  assert.match(stream.warning(8192), /before a final answer/);
  assert.ok(stream.warning(8192).includes((8192).toLocaleString()));
  assert.match(stream.warning(8192), /larger context/);
});

test("partial answer truncation warns without discarding the answer", () => {
  const stream = new ChatStream();
  stream.push(encode(event({
    choices: [{ delta: { content: "Partial answer" }, finish_reason: "length" }],
  }) + "data: [DONE]\n\n"));
  assert.equal(stream.content, "Partial answer");
  assert.match(stream.warning(512), /may be incomplete/);
});

test("missing telemetry is explicit and never fabricates a token rate", () => {
  const stream = new ChatStream();
  assert.match(stream.metrics(1, "Waiting for model"), /speed pending/);
  stream.push(encode(event(delta({ content: "Hello" })) + "data: [DONE]\n\n"));
  assert.match(stream.metrics(1, "Complete"), /speed unavailable/);
  assert.doesNotMatch(stream.metrics(1, "Complete"), /tok\/s|output tokens/);
});

test("empty and reasoning-only completed responses have actionable warnings", () => {
  const empty = new ChatStream();
  empty.push(encode("data: [DONE]\n\n"));
  assert.match(empty.warning(8192), /returned no answer/);
  const thinking = new ChatStream();
  thinking.push(encode(event(delta({ reasoning_content: "Thinking" })) + "data: [DONE]\n\n"));
  assert.match(thinking.warning(8192), /reasoning but no final answer/);
});

test("backend errors and malformed JSON are surfaced", () => {
  assert.throws(() => new ChatStream().push(encode(event({ error: { message: "Context full" } }))), /Context full/);
  assert.throws(() => new ChatStream().push(encode("data: not-json\n\n")), SyntaxError);
});

test("EOF without DONE is not a successful completion", () => {
  const stream = new ChatStream();
  stream.push(encode(event(delta({ content: "Partial" }))));
  assert.throws(() => stream.push(undefined, true), /ended unexpectedly/);
  assert.equal(stream.content, "Partial");
});
