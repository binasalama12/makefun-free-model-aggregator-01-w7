#!/usr/bin/env python3
"""Aggregator entry for free-model-aggregator-01, build "lean-fallback".

One process fronts ten free-tier upstreams behind two client-facing wires:
the OpenAI chat-completions shape and the Anthropic messages shape, each
offered as plain JSON or as a server-sent-event stream. For every incoming
request the server picks the best-scoring free model that can actually meet
the request's demands and is still responding, and moves on to the next
candidate whenever an upstream turns a request away with a 402 or a 429.
At startup it tells the trial harness where it lives and what it offers,
then keeps serving until the harness signals that the workload is finished.

Only the standard library is used. Choices are fully reproducible: they
follow from the brief, the request at hand, and the answers already seen
from upstreams -- the clock never influences a routing decision.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Build dials: this wallet's copy tunes each of these values.
# ---------------------------------------------------------------------------
VARIANT_NAME = "lean-fallback"
PREFER_SAME_WIRE_BONUS = 0
REFUSAL_LIMIT = 2
LONG_CONTEXT_CHARS = 3000
LONG_CONTEXT_MIN = 200000
ANTHROPIC_JSON_SAFE = False
STREAM_PIECE_CHARS = 0
TIE_BREAK = "alpha"

FIXED_CREATED = 1767225600
ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
PORT = 8791

# ---------------------------------------------------------------------------
# Tiny HTTP helper built on urllib, with proxies switched off: trials hand
# us plain base URLs, so no proxy should ever get in the way.
# ---------------------------------------------------------------------------
_FETCHER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_url(method, url, payload=None, extra_headers=None, timeout=15):
    """Perform a single HTTP round trip.

    The result is a triple of (status code, headers mapping, raw body).
    A failure to connect at all is reported as status 0 with empty
    headers and an empty body, so callers can treat it like any other
    unsuccessful answer.
    """
    encoded = None
    sent_headers = dict(extra_headers or {})
    if payload is not None:
        if isinstance(payload, bytes):
            encoded = payload
        else:
            encoded = json.dumps(payload).encode("utf-8")
        sent_headers.setdefault("content-type", "application/json")
    outgoing = urllib.request.Request(url, data=encoded, headers=sent_headers, method=method)
    try:
        with _FETCHER.open(outgoing, timeout=timeout) as incoming:
            return incoming.status, dict(incoming.headers.items()), incoming.read()
    except urllib.error.HTTPError as failure:
        failure_headers = dict(failure.headers.items()) if failure.headers else {}
        return failure.code, failure_headers, failure.read()
    except Exception:
        return 0, {}, b""


def fetch_json(method, url, payload=None, extra_headers=None, timeout=15):
    """Like fetch_url, but with the body decoded as JSON when possible."""
    status, headers, raw = fetch_url(method, url, payload, extra_headers, timeout)
    if not raw:
        return status, headers, None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        parsed = None
    return status, headers, parsed


# ---------------------------------------------------------------------------
# Reading the brief: the provider directory, the quality table and the
# workload manifest ride inside the prompt file as embedded JSON blobs.
# ---------------------------------------------------------------------------
def _pull_doc(text, schema_name):
    """Fish one embedded JSON document out of the prompt prose by schema."""
    needle = '"schema":"' + schema_name + '"'
    found = text.find(needle)
    if found < 0:
        needle = '"schema": "' + schema_name + '"'
        found = text.find(needle)
    if found < 0:
        return None
    opening = text.rfind("{", 0, found)
    if opening < 0:
        return None
    try:
        document, _ = json.JSONDecoder().raw_decode(text[opening:])
    except Exception:
        return None
    return document


class Upstream:
    """One provider row from the brief's directory, with its auth details."""

    __slots__ = ("id", "name", "wire", "auth", "key_env", "key", "path")

    def __init__(self, row):
        self.id = row["id"]
        self.name = row.get("name", row["id"])
        self.wire = row["wire"]
        self.auth = row.get("auth", "none")
        self.key_env = row.get("key_env", "")
        self.key = row.get("key", "")
        self.path = row["path"]

    def base(self):
        """The provider's root URL under the trial environment."""
        return ENV_URL + self.path

    def credentials(self):
        """Auth headers for this provider, or nothing when it needs none."""
        kind = self.auth
        if kind == "bearer":
            return {"authorization": "Bearer " + self.key}
        if kind == "x-api-key":
            return {"x-api-key": self.key}
        if kind == "x-goog-api-key":
            return {"x-goog-api-key": self.key}
        return {}


def read_brief():
    """Return (upstreams, rating rows, manifest) from the prompt file.

    An absent or unreadable prompt file yields three empty lists, which
    downstream code treats as "nothing to serve".
    """
    if not PROMPT_FILE:
        return [], [], []
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as handle:
            prose = handle.read()
    except Exception:
        return [], [], []
    directory = _pull_doc(prose, "free-model-aggregator-01/providers/v1") or {}
    scores_doc = _pull_doc(prose, "free-model-aggregator-01/ratings/v1") or {}
    workload_doc = _pull_doc(prose, "free-model-aggregator-01/workload/v1") or {}
    ups = [Upstream(row) for row in directory.get("providers", [])]
    score_rows = scores_doc.get("ratings", [])
    manifest = workload_doc.get("manifest", [])
    return ups, score_rows, manifest

# ---------------------------------------------------------------------------
# The catalog: ask each upstream for its own model list (ground truth),
# and use the brief's quality table only where a probe comes back empty.
# ---------------------------------------------------------------------------
def _scan_models(upstream):
    """List (model id, free flag, context length) from the upstream, or None."""
    if upstream.wire == "gemini":
        target = upstream.base() + "/v1beta/models"
    else:
        target = upstream.base() + "/v1/models"
    # Slow catalog endpoints need a generous window; 15s keeps flaky
    # probes from dropping models that are actually available.
    status, _headers, doc = fetch_json("GET", target, extra_headers=upstream.credentials(), timeout=15)
    if status < 200 or status >= 300:
        return None
    if not isinstance(doc, dict):
        return None
    found = []
    if upstream.wire == "gemini":
        for entry in doc.get("models", []):
            raw_name = str(entry.get("name", ""))
            mid = raw_name.removeprefix("models/")
            if not mid:
                continue
            limit = entry.get("context_length") or entry.get("inputTokenLimit")
            found.append((mid, entry.get("free") is True, limit))
    else:
        for entry in doc.get("data", []):
            mid = entry.get("id")
            if isinstance(mid, str) and mid:
                found.append((mid, entry.get("free") is True, entry.get("context_length")))
    if not found:
        return None
    return found


def assemble_catalog(upstreams, score_rows):
    """Build the catalog document the harness expects at registration."""
    fallback = {}
    for row in score_rows:
        bucket = fallback.setdefault(row["provider"], [])
        bucket.append((row["model"], row.get("free") is True, row.get("context_length")))
    listed = []
    for upstream in upstreams:
        probed = _scan_models(upstream) if ENV_URL else None
        chosen = probed if probed is not None else fallback.get(upstream.id, [])
        models = []
        for mid, free, ctx in chosen:
            models.append({"id": mid, "free": free, "context_length": ctx})
        listed.append({
            "id": upstream.id,
            "base_url": upstream.path,
            "key_env": upstream.key_env,
            "wire": upstream.wire,
            "models": models,
        })
    return {"providers": listed}


def catalog_json(catalog):
    """Serialize a catalog document to its JSON text form."""
    return json.dumps(catalog)


# ---------------------------------------------------------------------------
# Canonical request: the client's request reshaped into a wire-neutral
# form, keeping every part the client sent -- nothing is dropped.
# ---------------------------------------------------------------------------
def _text_of_blocks(content):
    """Pull the plain text out of a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunks)
    return ""


def _picture_part(url):
    """Turn a client image URL into a canonical image part.

    Inline base64 data URLs become embedded image parts; anything else
    stays a reference to the remote URL.
    """
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        marker = ";base64,"
        media = url[5:url.index(marker)]
        payload = url.split(marker, 1)[1]
        return {"type": "image", "media_type": media, "data": payload}
    return {"type": "image_ref", "url": url}


class NormalizedRequest:
    """A request in wire-neutral form.

    Fields: system is a string or None; messages is a list of
    {"role": ..., "parts": [...]} where each part is one of
    {"type": "text", "text": ...}, {"type": "image", "media_type": ...,
    "data": ...} or {"type": "image_ref", "url": ...}; tools holds
    {"name", "description", "parameters"} entries; raw keeps the exact
    client body for same-wire pass-through.
    """

    def __init__(self, api, body):
        self.api = api
        self.raw = body
        self.stream = body.get("stream") is True
        self.max_tokens = None
        self.stops = []
        self.json_mode = False
        self.response_format = None
        self.system = None
        self.messages = []
        self.tools = []
        self.tool_choice = None
        if api == "openai":
            self._ingest_openai(body)
        else:
            self._ingest_anthropic(body)

    def _ingest_openai(self, body):
        limit = body.get("max_tokens")
        if not isinstance(limit, (int, float)):
            limit = body.get("max_completion_tokens")
        if isinstance(limit, (int, float)):
            self.max_tokens = limit
        stop = body.get("stop")
        if isinstance(stop, str):
            self.stops = [stop]
        elif isinstance(stop, list):
            self.stops = [item for item in stop if isinstance(item, str)]
        fmt = body.get("response_format")
        if isinstance(fmt, dict):
            self.response_format = fmt
            self.json_mode = fmt.get("type") in ("json_object", "json_schema")
        self.tool_choice = body.get("tool_choice")
        for tool in body.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if not isinstance(fn, dict):
                continue
            if not isinstance(fn.get("name"), str):
                continue
            self.tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object"},
            })
        system_chunks = []
        for msg in body.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                text = _text_of_blocks(content)
                if text:
                    system_chunks.append(text)
                continue
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image_url":
                        image_url = block.get("image_url")
                        if isinstance(image_url, dict):
                            parts.append(_picture_part(image_url.get("url", "")))
            self.messages.append({"role": role or "user", "parts": parts})
        if system_chunks:
            self.system = "\n".join(system_chunks)

    def _ingest_anthropic(self, body):
        limit = body.get("max_tokens")
        if isinstance(limit, (int, float)):
            self.max_tokens = limit
        stops = body.get("stop_sequences")
        if isinstance(stops, list):
            self.stops = [item for item in stops if isinstance(item, str)]
        self.tool_choice = body.get("tool_choice")
        system = body.get("system")
        if isinstance(system, str):
            self.system = system or None
        elif isinstance(system, list):
            text = _text_of_blocks(system)
            self.system = text or None
        for tool in body.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            if not isinstance(tool.get("name"), str):
                continue
            self.tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object"},
            })
        for msg in body.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image":
                        src = block.get("source")
                        if not isinstance(src, dict):
                            continue
                        if src.get("type") == "base64" and isinstance(src.get("data"), str):
                            parts.append({
                                "type": "image",
                                "media_type": src.get("media_type", "image/png"),
                                "data": src["data"],
                            })
                        elif src.get("type") == "url":
                            parts.append({"type": "image_ref", "url": src.get("url", "")})
            self.messages.append({"role": msg.get("role") or "user", "parts": parts})

    # -- What this request demands, read off the request itself ------------
    @property
    def wants_tools(self):
        """True when the client supplied at least one tool definition."""
        return len(self.tools) > 0

    @property
    def wants_vision(self):
        """True when any message carries an image part or image reference."""
        for msg in self.messages:
            for part in msg["parts"]:
                if part["type"] in ("image", "image_ref"):
                    return True
        return False

    @property
    def last_user_text(self):
        """Text of the newest user message, or "" when there is none."""
        for msg in reversed(self.messages):
            if msg["role"] != "user":
                continue
            texts = [p["text"] for p in msg["parts"] if p["type"] == "text"]
            return "\n".join(texts)
        return ""

# ---------------------------------------------------------------------------
# Translating the canonical request onto each upstream wire.
# ---------------------------------------------------------------------------
def _to_openai_msgs(req):
    """Canonical messages rewritten in OpenAI chat format."""
    converted = []
    if req.system is not None:
        converted.append({"role": "system", "content": req.system})
    for msg in req.messages:
        parts = msg["parts"]
        text_only = all(p["type"] == "text" for p in parts)
        if text_only:
            joined = "\n".join(p["text"] for p in parts)
            converted.append({"role": msg["role"], "content": joined})
            continue
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                data_url = "data:" + p["media_type"] + ";base64," + p["data"]
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            else:
                content.append({"type": "image_url", "image_url": {"url": p["url"]}})
        converted.append({"role": msg["role"], "content": content})
    return converted


def _to_anthropic_msgs(req):
    """Canonical messages rewritten in Anthropic messages format."""
    converted = []
    for msg in req.messages:
        parts = msg["parts"]
        text_only = all(p["type"] == "text" for p in parts)
        if text_only:
            joined = "\n".join(p["text"] for p in parts)
            converted.append({"role": msg["role"], "content": joined})
            continue
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": p["media_type"], "data": p["data"]}})
            else:
                content.append({"type": "image", "source": {"type": "url", "url": p["url"]}})
        converted.append({"role": msg["role"], "content": content})
    return converted


def _to_gemini_payload(req):
    """Canonical request rewritten as a Gemini generateContent body."""
    contents = []
    for msg in req.messages:
        gemini_parts = []
        for p in msg["parts"]:
            if p["type"] == "text":
                gemini_parts.append({"text": p["text"]})
            elif p["type"] == "image":
                gemini_parts.append({"inlineData": {"mimeType": p["media_type"], "data": p["data"]}})
            else:
                gemini_parts.append({"fileData": {"fileUri": p["url"]}})
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": gemini_parts})
    payload = {"contents": contents}
    if req.system is not None:
        payload["systemInstruction"] = {"parts": [{"text": req.system}]}
    if req.tools:
        declarations = []
        for tool in req.tools:
            declarations.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            })
        payload["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    if req.max_tokens is not None:
        config["maxOutputTokens"] = req.max_tokens
    if req.stops:
        config["stopSequences"] = list(req.stops)
    if req.json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        payload["generationConfig"] = config
    return payload


def _translate_tool_choice(wire, req):
    """Map the client's tool_choice onto the target wire.

    Returns (translated choice, usable flag). A choice the target wire
    cannot express comes back as (None, False) and is left out.
    """
    choice = req.tool_choice
    if choice is None:
        return None, False
    if wire == "openai":
        return choice, True
    if wire == "anthropic":
        if isinstance(choice, str):
            kind = choice if choice in ("auto", "any", "none") else "auto"
            return {"type": kind}, True
        if isinstance(choice, dict):
            if choice.get("type") == "function":
                fn = choice.get("function")
                if isinstance(fn, dict):
                    return {"type": "tool", "name": fn.get("name")}, True
            if choice.get("type") in ("auto", "any", "none"):
                return {"type": choice["type"]}, True
        return None, False
    return None, False


def make_upstream_call(upstream, model_id, req):
    """Return (url, headers, body dict) for one inference call upstream."""
    headers = upstream.credentials()
    if upstream.wire == "openai":
        if req.api == "openai":
            # Same wire: forward the client's own body verbatim, retargeted.
            body = json.loads(json.dumps(req.raw))
            body["model"] = model_id
        else:
            body = {"model": model_id, "messages": _to_openai_msgs(req)}
            if req.max_tokens is not None:
                body["max_tokens"] = req.max_tokens
            if req.stops:
                body["stop"] = list(req.stops)
            if req.tools:
                body["tools"] = [
                    {"type": "function", "function": {
                        "name": t["name"], "description": t["description"],
                        "parameters": t["parameters"]}}
                    for t in req.tools
                ]
            translated, usable = _translate_tool_choice("openai", req)
            if usable:
                body["tool_choice"] = translated
            if req.stream:
                body["stream"] = True
        if req.stream:
            body["stream"] = True
            stream_opts = dict(body.get("stream_options") or {})
            stream_opts["include_usage"] = True
            body["stream_options"] = stream_opts
        return upstream.base() + "/v1/chat/completions", headers, body
    if upstream.wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        if req.api == "anthropic":
            body = json.loads(json.dumps(req.raw))
            body["model"] = model_id
        else:
            body = {"model": model_id, "messages": _to_anthropic_msgs(req)}
            if req.system is not None:
                body["system"] = req.system
            if req.max_tokens is not None:
                body["max_tokens"] = req.max_tokens
            if req.stops:
                body["stop_sequences"] = list(req.stops)
            if req.tools:
                body["tools"] = [
                    {"name": t["name"], "description": t["description"],
                     "input_schema": t["parameters"]}
                    for t in req.tools
                ]
            translated, usable = _translate_tool_choice("anthropic", req)
            if usable:
                body["tool_choice"] = translated
            if req.stream:
                body["stream"] = True
        return upstream.base() + "/v1/messages", headers, body
    # Gemini wire: the model rides in the URL, never in the body.
    body = _to_gemini_payload(req)
    verb = "streamGenerateContent" if req.stream else "generateContent"
    quoted = urllib.parse.quote(model_id, safe="")
    url = upstream.base() + "/v1beta/models/" + quoted + ":" + verb
    if req.stream:
        url += "?alt=sse"
    return url, headers, body


# ---------------------------------------------------------------------------
# Reading an upstream answer back into (text, pieces, stop, usage in/out).
# ---------------------------------------------------------------------------
def _sse_frames(raw):
    """Split a server-sent-event body into (event name, data text) pairs."""
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    frames = []
    for block in text.split("\n\n"):
        event_name = None
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        if data_lines:
            frames.append((event_name, "\n".join(data_lines)))
    return frames


def decode_reply(wire, raw, stream):
    """Decode one upstream answer.

    The result is a dict with keys text, pieces, stop and usage, where
    usage is an (input, output) token pair or None. Any malformed or
    unexpected answer yields None so the caller can move on.
    """
    try:
        if wire == "openai":
            if not stream:
                doc = json.loads(raw.decode("utf-8"))
                choice = doc["choices"][0]
                content = choice["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        b.get("text", "") for b in content if isinstance(b, dict))
                usage_doc = doc.get("usage") or {}
                usage = (usage_doc.get("prompt_tokens"), usage_doc.get("completion_tokens"))
                return {"text": content, "pieces": [content],
                        "stop": choice.get("finish_reason"), "usage": usage}
            text = ""
            pieces = []
            stop = None
            usage = None
            for _event, data in _sse_frames(raw):
                if data.strip() == "[DONE]":
                    continue
                chunk = json.loads(data)
                if isinstance(chunk.get("usage"), dict):
                    usage_doc = chunk["usage"]
                    usage = (usage_doc.get("prompt_tokens"),
                             usage_doc.get("completion_tokens"))
                choices = chunk.get("choices")
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    fragment = delta.get("content")
                    if isinstance(fragment, str) and fragment:
                        pieces.append(fragment)
                        text += fragment
                    reason = choice.get("finish_reason")
                    if isinstance(reason, str):
                        stop = reason
            return {"text": text, "pieces": pieces, "stop": stop, "usage": usage}
        if wire == "anthropic":
            if not stream:
                doc = json.loads(raw.decode("utf-8"))
                text = "".join(
                    b.get("text", "") for b in doc.get("content", [])
                    if isinstance(b, dict) and b.get("type") == "text")
                usage_doc = doc.get("usage") or {}
                usage = (usage_doc.get("input_tokens"), usage_doc.get("output_tokens"))
                return {"text": text, "pieces": [text],
                        "stop": doc.get("stop_reason"), "usage": usage}
            text = ""
            pieces = []
            stop = None
            tokens_in = None
            tokens_out = None
            for event, data in _sse_frames(raw):
                doc = json.loads(data)
                if event == "message_start":
                    message = doc.get("message") or {}
                    tokens_in = message.get("usage", {}).get("input_tokens")
                elif event == "content_block_delta":
                    delta = doc.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        fragment = delta.get("text")
                        if isinstance(fragment, str):
                            pieces.append(fragment)
                            text += fragment
                elif event == "message_delta":
                    delta = doc.get("delta") or {}
                    stop = delta.get("stop_reason") or stop
                    usage_doc = doc.get("usage") or {}
                    tokens_out = usage_doc.get("output_tokens", tokens_out)
            if tokens_in is None and tokens_out is None:
                usage = None
            else:
                usage = (tokens_in, tokens_out)
            return {"text": text, "pieces": pieces, "stop": stop, "usage": usage}
        # Gemini wire.
        if not stream:
            doc = json.loads(raw.decode("utf-8"))
            candidate = doc["candidates"][0]
            text = "".join(
                p.get("text", "") for p in candidate["content"]["parts"]
                if isinstance(p, dict))
            meta = doc.get("usageMetadata") or {}
            usage = (meta.get("promptTokenCount"), meta.get("candidatesTokenCount"))
            return {"text": text, "pieces": [text],
                    "stop": candidate.get("finishReason"), "usage": usage}
        text = ""
        pieces = []
        stop = None
        usage = None
        for _event, data in _sse_frames(raw):
            doc = json.loads(data)
            for candidate in doc.get("candidates", []):
                content = candidate.get("content") or {}
                for p in content.get("parts", []):
                    if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"]:
                        pieces.append(p["text"])
                        text += p["text"]
                reason = candidate.get("finishReason")
                if isinstance(reason, str):
                    stop = reason
            meta = doc.get("usageMetadata")
            if isinstance(meta, dict):
                usage = (meta.get("promptTokenCount"), meta.get("candidatesTokenCount"))
        return {"text": text, "pieces": pieces, "stop": stop, "usage": usage}
    except Exception:
        return None


FINISH_TO_OPENAI = {
    "stop": "stop", "end_turn": "stop", "STOP": "stop",
    "max_tokens": "length", "MAX_TOKENS": "length",
    "tool_use": "tool_calls", "refusal": "content_filter", "SAFETY": "content_filter",
}
FINISH_TO_ANTHROPIC = {
    "stop": "end_turn", "STOP": "end_turn", "end_turn": "end_turn",
    "length": "max_tokens", "MAX_TOKENS": "max_tokens",
    "tool_calls": "tool_use", "function_call": "tool_use",
    "content_filter": "refusal", "SAFETY": "refusal",
}


def _stream_chunks(answer):
    """Pieces to emit for a streamed reply, re-split only if configured."""
    pieces = [p for p in answer["pieces"] if p]
    if STREAM_PIECE_CHARS > 0:
        full = answer["text"]
        size = STREAM_PIECE_CHARS
        pieces = [full[i:i + size] for i in range(0, len(full), size)] or [""]
    if not pieces:
        pieces = [""]
    return pieces

# ---------------------------------------------------------------------------
# Answering the client back in the wire it spoke.
# ---------------------------------------------------------------------------
def render_reply(req, model_id, answer):
    """Build (content type, body bytes) for the client's wire and stream mode."""
    usage = answer["usage"] or (None, None)
    if req.api == "openai":
        stop = FINISH_TO_OPENAI.get(answer["stop"], "stop")
        if not req.stream:
            doc = {
                "id": "chatcmpl-agg",
                "object": "chat.completion",
                "created": FIXED_CREATED,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer["text"]},
                    "finish_reason": stop,
                    "logprobs": None,
                }],
                "usage": {
                    "prompt_tokens": usage[0],
                    "completion_tokens": usage[1],
                    "total_tokens": (usage[0] or 0) + (usage[1] or 0),
                },
            }
            return "application/json", json.dumps(doc).encode("utf-8")
        frames = []

        def make_chunk(delta, finish):
            return {
                "id": "chatcmpl-agg",
                "object": "chat.completion.chunk",
                "created": FIXED_CREATED,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        first = make_chunk({"role": "assistant", "content": ""}, None)
        frames.append("data: " + json.dumps(first) + "\n\n")
        for piece in _stream_chunks(answer):
            frames.append("data: " + json.dumps(make_chunk({"content": piece}, None)) + "\n\n")
        frames.append("data: " + json.dumps(make_chunk({}, stop)) + "\n\n")
        if usage[0] is not None:
            usage_chunk = {
                "id": "chatcmpl-agg",
                "object": "chat.completion.chunk",
                "created": FIXED_CREATED,
                "model": model_id,
                "choices": [],
                "usage": {
                    "prompt_tokens": usage[0],
                    "completion_tokens": usage[1],
                    "total_tokens": (usage[0] or 0) + (usage[1] or 0),
                },
            }
            frames.append("data: " + json.dumps(usage_chunk) + "\n\n")
        frames.append("data: [DONE]\n\n")
        return "text/event-stream", "".join(frames).encode("utf-8")
    # Anthropic wire.
    stop = FINISH_TO_ANTHROPIC.get(answer["stop"], "end_turn")
    if not req.stream:
        doc = {
            "id": "msg_agg",
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "content": [{"type": "text", "text": answer["text"]}],
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
        }
        return "application/json", json.dumps(doc).encode("utf-8")
    frames = []

    def make_frame(event, doc):
        return "event: " + event + "\ndata: " + json.dumps(doc) + "\n\n"

    frames.append(make_frame("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_agg", "type": "message", "role": "assistant",
            "model": model_id, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": usage[0], "output_tokens": 0},
        },
    }))
    frames.append(make_frame("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in _stream_chunks(answer):
        frames.append(make_frame("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(make_frame("content_block_stop", {
        "type": "content_block_stop", "index": 0,
    }))
    frames.append(make_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": usage[1]},
    }))
    frames.append(make_frame("message_stop", {"type": "message_stop"}))
    return "text/event-stream", "".join(frames).encode("utf-8")


# ---------------------------------------------------------------------------
# Routing: order the free models that satisfy the request by the brief's
# rating, try them one by one, and remember refusals as they arrive.
# ---------------------------------------------------------------------------
class Chooser:
    """Holds the rating table plus per-provider refusal state for a run."""

    def __init__(self, upstreams, score_rows):
        self.upstreams = {u.id: u for u in upstreams}
        self.order = [u.id for u in upstreams]
        self.table = {}
        for row in score_rows:
            self.table[(row["provider"], row["model"])] = row
        self.spent = {pid: 0 for pid in self.order}
        self.refusals = {pid: 0 for pid in self.order}
        self.exhausted = set()

    def rank_candidates(self, req):
        """Eligible (upstream, model) pairs, best score first."""
        needs_json = req.json_mode or (req.api == "anthropic" and ANTHROPIC_JSON_SAFE)
        is_long = len(req.last_user_text) >= LONG_CONTEXT_CHARS
        scored = []
        for position, pid in enumerate(self.order):
            upstream = self.upstreams[pid]
            for (prov_id, model_id), row in self.table.items():
                if prov_id != pid:
                    continue
                if row.get("free") is not True:
                    continue
                if req.wants_tools and not row.get("tools"):
                    continue
                if req.wants_vision and not row.get("vision"):
                    continue
                if needs_json and not row.get("json"):
                    continue
                if is_long and (row.get("context_length") or 0) < LONG_CONTEXT_MIN:
                    continue
                points = row.get("rating_milli", 0)
                if upstream.wire == req.api:
                    points += PREFER_SAME_WIRE_BONUS
                scored.append((-points, position, upstream, model_id))
        if TIE_BREAK == "alpha":
            scored.sort(key=lambda item: (item[0], item[2].id, item[3]))
        else:
            scored.sort(key=lambda item: (item[0], item[1]))
        return [(upstream, model_id) for _neg, _pos, upstream, model_id in scored]

    def dispatch(self, req):
        """Try candidates in rank order; return (model, answer) or None."""
        for upstream, model_id in self.rank_candidates(req):
            if upstream.id in self.exhausted:
                continue
            url, headers, body = make_upstream_call(upstream, model_id, req)
            status, _headers, raw = fetch_url("POST", url, body, headers, timeout=20)
            self.spent[upstream.id] += 1
            if 200 <= status < 300:
                answer = decode_reply(upstream.wire, raw, req.stream)
                if answer is not None:
                    return model_id, answer
                continue
            if status in (402, 429):
                self.refusals[upstream.id] += 1
                if self.refusals[upstream.id] >= REFUSAL_LIMIT:
                    self.exhausted.add(upstream.id)
                continue
    # Anything else (400, 5xx, no answer at all): try the next model.
    # Note: a 200 whose body cannot be decoded is handled above by
    # decode_reply returning None, which also falls through to the next
    # candidate instead of failing the whole request.
        return None


# ---------------------------------------------------------------------------
# The HTTP server the harness client talks to.
# ---------------------------------------------------------------------------
SHARED = {"router": None, "catalog": None}


def list_models(headers):
    """Model listing for GET /v1/models, in the shape the caller expects."""
    catalog = SHARED["catalog"] or {"providers": []}
    ids = []
    for provider in catalog["providers"]:
        for model in provider["models"]:
            if model["free"] and model["id"] not in ids:
                ids.append(model["id"])
    if headers.get("x-api-key") is not None:
        doc = {
            "data": [
                {"type": "model", "id": mid, "display_name": mid,
                 "created_at": "2026-01-01T00:00:00Z"}
                for mid in ids
            ],
            "has_more": False,
            "first_id": ids[0] if ids else None,
            "last_id": ids[-1] if ids else None,
        }
    else:
        doc = {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "created": FIXED_CREATED,
                 "owned_by": "aggregator"}
                for mid in ids
            ],
        }
    return json.dumps(doc).encode("utf-8")


def err_payload(api, message):
    """Error body in the caller's wire format."""
    if api == "anthropic":
        doc = {"type": "error",
               "error": {"type": "api_error", "message": message}}
        return json.dumps(doc).encode("utf-8")
    doc = {"error": {"message": message, "type": "api_error",
                     "param": None, "code": "no_provider"}}
    return json.dumps(doc).encode("utf-8")


class Gateway(BaseHTTPRequestHandler):
    """HTTP endpoint: model listing plus the two chat wires."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, body):
        try:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            self._send(200, "application/json", list_models(self.headers))
        else:
            self._send(404, "application/json", err_payload("openai", "not found"))

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        else:
            self._send(404, "application/json", err_payload("openai", "not found"))
            return
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body is not an object")
        except Exception:
            self._send(400, "application/json",
                       err_payload(api, "the request body is not one JSON object"))
            return
        req = NormalizedRequest(api, body)
        served = SHARED["router"].dispatch(req)
        if served is None:
            self._send(502, "application/json",
                       err_payload(api, "no free model could serve this request"))
            return
        model_id, answer = served
        content_type, payload = render_reply(req, model_id, answer)
        self._send(200, content_type, payload)


# ---------------------------------------------------------------------------
# Trial lifecycle: build the catalog, register with the harness, serve
# until the harness reports the workload done, then exit 0.
# ---------------------------------------------------------------------------
def main(argv):
    upstreams, score_rows, _manifest = read_brief()
    if "--list" in argv:
        if not upstreams:
            print('{"providers":[]}')
            return 0
        print(catalog_json(assemble_catalog(upstreams, score_rows)))
        return 0
    catalog = assemble_catalog(upstreams, score_rows)
    SHARED["catalog"] = catalog
    SHARED["router"] = Chooser(upstreams, score_rows)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Gateway)
    server.daemon_threads = True
    import threading
    import time
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    worker.daemon = True
    worker.start()
    if ENV_URL:
        for _attempt in range(30):
            status, _headers, doc = fetch_json(
                "POST", ENV_URL + "/register",
                {"protocol_version": 1, "port": PORT, "catalog": catalog},
                timeout=5)
            if status == 200 and isinstance(doc, dict) and doc.get("ok") is True:
                break
            time.sleep(0.5)
        misses = 0
        waited = 0.0
        while waited < 170.0:
            status, _headers, doc = fetch_json("GET", ENV_URL + "/work", timeout=5)
            if status == 200 and isinstance(doc, dict):
                misses = 0
                if doc.get("done") is True:
                    break
            else:
                misses += 1
                if misses > 40:
                    break
            time.sleep(0.2)
            waited += 0.2
    else:
        worker.join()
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
