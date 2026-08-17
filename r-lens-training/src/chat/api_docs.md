# r-lens Model API

This server hosts the language model **{MODEL_NAME}** ({NUM_LAYERS} decoder
layers) together with trained LoRA adapters and per-layer **r-lens** matrices.
The r-lens maps an intermediate residual-stream activation through a fitted
linear map plus the model's own unembedding, yielding a distribution over
vocabulary tokens: "what the model's computation at that token and layer is
promoting". This deployment is part of an introspection-training hackathon.

This page is both the human documentation and the machine documentation: if
you are an LLM agent, everything you need is on this page. A raw-markdown
copy is at `GET /v1/docs.md`.

## Authentication

Every `/v1/*` endpoint except `/v1/health` requires an API key:

    Authorization: Bearer YOUR_KEY

Keys are distributed by the server operator. Requests without a valid key get
HTTP 401. There is no other access: the API exposes only the endpoints below —
no file access, no shell, no configuration.

## Adapters

Available adapters (LoRA fine-tunes of the base model): {ADAPTERS}

`"base"` is the unmodified model. Every request may set `"adapter"` to any of
these names; if omitted, the server's current default adapter is used
(changeable via `/v1/adapter/switch`).

## Endpoints

### GET /v1/health  (no auth)
Liveness probe. Returns `{"status": "ok", "uptime_s": ..., "requests_served": ...}`.

### GET /v1/models
Model name, layer count, adapter list, current default adapter, and whether
r-lens matrices are loaded.

### POST /v1/chat
Chat with the model.

Request body (JSON):

    {
      "messages": [{"role": "user", "content": "Hello!"}],
      "adapter": "base",            // optional, see /v1/models
      "thinking": false,            // optional: enable the model's thinking mode
      "temperature": 0.7,           // optional, 0.0-2.0 (0 = greedy)
      "max_new_tokens": 512         // optional, 1-1024
    }

Roles must be "user", "assistant", or "system". At most 50 messages and
30000 total characters per request.

Response:

    {
      "completion": "Hi! How can I help?",
      "completion_tokens": 8,
      "prompt_tokens": 12,
      "adapter": "base"
    }

Example:

    curl -s http://HOST:PORT/v1/chat \
      -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "What is 16 x 2?"}]}'

### POST /v1/lens
Read the model's r-lens at a token position: run the given text through the
model and decode the residual-stream activation at `position`, averaged over
layers `layer_lo..layer_hi` (1-indexed, inclusive, clamped to 1..{NUM_LAYERS}).

Request body:

    {
      "text": "The quick brown fox",   // max 4000 chars
      "position": 2,                   // 0-indexed into the RETURNED tokens array
      "layer_lo": 15,
      "layer_hi": 20,
      "top_k": 10,                     // 1-50
      "adapter": "base"                // optional
    }

Response includes `tokens` (the tokenization the server used, including a
leading BOS token if the model requires one — index `position` refers to THIS
array) and `top`, the top-k readout tokens with probabilities:

    {
      "tokens": ["<bos>", "The", " quick", " brown", " fox"],
      "position": 2,
      "layers": [15, 20],
      "adapter": "base",
      "top": [{"token": " fast", "prob": 0.1234}, ...]
    }

Tip for introspection experiments: first call with your text, read `tokens`
to find the index of the token you care about, then call again with that
index.

### POST /v1/adapter/switch
Set the server's default adapter for subsequent requests that omit "adapter".

    {"adapter": "base"}

### POST /v1/reboot
Reload every adapter's weights from disk (useful when the operator's training
run has written new checkpoints). Takes a few seconds; in-flight requests
finish first. Returns the reloaded adapter list.

## Behavior notes

- Concurrent requests are batched server-side: parallel calls with the same
  adapter and sampling parameters share one GPU batch. Submitting several
  requests at once is encouraged.
- Generation stops at the model's end-of-turn token or `max_new_tokens`,
  whichever comes first. There is a 600 s server-side timeout per request.
- Errors are JSON: `{"detail": "explanation"}` with an appropriate HTTP
  status (400 invalid input, 401 auth, 500 generation failure, 504 timeout).
