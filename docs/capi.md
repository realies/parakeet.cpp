# C API (`libparakeet.so`)

The README has the overview and the ABI note. This page has the longer examples.
The current ABI version is 10 (`parakeet_capi_abi_version()`); all additions since v5 are additive.
The full symbol list and the exact signatures are in `include/parakeet_capi.h`.

`include/parakeet_capi.h` defines a flat, exception-free C-API meant for `dlopen` / FFI / LocalAI integration. Build the shared library with `-DPARAKEET_SHARED=ON`:

```c
#include "parakeet_capi.h"

parakeet_ctx *ctx = parakeet_capi_load("model.gguf");  // load ONCE
if (!ctx) { fprintf(stderr, "%s\n", parakeet_capi_last_error(ctx)); return 1; }

char *text = parakeet_capi_transcribe_path(ctx, "audio.wav", 0 /*default*/);
if (text) { printf("%s\n", text); parakeet_capi_free_string(text); }

parakeet_capi_free(ctx);
```

In-memory PCM:
```c
char *text = parakeet_capi_transcribe_pcm(ctx, samples, n_samples,
                                          sample_rate, 0 /*default*/);
```

An offline TDT context can restrict greedy decoding with
`parakeet_capi_set_allowed_tokens(ctx, ids, n)`. Build the id set from the UTF-8
pieces returned by `parakeet_capi_token_piece(ctx, id)`; iterate ids below
`parakeet_capi_vocab_size(ctx)`. That size is also the blank id; blank is always
allowed, and its piece is `""`. Piece pointers are borrowed for the context's
life. The set is copied, persists across calls, and applies to text, timestamped
JSON, batch (including a batch of one), language and VAD paths. Duration logits
are unchanged; confidence uses the masked token distribution.

Pass NULL ids or n=0 to clear the set. Success returns 0; negative n or an id
outside `[0, vocab_size]` returns nonzero with `last_error`, preserving the old
set. CTC/RNNT-only, streaming and non-ASR contexts refuse this setter. On a TDT
context with an active set, CTC, raw CTC logits and TDT beam search return an
error; clear the set before using them. Configure while no request is running,
as with `set_concurrency`; concurrent transcriptions may then share it. These
three symbols are additive and keep ABI v10; FFI callers check symbol presence.

Timestamps and confidence as JSON (matches NeMo `timestamps=True` + `max_prob`):
```c
char *json = parakeet_capi_transcribe_path_json(ctx, "audio.wav", 0 /*default*/);
// {"text":"...",
//  "frame_sec":0.080000,
//  "words":[{"w":"Well,","start":0.480,"end":0.640,"conf":0.7859}, ...],
//  "tokens":[{"id":639,"t":0.480,"conf":0.9969}, ...]}
if (json) { printf("%s\n", json); parakeet_capi_free_string(json); }
```
`start`/`end`/`t` are in seconds; `conf` is the rescaled softmax probability of the emitted token in `(0,1]` (a word's `conf` is the `min` over its tokens). `frame_sec` is the encoder frame stride in seconds (`hop x subsampling / sample_rate`); multiply a frame-unit segment gap threshold (NeMo's `segment_gap_threshold`) by it to get the seconds gap between words when forming segments.

## Streaming (cache-aware EOU model)

For `parakeet_realtime_eou_120m-v1`, a streaming session decodes 16 kHz mono f32 PCM as it arrives, returning newly-finalized text and signalling EOU/EOB events:

```c
parakeet_stream *s = parakeet_capi_stream_begin(ctx);
int eou = 0;
char *t = parakeet_capi_stream_feed(s, pcm, n_samples, &eou); // "" if none yet
if (t) { printf("%s", t); parakeet_capi_free_string(t); }
if (eou) printf(" [EOU]");
// ...feed more chunks...
char *tail = parakeet_capi_stream_finalize(s);                // flush the tail
if (tail) { printf("%s\n", tail); parakeet_capi_free_string(tail); }
parakeet_capi_stream_free(s);
```

`<EOU>` (end-of-utterance) and `<EOB>` (backchannel) are stripped from the text and surfaced via `*eou_out` (the CLI `--stream` prints them as `[EOU @ <t>s]` markers). The streaming transcript matches NeMo's cache-aware streaming exactly, and `finalize` flushes the end-of-stream tail without fabricating an `<EOU>` that NeMo would not emit.

The LocalAI backend (in the LocalAI repo) dlopens `libparakeet.so` and uses these symbols directly: the offline `parakeet_capi_transcribe_*` / `parakeet_capi_transcribe_path_json` and the streaming `parakeet_capi_stream_*`. See `include/parakeet_capi.h` for the full API. The C++ streaming session (`pk::StreamingSession`) also exposes per-word timestamps and confidence as words finalize, via `drain_words()` alongside the EOU events, which the CLI `--stream --timestamps` path prints.
