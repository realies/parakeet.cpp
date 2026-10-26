#include "parakeet_capi.h"
#include "audio_io.hpp"
#include "ggml_graph.hpp"
#include "model.hpp"
#include "transcription_json.hpp"

#include <algorithm>
#include <codecvt>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <locale>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Env: PARAKEET_TEST_ALLOWED_TOKENS_GGUF + PARAKEET_TEST_ALLOWED_TOKENS_WAV.
// Optional PARAKEET_TEST_ALLOWED_TOKENS_TEXT checks the masked English text.
// The clip stays outside the repository. No checkpoint/audio -> skip (77).
static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static std::u32string codepoints(const std::string& s) {
    return std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>().from_bytes(s);
}

static bool cyrillic(char32_t c) {
    return (c >= 0x0400 && c <= 0x052f) || (c >= 0x1c80 && c <= 0x1c8f) ||
           (c >= 0x2de0 && c <= 0x2dff) || (c >= 0xa640 && c <= 0xa69f);
}

static bool latin_piece(const std::string& piece) {
    // ponytail: covers this model's European vocabulary; use Unicode Script
    // tables if testing arbitrary vocabularies with further Latin extensions.
    for (char32_t c : codepoints(piece)) {
        const bool latin = (c >= 0x00c0 && c <= 0x024f && c != 0x00d7 && c != 0x00f7) ||
                           (c >= 0x1e00 && c <= 0x1eff);
        const bool common = (c >= 0x20 && c <= 0x7e) || (c >= 0x00a0 && c <= 0x00bf) ||
                            (c >= 0x2000 && c <= 0x206f) || c == 0x2581;
        if (!latin && !common) return false;
    }
    return true;
}

static std::string take(char* result, parakeet_ctx* ctx) {
    check(result != nullptr, parakeet_capi_last_error(ctx));
    const std::string s(result);
    parakeet_capi_free_string(result);
    return s;
}

static void token_scripts(const char* label, const std::string& doc,
                          const parakeet_ctx* ctx, const std::vector<uint8_t>* allowed) {
    std::fprintf(stderr, "%s TOKEN_SCRIPTS:", label);
    size_t count = 0;
    for (size_t p = 0; (p = doc.find("\"id\":", p)) != std::string::npos; p += 5) {
        const int id = std::stoi(doc.substr(p + 5));
        const char* piece = parakeet_capi_token_piece(ctx, id);
        check(piece != nullptr, "invalid token id in JSON");
        const auto chars = codepoints(piece);
        const bool cyr = std::any_of(chars.begin(), chars.end(), cyrillic);
        std::fprintf(stderr, " %d:%s[%s]", id, piece,
                     cyr ? "Cyrillic" : latin_piece(piece) ? "Latin/common" : "other");
        if (allowed) check(allowed->at((size_t)id) && !cyr, "disallowed token emitted");
        ++count;
    }
    std::fprintf(stderr, "\n");
    check(count > 0, "expected speech tokens");
    if (allowed) {
        const auto chars = codepoints(doc);
        check(!std::any_of(chars.begin(), chars.end(), cyrillic), "Cyrillic text with mask");
    }
}

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_ALLOWED_TOKENS_GGUF");
    const char* wav = std::getenv("PARAKEET_TEST_ALLOWED_TOKENS_WAV");
    if (!gguf || !wav) {
        std::fprintf(stderr, "allowed-token model/clip env not set; skip\n");
        return 77;
    }
    try {
        pk::set_num_threads(8);
        check(parakeet_capi_set_allowed_tokens(nullptr, nullptr, 0) != 0, "NULL setter");
        check(parakeet_capi_vocab_size(nullptr) == 0, "NULL vocab");
        check(parakeet_capi_token_piece(nullptr, 0) == nullptr, "NULL piece");
        std::unique_ptr<parakeet_ctx, decltype(&parakeet_capi_free)> owner(
            parakeet_capi_load(gguf), parakeet_capi_free);
        auto* ctx = owner.get();
        check(ctx != nullptr, parakeet_capi_load_error());
        pk::Audio audio;
        check(pk::load_audio_16k_mono(wav, audio) && !audio.samples.empty(), "clip load");
        const int n = (int)audio.samples.size();
        const int vocab = parakeet_capi_vocab_size(ctx);
        check(vocab > 0, "empty vocabulary");
        check(parakeet_capi_token_piece(ctx, -1) == nullptr, "negative piece id");
        check(parakeet_capi_token_piece(ctx, vocab + 1) == nullptr, "large piece id");
        check(std::strcmp(parakeet_capi_token_piece(ctx, vocab), "") == 0, "blank piece");
        const char* borrowed = parakeet_capi_token_piece(ctx, 0);
        check(borrowed != nullptr, "piece 0");
        const std::string saved_piece(borrowed);
        check(latin_piece("▁English-é") && !latin_piece("▁Я") && !latin_piece("α"), "script filter");

        auto batch = [&]() {
            return take(parakeet_capi_transcribe_pcm_batch_json(ctx, audio.samples.data(),
                        &n, 1, 16000, 2), ctx);
        };
        const std::string before = batch();
        std::fprintf(stderr, "UNMASKED %s\n", before.c_str());
        token_scripts("UNMASKED", before, ctx, nullptr);

        std::vector<int32_t> ids;
        std::vector<uint8_t> allowed((size_t)vocab + 1, 0);
        for (int id = 0; id < vocab; ++id) {
            const char* piece = parakeet_capi_token_piece(ctx, id);
            check(piece != nullptr, "missing vocabulary piece");
            if (latin_piece(piece)) { ids.push_back(id); allowed[(size_t)id] = 1; }
        }
        allowed[(size_t)vocab] = 1;
        check(!ids.empty() && ids.size() < (size_t)vocab, "mask must restrict vocabulary");
        // Omit blank deliberately: the decoder must always retain it.
        check(parakeet_capi_set_allowed_tokens(ctx, ids.data(), (int)ids.size()) == 0, "set mask");
        const std::string masked = batch();
        std::fprintf(stderr, "MASKED (%zu/%d pieces) %s\n", ids.size(), vocab, masked.c_str());
        token_scripts("MASKED", masked, ctx, &allowed);
        if (const char* text = std::getenv("PARAKEET_TEST_ALLOWED_TOKENS_TEXT")) {
            std::string field = "\"text\":";
            pk::append_json_string(field, text);
            check(masked.find(field) != std::string::npos, "English transcript mismatch");
        }

        // Invalid updates retain the configured set, including NULL + n < 0.
        for (int32_t bad : {-1, vocab + 1}) {
            check(parakeet_capi_set_allowed_tokens(ctx, &bad, 1) != 0, "invalid id accepted");
            check(*parakeet_capi_last_error(ctx), "missing invalid-id error");
        }
        check(parakeet_capi_set_allowed_tokens(ctx, nullptr, -1) != 0, "negative count accepted");
        check(*parakeet_capi_last_error(ctx), "missing count error");
        check(batch() == masked, "failed update changed mask");

        check(take(parakeet_capi_transcribe_pcm_batch_json_lang(ctx, audio.samples.data(),
                   &n, 1, 16000, 0, nullptr), ctx) == masked, "default/lang path ignored mask");
        const std::string single = take(parakeet_capi_transcribe_path_json(ctx, wav, 2), ctx);
        token_scripts("SINGLE", single, ctx, &allowed);
        const std::string plain = take(parakeet_capi_transcribe_pcm(ctx, audio.samples.data(), n, 16000, 2), ctx);
        const auto chars = codepoints(plain);
        check(!plain.empty() && !std::any_of(chars.begin(), chars.end(), cyrillic), "plain path ignored mask");
        // The other batched entry point and N > 1 use the same configured set.
        const float* samples[] = {audio.samples.data(), audio.samples.data()};
        const int lengths[] = {n, n};
        char* texts[2] = {};
        check(parakeet_capi_transcribe_pcm_batch(ctx, samples, lengths, 2, 16000, 2, texts) == 0,
              parakeet_capi_last_error(ctx));
        for (char* text : texts) {
            const std::string s = take(text, ctx);
            const auto points = codepoints(s);
            check(!s.empty() && !std::any_of(points.begin(), points.end(), cyrillic), "batch text ignored mask");
        }

        char* refused = parakeet_capi_transcribe_pcm_batch_json(ctx, audio.samples.data(), &n, 1, 16000, 1);
        check(refused == nullptr && std::strstr(parakeet_capi_last_error(ctx), "allowed tokens"), "masked CTC must fail");
        float* logits = nullptr;
        int frames = 0, classes = 0;
        check(parakeet_capi_transcribe_pcm_logits(ctx, audio.samples.data(), n, 16000,
                  &logits, &frames, &classes) != 0 && logits == nullptr &&
                  std::strstr(parakeet_capi_last_error(ctx), "allowed tokens"), "masked CTC logits must fail");
        refused = parakeet_capi_transcribe_pcm_nbest_json(ctx, audio.samples.data(), n, 16000, 1, 1, 1, nullptr);
        check(refused == nullptr && *parakeet_capi_last_error(ctx), "masked beam must fail");

        // A non-empty set containing only blank yields no text; duplicates work.
        const int32_t blanks[] = {vocab, vocab};
        check(parakeet_capi_set_allowed_tokens(ctx, blanks, 2) == 0, "blank-only mask");
        check(batch().find("\"text\":\"\"") != std::string::npos, "blank-only emitted text");
        check(parakeet_capi_set_allowed_tokens(ctx, ids.data(), 0) == 0, "clear with n=0");
        check(batch() == before, "n=0 did not restore unmasked decode");
        check(parakeet_capi_set_allowed_tokens(ctx, ids.data(), (int)ids.size()) == 0, "reset mask");
        check(parakeet_capi_set_allowed_tokens(ctx, nullptr, 10) == 0, "clear with NULL");
        check(batch() == before, "NULL did not restore unmasked decode");
        check(saved_piece == borrowed, "borrowed piece invalidated");

        // Exercise grouped VAD decoding with a synthetic all-speech detector.
        // Release the C context first so the full model is only resident once.
        owner.reset();
        auto model = pk::Model::load(gguf);
        check(model != nullptr, "VAD model load");
        model->set_allowed_tokens(ids.data(), (int)ids.size());
        pk::SegmenterOpts opts;
        opts.max_seg_sec = 2.0;
        opts.frame_sec = 0.08;
        pk::Model::VadProbabilityFn speech = [](const std::vector<float>& pcm) {
            return std::vector<float>((pcm.size() + 1279) / 1280, 1.0f);
        };
        const auto vad = model->transcribe_pcm_vad_with_timestamps(
            audio.samples, 16000, pk::Decoder::kTDT, "", opts, &speech);
        check(!vad.tokens.empty(), "no VAD speech tokens");
        for (const auto& token : vad.tokens)
            check(allowed.at((size_t)token.id), "grouped VAD ignored mask");
        check(model->transcribe_pcm_vad(audio.samples, 16000, pk::Decoder::kTDT, "", opts, &speech) == vad.text,
              "grouped plain/timestamped VAD mismatch");
        std::fprintf(stderr, "allowed tokens C API: PASS\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "allowed tokens C API: %s\n", e.what());
        return 1;
    }
}
