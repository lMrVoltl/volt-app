#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static mtmd_context* g_mtmd = nullptr;
static std::atomic<bool> g_stop{false};
static std::string g_lastMtmdErr;
static std::vector<llama_token> g_cached_tokens;

static void volt_mtmd_log_cb(enum ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR && text != nullptr) {
        g_lastMtmdErr = text;
        while (!g_lastMtmdErr.empty() && (g_lastMtmdErr.back() == '\n' || g_lastMtmdErr.back() == '\r')) {
            g_lastMtmdErr.pop_back();
        }
    }
}

static size_t valid_utf8_prefix_len(const std::string& s) {
    size_t i = 0;
    size_t n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t charLen = 0;
        if (c < 0x80) charLen = 1;
        else if ((c & 0xE0) == 0xC0) charLen = 2;
        else if ((c & 0xF0) == 0xE0) charLen = 3;
        else if ((c & 0xF8) == 0xF0) charLen = 4;
        else {
            i++;
            continue;
        }
        if (i + charLen > n) {
            break;
        }
        i += charLen;
    }
    return i;
}

static std::vector<llama_token> collect_banned_vision_tokens(const llama_vocab* vocab) {
    std::vector<llama_token> banned_ids;
    llama_token eos_id = llama_vocab_eos(vocab);
    llama_token eot_id = llama_vocab_eot(vocab);

    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (int32_t id = 0; id < n_vocab; id++) {
        if (id == eos_id || id == eot_id || llama_vocab_is_eog(vocab, id)) continue;
        const char* txt = llama_vocab_get_text(vocab, id);
        if (!txt || txt[0] == '\0') continue;

        if (std::strncmp(txt, "<|", 2) == 0 ||
            std::strcmp(txt, "<tool_call>") == 0 ||
            std::strcmp(txt, "</tool_call>") == 0 ||
            std::strcmp(txt, "<tool_response>") == 0 ||
            std::strcmp(txt, "</tool_response>") == 0) {
            banned_ids.push_back(id);
        }
    }
    return banned_ids;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_voltage_ai_MainActivity_nativeLoadModel(
    JNIEnv* env, jobject, jstring jpath, jstring jmmproj, jint threads, jint nCtx
) {
    if (g_mtmd) { mtmd_free(g_mtmd); g_mtmd = nullptr; }
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_cached_tokens.clear();
    g_lastMtmdErr.clear();

    const char* path = env->GetStringUTFChars(jpath, nullptr);
    llama_backend_init();
    mtmd_log_set(volt_mtmd_log_cb, nullptr);

    auto mparams = llama_model_default_params();
    mparams.use_mmap = true;
    g_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(jpath, path);
    if (!g_model) return JNI_FALSE;

    // Используем ровно столько потоков, сколько выбрал пользователь (рекомендуется 4 для big-кластера)
    int t = std::max(1, std::min((int)threads, 8));

    auto cparams = llama_context_default_params();
    cparams.n_ctx = nCtx > 0 ? nCtx : 4096;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.n_threads = t;
    cparams.n_threads_batch = t;
    // Оставляем стандартный быстрый F16 KV-кэш (Q8_0 без FlashAttn вызывает медленную деквантовку на CPU)
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) return JNI_FALSE;

    if (jmmproj != nullptr) {
        const char* mmpath = env->GetStringUTFChars(jmmproj, nullptr);
        if (mmpath && mmpath[0] != '\0') {
            auto mctx_params = mtmd_context_params_default();
            mctx_params.use_gpu = false;
            mctx_params.n_threads = t;
            g_mtmd = mtmd_init_from_file(mmpath, g_model, mctx_params);
        }
        env->ReleaseStringUTFChars(jmmproj, mmpath);
    }
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_voltage_ai_MainActivity_nativeStop(JNIEnv*, jobject) {
    g_stop = true;
}

extern "C" JNIEXPORT void JNICALL
Java_com_voltage_ai_MainActivity_nativeGenerate(
    JNIEnv* env, jobject thiz, jstring jprompt, jstring jsysPrompt,
    jbyteArray jrgbBytes, jint imgW, jint imgH,
    jfloat temp, jfloat topP, jint maxTokens
) {
    if (!g_model || !g_ctx) return;
    g_stop = false;

    jclass cls = env->GetObjectClass(thiz);
    jmethodID onToken = env->GetMethodID(cls, "onNativeToken", "(Ljava/lang/String;)V");
    jmethodID onStats = env->GetMethodID(cls, "onNativeStats", "(IDDILjava/lang/String;)V");
    if (env->ExceptionCheck()) { env->ExceptionClear(); return; }

    const char* raw = env->GetStringUTFChars(jprompt, nullptr);
    const char* sysRaw = env->GetStringUTFChars(jsysPrompt, nullptr);

    std::string sysStr = (sysRaw && sysRaw[0] != '\0')
        ? std::string(sysRaw)
        : "Ты локальный ассистент Volt AI.";
    std::string userHistory(raw ? raw : "");

    if (raw) env->ReleaseStringUTFChars(jprompt, raw);
    if (sysRaw) env->ReleaseStringUTFChars(jsysPrompt, sysRaw);

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    auto t_start = std::chrono::steady_clock::now();

    int visionState = 0;
    llama_pos n_past = 0;

    if (jrgbBytes != nullptr && imgW > 0 && imgH > 0) {
        llama_memory_clear(llama_get_memory(g_ctx), true);
        g_cached_tokens.clear();

        if (g_mtmd == nullptr) {
            visionState = -1;
            if (g_lastMtmdErr.empty()) g_lastMtmdErr = "mmproj не инициализирован";
        } else {
            jbyte* rgbPtr = env->GetByteArrayElements(jrgbBytes, nullptr);
            if (rgbPtr == nullptr) {
                visionState = -2;
            } else {
                mtmd_bitmap* bmp = mtmd_bitmap_init((uint32_t)imgW, (uint32_t)imgH, (const unsigned char*)rgbPtr);
                env->ReleaseByteArrayElements(jrgbBytes, rgbPtr, JNI_ABORT);

                if (bmp == nullptr) {
                    visionState = -3;
                } else {
                    g_lastMtmdErr.clear();

                    std::string visionSys = sysStr + "\nОтвечай строго на русском языке, понятным связным текстом, без вызова функций, координат рамок и технических тегов.";
                    std::string prefix = "<|im_start|>system\n" + visionSys + "<|im_end|>\n";
                    std::string suffix;

                    size_t pos = userHistory.rfind("<|im_start|>user\n");
                    if (pos != std::string::npos) {
                        prefix += userHistory.substr(0, pos + 17);
                        std::string tail = userHistory.substr(pos + 17);
                        size_t endPos = tail.rfind("<|im_end|>");
                        if (endPos != std::string::npos) {
                            tail.insert(endPos, "\n(Ответь на русском языке)");
                        } else {
                            tail += "\n(Ответь на русском языке)";
                        }
                        suffix = "\n" + tail + "<|im_start|>assistant\n";
                    } else {
                        prefix += "<|im_start|>user\n";
                        suffix = "\n" + userHistory + "\n(Ответь на русском языке)<|im_end|>\n<|im_start|>assistant\n";
                    }

                    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
                    mtmd_input_text t0 = { prefix.c_str(), false, true };
                    mtmd_input_text t2 = { suffix.c_str(), false, true };
                    mtmd_input_part p0 = { &t0, nullptr };
                    mtmd_input_part p1 = { nullptr, bmp };
                    mtmd_input_part p2 = { &t2, nullptr };
                    const mtmd_input_part* partsArr[3] = { &p0, &p1, &p2 };

                    int tokRes = mtmd_tokenize_from_parts(g_mtmd, chunks, partsArr, 3, true);
                    if (tokRes != 0) {
                        visionState = -10 - tokRes;
                    } else {
                        llama_pos new_n_past = 0;
                        int32_t n_batch = (int32_t)llama_n_batch(g_ctx);
                        int evalRes = mtmd_helper_eval_chunks(g_mtmd, g_ctx, chunks, 0, 0, n_batch, true, &new_n_past);
                        if (evalRes != 0) {
                            visionState = -20 - evalRes;
                        } else {
                            visionState = 1;
                            n_past = new_n_past;
                        }
                    }
                    mtmd_input_chunks_free(chunks);
                    mtmd_bitmap_free(bmp);
                }
            }
        }
    }

    if (visionState != 1) {
        std::string prompt;
        if (userHistory.find("<|im_start|>") != std::string::npos) {
            prompt = "<|im_start|>system\n" + sysStr + "<|im_end|>\n" + userHistory + "<|im_start|>assistant\n";
        } else {
            prompt = "<|im_start|>system\n" + sysStr + "<|im_end|>\n<|im_start|>user\n" + userHistory + "<|im_end|>\n<|im_start|>assistant\n";
        }

        std::vector<llama_token> tokens(prompt.size() + 32);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
        if (n_tokens < 0) return;
        tokens.resize(n_tokens);

        size_t common_prefix = 0;
        size_t max_common = std::min(g_cached_tokens.size(), tokens.size() > 0 ? tokens.size() - 1 : 0);
        while (common_prefix < max_common && g_cached_tokens[common_prefix] == tokens[common_prefix]) {
            common_prefix++;
        }

        llama_memory_t mem = llama_get_memory(g_ctx);
        if (common_prefix == 0) {
            llama_memory_clear(mem, true);
        } else {
            llama_memory_seq_rm(mem, 0, (llama_pos)common_prefix, -1);
        }

        size_t eval_count = tokens.size() - common_prefix;
        int32_t max_batch = (int32_t)llama_n_batch(g_ctx);

        for (size_t offset = 0; offset < eval_count; offset += max_batch) {
            size_t chunk_len = std::min((size_t)max_batch, eval_count - offset);
            llama_batch batch = llama_batch_init((int32_t)chunk_len, 0, 1);
            batch.n_tokens = (int32_t)chunk_len;
            for (size_t j = 0; j < chunk_len; j++) {
                size_t idx = common_prefix + offset + j;
                batch.token[j] = tokens[idx];
                batch.pos[j] = (llama_pos)idx;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = (idx == tokens.size() - 1);
            }
            int decRes = llama_decode(g_ctx, batch);
            llama_batch_free(batch);
            if (decRes != 0) {
                g_cached_tokens.clear();
                return;
            }
        }

        g_cached_tokens = tokens;
        n_past = (llama_pos)tokens.size();
    }

    auto t_gen_start = std::chrono::steady_clock::now();
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    auto banned_ids = collect_banned_vision_tokens(vocab);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> recent_tokens;
    recent_tokens.reserve(64);
    const float repeat_penalty = 1.15f;

    int generated = 0;
    int limit = maxTokens > 0 ? maxTokens : 1024;
    bool hasPrintedVisible = false;
    std::string utf8Pending;

    llama_batch stepBatch = llama_batch_init(1, 0, 1);

    for (int i = 0; i < limit && !g_stop; i++) {
        float* logits = llama_get_logits_ith(g_ctx, -1);
        if (logits != nullptr) {
            for (llama_token bid : banned_ids) {
                if (bid >= 0 && bid < n_vocab) {
                    logits[bid] = -INFINITY;
                }
            }
            if (generated == 0) {
                llama_token eos_id = llama_vocab_eos(vocab);
                llama_token eot_id = llama_vocab_eot(vocab);
                if (eos_id >= 0 && eos_id < n_vocab) logits[eos_id] = -INFINITY;
                if (eot_id >= 0 && eot_id < n_vocab) logits[eot_id] = -INFINITY;
            }
            for (llama_token rtok : recent_tokens) {
                if (rtok >= 0 && rtok < n_vocab) {
                    if (logits[rtok] > 0.0f) logits[rtok] /= repeat_penalty;
                    else logits[rtok] *= repeat_penalty;
                }
            }
        }

        llama_token id = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) {
            if (generated == 0) continue;
            break;
        }
        llama_sampler_accept(smpl, id);

        if (visionState != 1) {
            g_cached_tokens.push_back(id);
        }

        if (recent_tokens.size() >= 64) {
            recent_tokens.erase(recent_tokens.begin());
        }
        recent_tokens.push_back(id);

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        if (n > 0) {
            utf8Pending.append(buf, n);
            size_t validLen = valid_utf8_prefix_len(utf8Pending);
            if (validLen > 0) {
                std::string piece = utf8Pending.substr(0, validLen);
                utf8Pending.erase(0, validLen);

                if (piece.find("<|im_end|>") != std::string::npos || piece.find("<|endoftext|>") != std::string::npos) {
                    if (generated == 0) continue;
                    break;
                }

                if (!hasPrintedVisible) {
                    size_t firstChar = piece.find_first_not_of(" \n\r\t");
                    if (firstChar == std::string::npos) {
                        piece.clear();
                    } else {
                        piece = piece.substr(firstChar);
                        hasPrintedVisible = true;
                    }
                }

                if (!piece.empty()) {
                    generated++;
                    jstring jpiece = env->NewStringUTF(piece.c_str());
                    env->CallVoidMethod(thiz, onToken, jpiece);
                    env->DeleteLocalRef(jpiece);
                }
            }
        }

        stepBatch.n_tokens = 1;
        stepBatch.token[0] = id;
        stepBatch.pos[0] = n_past++;
        stepBatch.n_seq_id[0] = 1;
        stepBatch.seq_id[0][0] = 0;
        stepBatch.logits[0] = true;

        if (llama_decode(g_ctx, stepBatch) != 0) break;
    }

    llama_batch_free(stepBatch);

    auto t_end = std::chrono::steady_clock::now();
    double gen_sec = std::chrono::duration<double>(t_end - t_gen_start).count();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    double tps = gen_sec > 0.01 ? (generated / gen_sec) : 0.0;

    jstring jerr = env->NewStringUTF(g_lastMtmdErr.c_str());
    env->CallVoidMethod(thiz, onStats, generated, tps, total_sec, visionState, jerr);
    env->DeleteLocalRef(jerr);
    llama_sampler_free(smpl);
}
