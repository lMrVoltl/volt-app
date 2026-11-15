#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static mtmd_context* g_mtmd = nullptr;
static std::atomic<bool> g_stop{false};

extern "C" JNIEXPORT jboolean JNICALL
Java_com_voltage_ai_MainActivity_nativeLoadModel(
    JNIEnv* env, jobject, jstring jpath, jstring jmmproj, jint threads, jint nCtx
) {
    if (g_mtmd) { mtmd_free(g_mtmd); g_mtmd = nullptr; }
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }

    const char* path = env->GetStringUTFChars(jpath, nullptr);
    llama_backend_init();

    auto mparams = llama_model_default_params();
    g_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(jpath, path);
    if (!g_model) return JNI_FALSE;

    auto cparams = llama_context_default_params();
    cparams.n_ctx = nCtx > 0 ? nCtx : 4096;
    cparams.n_batch = 512;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) return JNI_FALSE;

    if (jmmproj != nullptr) {
        const char* mmpath = env->GetStringUTFChars(jmmproj, nullptr);
        if (mmpath && mmpath[0] != '\0') {
            auto mctx_params = mtmd_context_params_default();
            mctx_params.use_gpu = false;
            mctx_params.n_threads = threads;
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
    jmethodID onStats = env->GetMethodID(cls, "onNativeStats", "(IDD)V");

    const char* raw = env->GetStringUTFChars(jprompt, nullptr);
    const char* sysRaw = env->GetStringUTFChars(jsysPrompt, nullptr);

    std::string sysStr = (sysRaw && sysRaw[0] != '\0')
        ? std::string(sysRaw)
        : "Ты локальный ассистент Volt AI.";
    std::string userHistory(raw);

    env->ReleaseStringUTFChars(jprompt, raw);
    env->ReleaseStringUTFChars(jsysPrompt, sysRaw);

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    auto t_start = std::chrono::steady_clock::now();
    llama_memory_clear(llama_get_memory(g_ctx), true);

    bool usedVision = false;
    if (g_mtmd != nullptr && jrgbBytes != nullptr && imgW > 0 && imgH > 0) {
        jbyte* rgbPtr = env->GetByteArrayElements(jrgbBytes, nullptr);
        if (rgbPtr != nullptr) {
            mtmd_bitmap* bmp = mtmd_bitmap_init((uint32_t)imgW, (uint32_t)imgH, (const unsigned char*)rgbPtr);
            env->ReleaseByteArrayElements(jrgbBytes, rgbPtr, JNI_ABORT);

            if (bmp != nullptr) {
                std::string marker = mtmd_default_marker();
                size_t pos = userHistory.rfind("<|im_start|>user\n");
                if (pos != std::string::npos) {
                    userHistory.insert(pos + 17, marker + "\n");
                } else {
                    userHistory = marker + "\n" + userHistory;
                }
                std::string fullPrompt = "<|im_start|>system\n" + sysStr + "<|im_end|>\n" + userHistory + "<|im_start|>assistant\n";

                mtmd_input_chunks* chunks = mtmd_input_chunks_init();
                mtmd_input_text itxt;
                itxt.text = fullPrompt.c_str();
                itxt.add_special = true;
                itxt.parse_special = true;

                const mtmd_bitmap* bmps[1] = { bmp };
                if (mtmd_tokenize(g_mtmd, chunks, &itxt, bmps, 1) == 0) {
                    llama_pos new_n_past = 0;
                    if (mtmd_helper_eval_chunks(g_mtmd, g_ctx, chunks, 0, 0, 512, true, &new_n_past) == 0) {
                        usedVision = true;
                    }
                }
                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(bmp);
            }
        }
    }

    if (!usedVision) {
        std::string prompt = "<|im_start|>system\n" + sysStr + "<|im_end|>\n" + userHistory + "<|im_start|>assistant\n";
        std::vector<llama_token> tokens(prompt.size() + 32);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
        if (n_tokens < 0) return;
        tokens.resize(n_tokens);
        if (llama_decode(g_ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) return;
    }

    auto t_gen_start = std::chrono::steady_clock::now();
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    int generated = 0;
    int limit = maxTokens > 0 ? maxTokens : 1024;
    for (int i = 0; i < limit && !g_stop; i++) {
        llama_token id = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        generated++;
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            jstring jpiece = env->NewStringUTF(piece.c_str());
            env->CallVoidMethod(thiz, onToken, jpiece);
            env->DeleteLocalRef(jpiece);
        }
        if (llama_decode(g_ctx, llama_batch_get_one(&id, 1)) != 0) break;
    }

    auto t_end = std::chrono::steady_clock::now();
    double gen_sec = std::chrono::duration<double>(t_end - t_gen_start).count();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    double tps = gen_sec > 0.01 ? (generated / gen_sec) : 0.0;

    env->CallVoidMethod(thiz, onStats, generated, tps, total_sec);
    llama_sampler_free(smpl);
}
