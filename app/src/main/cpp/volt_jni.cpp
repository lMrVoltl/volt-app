#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include "llama.h"

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static std::atomic<bool> g_stop{false};

extern "C" JNIEXPORT jboolean JNICALL
Java_com_voltage_ai_MainActivity_nativeLoadModel(JNIEnv* env, jobject, jstring jpath, jint threads, jint nCtx) {
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
    return g_ctx != nullptr ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_voltage_ai_MainActivity_nativeStop(JNIEnv*, jobject) {
    g_stop = true;
}

extern "C" JNIEXPORT void JNICALL
Java_com_voltage_ai_MainActivity_nativeGenerate(
    JNIEnv* env, jobject thiz, jstring jprompt, jstring jsysPrompt,
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
        : "Ты Volt AI — быстрый локальный ассистент на смартфоне. Отвечай структурированно, используя Markdown.";
    std::string prompt = "<|im_start|>system\n" + sysStr + "<|im_end|>\n" + std::string(raw) + "<|im_start|>assistant\n";
    env->ReleaseStringUTFChars(jprompt, raw);
    env->ReleaseStringUTFChars(jsysPrompt, sysRaw);

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) return;
    tokens.resize(n_tokens);

    auto t_start = std::chrono::steady_clock::now();
    llama_memory_clear(llama_get_memory(g_ctx), true);
    if (llama_decode(g_ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) return;

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
