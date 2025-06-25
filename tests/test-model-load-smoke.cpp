#include "get-model.h"
#include "gguf.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

// Helper function to read entire file into buffer
static std::vector<uint8_t> read_file_to_buffer(const char * filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return {};
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    if (!file.read(reinterpret_cast<char *>(buffer.data()), file_size)) {
        fprintf(stderr, "Failed to read file: %s\n", filename);
        return {};
    }

    return buffer;
}

// Quick inference test for a single model
static int test_inference_single(llama_model * model, const char * load_method) {
    // Create context for inference
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = 512;
    ctx_params.n_batch              = 8;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "FAILURE: Failed to create context for %s\n", load_method);
        return 1;
    }

    // Simple completion prompt
    const char *        prompt = "The capital of France is";
    const llama_vocab * vocab  = llama_model_get_vocab(model);

    printf("Inference test (%s): \"%s\" -> \"", load_method, prompt);

    // Tokenize prompt
    std::vector<llama_token> prompt_tokens(32);
    int                      n_prompt_tokens =
        llama_tokenize(vocab, prompt, strlen(prompt), prompt_tokens.data(), prompt_tokens.size(), true, false);
    if (n_prompt_tokens < 0) {
        fprintf(stderr, "FAILURE: Failed to tokenize prompt for %s\n", load_method);
        llama_free(ctx);
        return 1;
    }
    prompt_tokens.resize(n_prompt_tokens);

    // Process prompt
    llama_batch prompt_batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, prompt_batch) != 0) {
        fprintf(stderr, "FAILURE: Failed to decode prompt for %s\n", load_method);
        llama_free(ctx);
        return 1;
    }

    // Generate a few tokens for the answer
    std::vector<llama_token> generated_tokens;
    std::string              response;

    for (int i = 0; i < 8; i++) {  // Generate up to 8 tokens
        // Get logits
        float * logits = llama_get_logits(ctx);
        if (!logits) {
            fprintf(stderr, "FAILURE: Failed to get logits for %s\n", load_method);
            llama_free(ctx);
            return 1;
        }

        // Find the token with highest probability
        int32_t     vocab_size = llama_vocab_n_tokens(vocab);
        llama_token next_token = 0;
        float       max_logit  = logits[0];
        for (int32_t j = 1; j < vocab_size; j++) {
            if (logits[j] > max_logit) {
                max_logit  = logits[j];
                next_token = j;
            }
        }

        generated_tokens.push_back(next_token);

        // Detokenize this token
        std::vector<char> token_text(32);
        int token_len = llama_detokenize(vocab, &next_token, 1, token_text.data(), token_text.size(), false, false);
        if (token_len > 0) {
            response.append(token_text.data(), token_len);
        } else {
            // If detokenization fails, show token ID
            response += "[" + std::to_string(next_token) + "]";
        }

        // Check if it's an end token (but allow some common tokens through)
        llama_token eos = llama_vocab_eos(vocab);
        llama_token eot = llama_vocab_eot(vocab);
        if ((eos != -1 && next_token == eos) || (eot != -1 && next_token == eot)) {
            break;
        }

        // Stop if we hit newline or common stop patterns
        if (token_len > 0 && (token_text[0] == '\n' || token_text[0] == '\r')) {
            break;
        }

        // Continue generation - add the new token to context
        llama_batch next_batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(ctx, next_batch) != 0) {
            break;  // Stop on decode error
        }
    }

    printf("%s\"\n", response.c_str());

    // Cleanup
    llama_free(ctx);
    return 0;
}

// Quick smoke test for buffer and file handle loading
static int smoke_test_model_loading(const char * model_path, bool test_inference) {
    // Read model into buffer
    auto buffer = read_file_to_buffer(model_path);
    if (buffer.empty()) {
        fprintf(stderr, "FAILURE: Failed to read model file into buffer\n");
        return 1;
    }

    llama_model_params params = llama_model_default_params();

    // First try full model to detect model type
    params.vocab_only           = false;
    llama_model * test_model    = llama_model_load_from_file(model_path, params);
    bool          is_full_model = (test_model != nullptr);
    if (test_model) {
        llama_model_free(test_model);
    }

    // If it's a full model, test it as a full model
    // Only use vocab_only if the model doesn't have tensors
    if (!is_full_model) {
        // Try vocab-only mode for vocab-only models
        params.vocab_only = true;
    }

    // Test 1: Regular file loading
    llama_model * model_file = llama_model_load_from_file(model_path, params);
    if (!model_file) {
        fprintf(stderr, "FAILURE: Failed to load model from file\n");
        return 1;
    }

    // Test 2: Buffer loading
    llama_model * model_buffer = llama_model_load_from_buffer(buffer.data(), buffer.size(), params);
    if (!model_buffer) {
        fprintf(stderr, "FAILURE: Failed to load model from buffer\n");
        llama_model_free(model_file);
        return 1;
    }

    // Test 3: File handle loading
    FILE * file_handle = fopen(model_path, "rb");
    if (!file_handle) {
        fprintf(stderr, "FAILURE: Failed to open file handle\n");
        llama_model_free(model_file);
        llama_model_free(model_buffer);
        return 1;
    }

    llama_model * model_handle = llama_model_load_from_file_handle(file_handle, params);
    fclose(file_handle);

    if (!model_handle) {
        fprintf(stderr, "FAILURE: Failed to load model from file handle\n");
        llama_model_free(model_file);
        llama_model_free(model_buffer);
        return 1;
    }

    // Quick comparison - just check vocab sizes match
    const llama_vocab * vocab_file   = llama_model_get_vocab(model_file);
    const llama_vocab * vocab_buffer = llama_model_get_vocab(model_buffer);
    const llama_vocab * vocab_handle = llama_model_get_vocab(model_handle);

    int32_t size_file   = llama_vocab_n_tokens(vocab_file);
    int32_t size_buffer = llama_vocab_n_tokens(vocab_buffer);
    int32_t size_handle = llama_vocab_n_tokens(vocab_handle);

    bool success = true;
    if (size_file != size_buffer || size_file != size_handle) {
        fprintf(stderr,
                "FAILURE: Vocabulary sizes don't match: file=%d, buffer=%d, handle=%d\n",
                size_file,
                size_buffer,
                size_handle);
        success = false;
    }

    // If full model and inference requested, test inference on all three loading methods
    if (success && is_full_model && test_inference) {
        if (test_inference_single(model_file, "file") != 0) {
            success = false;
        }
        if (test_inference_single(model_buffer, "buffer") != 0) {
            success = false;
        }
        if (test_inference_single(model_handle, "file_handle") != 0) {
            success = false;
        }
    }

    // Cleanup
    llama_model_free(model_file);
    llama_model_free(model_buffer);
    llama_model_free(model_handle);

    return success ? 0 : 1;
}

// Quick smoke test for GGUF loading methods
static int smoke_test_gguf_loading(const char * model_path) {
    auto buffer = read_file_to_buffer(model_path);
    if (buffer.empty()) {
        fprintf(stderr, "FAILURE: Failed to read model file into buffer for GGUF test\n");
        return 1;
    }

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };

    // Test file-based GGUF loading
    auto * ctx_file = gguf_init_from_file(model_path, params);
    if (!ctx_file) {
        fprintf(stderr, "FAILURE: gguf_init_from_file failed\n");
        return 1;
    }

    // Test buffer-based GGUF loading
    auto * ctx_buffer = gguf_init_from_buffer(buffer.data(), buffer.size(), params);
    if (!ctx_buffer) {
        fprintf(stderr, "FAILURE: gguf_init_from_buffer failed\n");
        gguf_free(ctx_file);
        return 1;
    }

    // Test file handle-based GGUF loading
    FILE * file_handle = fopen(model_path, "rb");
    if (!file_handle) {
        fprintf(stderr, "FAILURE: Failed to open file handle for GGUF test\n");
        gguf_free(ctx_file);
        gguf_free(ctx_buffer);
        return 1;
    }

    auto * ctx_handle = gguf_init_from_file_handle(file_handle, params);
    fclose(file_handle);

    if (!ctx_handle) {
        fprintf(stderr, "FAILURE: gguf_init_from_file_handle failed\n");
        gguf_free(ctx_file);
        gguf_free(ctx_buffer);
        return 1;
    }

    // Quick comparison
    int64_t n_kv_file   = gguf_get_n_kv(ctx_file);
    int64_t n_kv_buffer = gguf_get_n_kv(ctx_buffer);
    int64_t n_kv_handle = gguf_get_n_kv(ctx_handle);

    bool success = true;
    if (n_kv_file != n_kv_buffer || n_kv_file != n_kv_handle) {
        fprintf(stderr,
                "FAILURE: KV pair counts don't match: file=%lld, buffer=%lld, handle=%lld\n",
                (long long) n_kv_file,
                (long long) n_kv_buffer,
                (long long) n_kv_handle);
        success = false;
    }

    gguf_free(ctx_file);
    gguf_free(ctx_buffer);
    gguf_free(ctx_handle);

    return success ? 0 : 1;
}

int main(int argc, char * argv[]) {
    auto * model_path = get_model_or_exit(argc, argv);

    // Check for --inference flag
    bool test_inference = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--inference") == 0) {
            test_inference = true;
            break;
        }
    }

    // Initialize backend
    llama_backend_init();

    int failures = 0;

    // Run smoke tests
    failures += smoke_test_gguf_loading(model_path);
    failures += smoke_test_model_loading(model_path, test_inference);

    // Cleanup
    llama_backend_free();

    if (failures != 0) {
        fprintf(stderr, "FAILURE: Smoke test failed with %d errors\n", failures);
        return EXIT_FAILURE;
    }

    // Success - no output in success case
    return EXIT_SUCCESS;
}
