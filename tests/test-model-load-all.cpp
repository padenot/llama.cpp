#include "get-model.h"
#include "llama.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

// Helper struct to hold file state for IO functions
struct file_io_state {
    FILE* file;
    file_io_state(FILE* f) : file(f) {}
};

// IO function implementations that wrap FILE* operations
static int file_io_read(void* user_data, void* buffer, size_t size) {
    file_io_state* state = static_cast<file_io_state*>(user_data);
    size_t bytes_read = fread(buffer, 1, size, state->file);
    return (bytes_read == size) ? 0 : -1;
}

static int file_io_seek(void* user_data, size_t position) {
    file_io_state* state = static_cast<file_io_state*>(user_data);
    return fseek(state->file, position, SEEK_SET);
}

static size_t file_io_tell(void* user_data) {
    file_io_state* state = static_cast<file_io_state*>(user_data);
    return ftell(state->file);
}

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

// Compare two models by checking basic properties
static bool compare_models(const llama_model * model_a, const llama_model * model_b, 
                          const char * name_a, const char * name_b, const char * test_name) {
    if (!model_a || !model_b) {
        fprintf(stderr, "FAILURE: One or both models failed to load for test: %s\n", test_name);
        return false;
    }

    const llama_vocab * vocab_a = llama_model_get_vocab(model_a);
    const llama_vocab * vocab_b = llama_model_get_vocab(model_b);

    if (!vocab_a || !vocab_b) {
        fprintf(stderr, "FAILURE: Failed to get vocabularies for test: %s\n", test_name);
        return false;
    }

    // Check vocabulary size
    int32_t vocab_size_a = llama_vocab_n_tokens(vocab_a);
    int32_t vocab_size_b = llama_vocab_n_tokens(vocab_b);

    if (vocab_size_a != vocab_size_b) {
        fprintf(stderr, "FAILURE: Vocabulary sizes differ for test %s: %s=%d, %s=%d\n", test_name, name_a, vocab_size_a, name_b, vocab_size_b);
        return false;
    }

    // Check model size
    uint64_t size_a = llama_model_size(model_a);
    uint64_t size_b = llama_model_size(model_b);

    if (size_a != size_b) {
        fprintf(stderr, "FAILURE: Model sizes differ for test %s: %s=%llu, %s=%llu\n", 
                test_name, name_a, (unsigned long long)size_a, name_b, (unsigned long long)size_b);
        return false;
    }

    // Check parameter count
    uint64_t params_a = llama_model_n_params(model_a);
    uint64_t params_b = llama_model_n_params(model_b);

    if (params_a != params_b) {
        fprintf(stderr, "FAILURE: Parameter counts differ for test %s: %s=%llu, %s=%llu\n", 
                test_name, name_a, (unsigned long long)params_a, name_b, (unsigned long long)params_b);
        return false;
    }

    // Check embedding dimensions
    int32_t n_embd_a = llama_model_n_embd(model_a);
    int32_t n_embd_b = llama_model_n_embd(model_b);

    if (n_embd_a != n_embd_b) {
        fprintf(stderr, "FAILURE: Embedding dimensions differ for test %s: %s=%d, %s=%d\n", test_name, name_a, n_embd_a, name_b, n_embd_b);
        return false;
    }

    // Check layer count
    int32_t n_layer_a = llama_model_n_layer(model_a);
    int32_t n_layer_b = llama_model_n_layer(model_b);

    if (n_layer_a != n_layer_b) {
        fprintf(stderr, "FAILURE: Layer counts differ for test %s: %s=%d, %s=%d\n", test_name, name_a, n_layer_a, name_b, n_layer_b);
        return false;
    }

    // Check special tokens
    llama_token bos_a = llama_vocab_bos(vocab_a);
    llama_token bos_b = llama_vocab_bos(vocab_b);

    if (bos_a != bos_b) {
        fprintf(stderr, "FAILURE: BOS tokens differ for test %s: %s=%d, %s=%d\n", test_name, name_a, bos_a, name_b, bos_b);
        return false;
    }

    llama_token eos_a = llama_vocab_eos(vocab_a);
    llama_token eos_b = llama_vocab_eos(vocab_b);

    if (eos_a != eos_b) {
        fprintf(stderr, "FAILURE: EOS tokens differ for test %s: %s=%d, %s=%d\n", test_name, name_a, eos_a, name_b, eos_b);
        return false;
    }

    return true;
}

// Test tokenization between two models
static bool test_tokenization_comparison(const llama_model * model_a, const llama_model * model_b,
                                        const char * name_a, const char * name_b,
                                        const char * test_text) {
    const llama_vocab * vocab_a = llama_model_get_vocab(model_a);
    const llama_vocab * vocab_b = llama_model_get_vocab(model_b);

    // Tokenize with both models
    std::vector<llama_token> tokens_a(256);
    std::vector<llama_token> tokens_b(256);

    int32_t n_tokens_a = llama_tokenize(vocab_a, test_text, strlen(test_text), 
                                       tokens_a.data(), tokens_a.size(), true, false);
    int32_t n_tokens_b = llama_tokenize(vocab_b, test_text, strlen(test_text), 
                                       tokens_b.data(), tokens_b.size(), true, false);

    if (n_tokens_a != n_tokens_b) {
        fprintf(stderr, "FAILURE: Token counts differ for \"%s\": %s=%d, %s=%d\n", 
                test_text, name_a, n_tokens_a, name_b, n_tokens_b);
        return false;
    }

    if (n_tokens_a < 0) {
        fprintf(stderr, "FAILURE: Tokenization failed with error: %d\n", n_tokens_a);
        return false;
    }

    tokens_a.resize(n_tokens_a);
    tokens_b.resize(n_tokens_b);

    for (int i = 0; i < n_tokens_a; i++) {
        if (tokens_a[i] != tokens_b[i]) {
            fprintf(stderr, "FAILURE: Token %d differs: %s=%d, %s=%d\n", 
                    i, name_a, tokens_a[i], name_b, tokens_b[i]);
            return false;
        }
    }

    // Test detokenization
    std::vector<char> text_a(1024);
    std::vector<char> text_b(1024);

    int32_t text_len_a = llama_detokenize(vocab_a, tokens_a.data(), n_tokens_a, 
                                         text_a.data(), text_a.size(), true, false);
    int32_t text_len_b = llama_detokenize(vocab_b, tokens_b.data(), n_tokens_b, 
                                         text_b.data(), text_b.size(), true, false);

    if (text_len_a != text_len_b) {
        fprintf(stderr, "FAILURE: Detokenized text lengths differ: %s=%d, %s=%d\n", 
                name_a, text_len_a, name_b, text_len_b);
        return false;
    }

    if (text_len_a < 0) {
        fprintf(stderr, "FAILURE: Detokenization failed with error: %d\n", text_len_a);
        return false;
    }

    text_a.resize(text_len_a);
    text_b.resize(text_len_b);

    if (memcmp(text_a.data(), text_b.data(), text_len_a) != 0) {
        fprintf(stderr, "FAILURE: Detokenized texts differ\n");
        return false;
    }

    return true;
}

// Run comprehensive test comparing all four loading methods
int run_comprehensive_test(const char * model_path, const char * test_name, llama_model_params params) {
    // Read buffer for buffer-based loading
    auto buffer = read_file_to_buffer(model_path);
    if (buffer.empty()) {
        fprintf(stderr, "FAILURE: Failed to read model file into buffer for test: %s\n", test_name);
        return 1;
    }

    // Load model using four different methods
    // Method 1: Regular file loading
    llama_model * model_file = llama_model_load_from_file(model_path, params);
    if (!model_file) {
        fprintf(stderr, "FAILURE: Failed to load model from file for test: %s\n", test_name);
        return 1;
    }

    // Method 2: Buffer loading
    llama_model * model_buffer = llama_model_load_from_buffer(buffer.data(), buffer.size(), params);
    if (!model_buffer) {
        fprintf(stderr, "FAILURE: Failed to load model from buffer for test: %s\n", test_name);
        llama_model_free(model_file);
        return 1;
    }

    // Method 3: File handle loading
    FILE * file_handle = fopen(model_path, "rb");
    if (!file_handle) {
        fprintf(stderr, "FAILURE: Failed to open file handle for test: %s\n", test_name);
        llama_model_free(model_file);
        llama_model_free(model_buffer);
        return 1;
    }

    llama_model * model_handle = llama_model_load_from_file_handle(file_handle, params);
    fclose(file_handle);
    
    if (!model_handle) {
        fprintf(stderr, "FAILURE: Failed to load model from file handle for test: %s\n", test_name);
        llama_model_free(model_file);
        llama_model_free(model_buffer);
        return 1;
    }

    // Method 4: IO functions loading
    FILE * io_file = fopen(model_path, "rb");
    if (!io_file) {
        fprintf(stderr, "FAILURE: Failed to open file for IO functions for test: %s\n", test_name);
        llama_model_free(model_file);
        llama_model_free(model_buffer);
        llama_model_free(model_handle);
        return 1;
    }

    file_io_state io_state(io_file);
    gguf_io_functions io_funcs = {
        .user_data = &io_state,
        .read = file_io_read,
        .seek = file_io_seek,
        .tell = file_io_tell
    };

    llama_model * model_io = llama_model_load_from_io(io_funcs, params);
    fclose(io_file);

    if (!model_io) {
        fprintf(stderr, "FAILURE: Failed to load model from IO functions for test: %s\n", test_name);
        llama_model_free(model_file);
        llama_model_free(model_buffer);
        llama_model_free(model_handle);
        return 1;
    }

    // Compare models pairwise
    bool all_match = true;
    
    if (!compare_models(model_file, model_buffer, "file", "buffer", test_name)) {
        all_match = false;
    }
    
    if (!compare_models(model_file, model_handle, "file", "handle", test_name)) {
        all_match = false;
    }
    
    if (!compare_models(model_file, model_io, "file", "io", test_name)) {
        all_match = false;
    }
    
    if (!compare_models(model_buffer, model_handle, "buffer", "handle", test_name)) {
        all_match = false;
    }
    
    if (!compare_models(model_buffer, model_io, "buffer", "io", test_name)) {
        all_match = false;
    }
    
    if (!compare_models(model_handle, model_io, "handle", "io", test_name)) {
        all_match = false;
    }

    // Test tokenization if models match
    if (all_match && !params.vocab_only) {
        const char * test_texts[] = {
            "Hello world",
            "The quick brown fox jumps over the lazy dog.",
            "Hello, world! How are you?",
            "Testing 123 with numbers and symbols: @#$%",
            ""  // Empty string test
        };

        for (const char * test_text : test_texts) {
            if (!test_tokenization_comparison(model_file, model_buffer, 
                                             "file", "buffer", test_text)) {
                all_match = false;
                break;
            }
            
            if (!test_tokenization_comparison(model_file, model_handle, 
                                             "file", "handle", test_text)) {
                all_match = false;
                break;
            }
            
            if (!test_tokenization_comparison(model_file, model_io, 
                                             "file", "io", test_text)) {
                all_match = false;
                break;
            }
            
            if (!test_tokenization_comparison(model_buffer, model_handle, 
                                             "buffer", "handle", test_text)) {
                all_match = false;
                break;
            }
            
            if (!test_tokenization_comparison(model_buffer, model_io, 
                                             "buffer", "io", test_text)) {
                all_match = false;
                break;
            }
            
            if (!test_tokenization_comparison(model_handle, model_io, 
                                             "handle", "io", test_text)) {
                all_match = false;
                break;
            }
        }
    }

    // Cleanup
    llama_model_free(model_file);
    llama_model_free(model_buffer);
    llama_model_free(model_handle);
    llama_model_free(model_io);

    if (!all_match) {
        fprintf(stderr, "FAILURE: Test '%s' failed\n", test_name);
        return 1;
    }
    return 0;
}

// Test GGUF buffer loading
int test_gguf_buffer_loading(const char * model_path) {
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

    int64_t n_kv_file = gguf_get_n_kv(ctx_file);
    int64_t n_tensors_file = gguf_get_n_tensors(ctx_file);

    // Test buffer-based GGUF loading
    auto * ctx_buffer = gguf_init_from_buffer(buffer.data(), buffer.size(), params);
    if (!ctx_buffer) {
        fprintf(stderr, "FAILURE: gguf_init_from_buffer failed\n");
        gguf_free(ctx_file);
        return 1;
    }

    int64_t n_kv_buffer = gguf_get_n_kv(ctx_buffer);
    int64_t n_tensors_buffer = gguf_get_n_tensors(ctx_buffer);

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

    int64_t n_kv_handle = gguf_get_n_kv(ctx_handle);
    int64_t n_tensors_handle = gguf_get_n_tensors(ctx_handle);

    // Test IO functions-based GGUF loading
    FILE * io_file = fopen(model_path, "rb");
    if (!io_file) {
        fprintf(stderr, "FAILURE: Failed to open file for GGUF IO test\n");
        gguf_free(ctx_file);
        gguf_free(ctx_buffer);
        gguf_free(ctx_handle);
        return 1;
    }

    file_io_state io_state(io_file);
    gguf_io_functions io_funcs = {
        .user_data = &io_state,
        .read = file_io_read,
        .seek = file_io_seek,
        .tell = file_io_tell
    };

    auto * ctx_io = gguf_init_from_io(io_funcs, params);
    fclose(io_file);

    if (!ctx_io) {
        fprintf(stderr, "FAILURE: gguf_init_from_io failed\n");
        gguf_free(ctx_file);
        gguf_free(ctx_buffer);
        gguf_free(ctx_handle);
        return 1;
    }

    int64_t n_kv_io = gguf_get_n_kv(ctx_io);
    int64_t n_tensors_io = gguf_get_n_tensors(ctx_io);

    // Compare results
    bool match = true;
    if (n_kv_file != n_kv_buffer || n_kv_file != n_kv_handle || n_kv_file != n_kv_io) {
        fprintf(stderr, "FAILURE: KV pair counts don't match: file=%lld, buffer=%lld, handle=%lld, io=%lld\n",
                (long long)n_kv_file, (long long)n_kv_buffer, (long long)n_kv_handle, (long long)n_kv_io);
        match = false;
    }
    if (n_tensors_file != n_tensors_buffer || n_tensors_file != n_tensors_handle || n_tensors_file != n_tensors_io) {
        fprintf(stderr, "FAILURE: Tensor counts don't match: file=%lld, buffer=%lld, handle=%lld, io=%lld\n",
                (long long)n_tensors_file, (long long)n_tensors_buffer, (long long)n_tensors_handle, (long long)n_tensors_io);
        match = false;
    }

    gguf_free(ctx_file);
    gguf_free(ctx_buffer);
    gguf_free(ctx_handle);
    gguf_free(ctx_io);

    return match ? 0 : 1;
}

int main(int argc, char * argv[]) {
    auto * model_path = get_model_or_exit(argc, argv);

    // Initialize backend
    llama_backend_init();

    int total_failures = 0;

    // Test GGUF loading first
    total_failures += test_gguf_buffer_loading(model_path);

    // Detect if this is a vocab-only model by trying to load it without vocab_only flag
    bool is_vocab_only_model = false;
    {
        auto params = llama_model_default_params();
        params.vocab_only = false;
        llama_model * test_model = llama_model_load_from_file(model_path, params);
        if (!test_model) {
            // Failed to load as full model, try as vocab-only
            params.vocab_only = true;
            test_model = llama_model_load_from_file(model_path, params);
            if (test_model) {
                is_vocab_only_model = true;
                llama_model_free(test_model);
            }
        } else {
            llama_model_free(test_model);
        }
    }

    // Test 1: Default parameters (skip for vocab-only models as it would fail)
    if (!is_vocab_only_model) {
        auto params = llama_model_default_params();
        total_failures += run_comprehensive_test(model_path, "Default Parameters", params);
    }

    // Test 2: Vocab only mode (always run)
    {
        auto params = llama_model_default_params();
        params.vocab_only = true;
        total_failures += run_comprehensive_test(model_path, "Vocab Only", params);
    }

    // Test 3: No memory mapping (skip for vocab-only models)
    if (!is_vocab_only_model) {
        auto params = llama_model_default_params();
        params.use_mmap = false;
        total_failures += run_comprehensive_test(model_path, "No Memory Mapping", params);
    }

    // Test 4: CPU only (skip for vocab-only models)
    if (!is_vocab_only_model) {
        auto params = llama_model_default_params();
        params.main_gpu = -1;
        total_failures += run_comprehensive_test(model_path, "CPU Only", params);
    }

    // Test 5: With tensor checking (skip for vocab-only models)
    if (!is_vocab_only_model) {
        auto params = llama_model_default_params();
        params.check_tensors = true;
        total_failures += run_comprehensive_test(model_path, "Tensor Checking", params);
    }

    // Cleanup
    llama_backend_free();

    if (total_failures != 0) {
        fprintf(stderr, "FAILURE: %d test(s) failed\n", total_failures);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}