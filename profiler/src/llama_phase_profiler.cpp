#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using steady_clock_t = std::chrono::steady_clock;

static inline int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(steady_clock_t::now().time_since_epoch()).count();
}

static inline double us_to_ms(const int64_t us) { return us / 1000.0; }

static std::string read_file_all(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static double read_rss_mb() {
    // VmRSS from /proc/self/status in kB
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            // "VmRSS:\t  123456 kB"
            std::istringstream iss(line);
            std::string key, unit;
            double kb = 0;
            iss >> key >> kb >> unit;
            return kb / 1024.0;
        }
    }
    return -1.0;
}

static uint64_t file_size_bytes(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) return 0;
    return (uint64_t) f.tellg();
}

struct PhaseAgg {
    double total_us = 0.0;
    int64_t n_tokens = 0;
    std::unordered_map<std::string, double> per_op_us;
    std::unordered_map<std::string, double> per_sublayer_us;
    std::vector<double> per_step_us;
};

struct RunAgg {
    // phase timers (us)
    std::unordered_map<std::string, double> phases_us;
    // memory snapshots (MB)
    std::unordered_map<std::string, double> mem_mb;
    PhaseAgg prefill;
    PhaseAgg decode;
    double sample_us = 0.0;
    int64_t sample_calls = 0;
    double detokenize_us = 0.0;
    int64_t detokenize_calls = 0;
};

enum class CurrentPhase : int {
    none = 0,
    prefill = 1,
    decode = 2,
};

struct OpEvent {
    std::string phase;
    int step = -1;
    std::string op;
    std::string name;
    int layer = -1;
    std::string sublayer;
    int64_t time_us = 0;
};

struct CallbackState {
    std::atomic<int> phase{ (int)CurrentPhase::none };
    std::atomic<int> decode_step{-1};
    std::atomic<bool> enabled{false};

    // last op attribution (single-threaded decode path in llama.cpp)
    int64_t last_ts_us = 0;
    std::string last_op;
    std::string last_name;
    int last_layer = -1;
    std::string last_sublayer;

    bool dump_raw_ops = false;
    std::vector<OpEvent> raw_ops;

    PhaseAgg * prefill = nullptr;
    PhaseAgg * decode = nullptr;
};

static std::pair<int, std::string> parse_layer_sublayer(const char * tname) {
    // Names typically look like:
    //   "blk.0.attn_q.weight" / "blk.12.ffn_down.weight" / "blk.2.attn_norm.weight"
    // We'll map to coarse tags used for aggregation.
    if (!tname || !tname[0]) return {-1, ""};

    static const std::regex re_blk(R"(blk\.(\d+)\.([A-Za-z0-9_]+))");
    std::cmatch m;
    if (!std::regex_search(tname, m, re_blk)) {
        return {-1, ""};
    }
    const int layer = std::atoi(m[1].str().c_str());
    const std::string part = m[2].str();

    auto tag = [&](const char * s) { return std::string(s); };

    // attention projections sometimes appear as attn_q/attn_k/attn_v or combined attn_qkv
    if (part == "attn_q" || part.find("attn_q") != std::string::npos) return {layer, tag("q_proj")};
    if (part == "attn_k" || part.find("attn_k") != std::string::npos) return {layer, tag("k_proj")};
    if (part == "attn_v" || part.find("attn_v") != std::string::npos) return {layer, tag("v_proj")};
    if (part.find("attn_qkv") != std::string::npos) return {layer, tag("qkv_proj")};
    if (part.find("attn_output") != std::string::npos) return {layer, tag("o_proj")};
    if (part.find("attn_norm") != std::string::npos) return {layer, tag("rms_norm")};
    if (part.find("ffn_norm") != std::string::npos) return {layer, tag("rms_norm")};

    if (part.find("ffn_gate") != std::string::npos) return {layer, tag("ffn_gate")};
    if (part.find("ffn_up") != std::string::npos) return {layer, tag("ffn_up")};
    if (part.find("ffn_down") != std::string::npos) return {layer, tag("ffn_down")};

    if (part.find("rope") != std::string::npos) return {layer, tag("rope")};

    return {layer, ""};
}

static void add_us(PhaseAgg & agg, const std::string & key, const double us) {
    if (key.empty()) return;
    agg.per_op_us[key] += us;
}

static void add_sublayer_us(PhaseAgg & agg, const std::string & key, const double us) {
    if (key.empty()) return;
    agg.per_sublayer_us[key] += us;
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = reinterpret_cast<CallbackState *>(user_data);
    if (!st || !st->enabled.load(std::memory_order_relaxed)) {
        return true;
    }
    // ggml calls the callback twice per op:
    //   ask=true  -> "do you want this tensor's data afterwards?" (we say yes)
    //   ask=false -> "this op has just finished executing"
    // Only the ask=false calls carry useful wall-clock boundaries for us.
    if (ask) {
        return true;
    }

    const int64_t ts = now_us();

    // Attribute time since last callback to the previous op (best-effort).
    if (st->last_ts_us != 0) {
        const int64_t delta = ts - st->last_ts_us;
        const int cur_phase = st->phase.load(std::memory_order_relaxed);
        PhaseAgg * agg = nullptr;
        const char * phase_name = "none";
        int step = -1;
        if (cur_phase == (int)CurrentPhase::prefill) {
            agg = st->prefill;
            phase_name = "prefill";
        } else if (cur_phase == (int)CurrentPhase::decode) {
            agg = st->decode;
            phase_name = "decode";
            step = st->decode_step.load(std::memory_order_relaxed);
        }

        if (agg && !st->last_op.empty()) {
            agg->total_us += (double)delta;
            add_us(*agg, st->last_op, (double)delta);
            add_sublayer_us(*agg, st->last_sublayer, (double)delta);

            if (st->dump_raw_ops) {
                OpEvent ev;
                ev.phase = phase_name;
                ev.step = step;
                ev.op = st->last_op;
                ev.name = st->last_name;
                ev.layer = st->last_layer;
                ev.sublayer = st->last_sublayer;
                ev.time_us = delta;
                st->raw_ops.push_back(std::move(ev));
            }
        }
    }

    // Record the current tensor as the "last" op for next delta.
    st->last_ts_us = ts;
    st->last_op = t ? ggml_op_name(t->op) : "UNKNOWN";
    st->last_name = (t && t->name[0]) ? t->name : "";
    auto [layer, sublayer] = parse_layer_sublayer(st->last_name.c_str());
    st->last_layer = layer;
    st->last_sublayer = std::move(sublayer);

    return true;
}

static void json_escape_and_write(std::ostream & os, const std::string & s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    os << buf;
                } else {
                    os << c;
                }
        }
    }
    os << '"';
}

static void write_kv_num(std::ostream & os, const std::string & k, const double v, bool & first) {
    if (!first) os << ",";
    first = false;
    json_escape_and_write(os, k);
    os << ":" << v;
}

static void write_kv_int(std::ostream & os, const std::string & k, const int64_t v, bool & first) {
    if (!first) os << ",";
    first = false;
    json_escape_and_write(os, k);
    os << ":" << v;
}

static void write_kv_str(std::ostream & os, const std::string & k, const std::string & v, bool & first) {
    if (!first) os << ",";
    first = false;
    json_escape_and_write(os, k);
    os << ":";
    json_escape_and_write(os, v);
}

static void write_map_num(std::ostream & os, const std::unordered_map<std::string, double> & m) {
    os << "{";
    bool first = true;
    for (const auto & [k, v] : m) {
        write_kv_num(os, k, us_to_ms((int64_t) v), first);
    }
    os << "}";
}

static void write_map_mb(std::ostream & os, const std::unordered_map<std::string, double> & m) {
    os << "{";
    bool first = true;
    for (const auto & [k, v] : m) {
        write_kv_num(os, k, v, first);
    }
    os << "}";
}

static void usage() {
    std::fprintf(stderr,
        "llama-phase-profiler\n"
        "Usage:\n"
        "  llama-phase-profiler -m model.gguf --prompt-file prompt.txt --n-gen 64 -t 4 [--seed 42] [--n-ctx 2048] [--dump-raw-ops]\n"
    );
}

struct Args {
    std::string model_path;
    std::string prompt_file;
    int n_gen = 64;
    int n_threads = 4;
    int seed = 42;
    int n_ctx = 2048;
    bool dump_raw_ops = false;
};

static bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; i++) {
        const std::string s = argv[i];
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (s == "-m" || s == "--model") {
            const char * v = need(s.c_str()); if (!v) return false;
            a.model_path = v;
        } else if (s == "--prompt-file") {
            const char * v = need(s.c_str()); if (!v) return false;
            a.prompt_file = v;
        } else if (s == "--n-gen") {
            const char * v = need(s.c_str()); if (!v) return false;
            a.n_gen = std::atoi(v);
        } else if (s == "-t" || s == "--threads") {
            const char * v = need(s.c_str()); if (!v) return false;
            a.n_threads = std::atoi(v);
        } else if (s == "--seed") {
            const char * v = need(s.c_str()); if (!v) return false;
            a.seed = std::atoi(v);
        } else if (s == "--n-ctx") {
            const char * v = need(s.c_str()); if (!v) return false;
            a.n_ctx = std::atoi(v);
        } else if (s == "--dump-raw-ops") {
            a.dump_raw_ops = true;
        } else if (s == "-h" || s == "--help") {
            usage();
            return false;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", s.c_str());
            usage();
            return false;
        }
    }
    if (a.model_path.empty() || a.prompt_file.empty()) {
        usage();
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 2;
    }

    RunAgg run;
    run.mem_mb["rss_before_load"] = read_rss_mb();

    const int64_t t0 = now_us();

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // v1: CPU-only on Android

    llama_model * model = nullptr;
    {
        const int64_t t_load0 = now_us();
        model = llama_load_model_from_file(args.model_path.c_str(), mparams);
        const int64_t t_load1 = now_us();
        run.phases_us["model_load"] = (double)(t_load1 - t_load0);
    }
    if (!model) {
        std::fprintf(stderr, "{\"error\":\"failed to load model\"}\n");
        return 1;
    }

    run.mem_mb["rss_after_load"] = read_rss_mb();

    // Read prompt
    const std::string prompt = read_file_all(args.prompt_file);

    // Tokenize
    std::vector<llama_token> prompt_tokens;
    {
        const int64_t t_tok0 = now_us();
        const int n_est = -llama_tokenize(model, prompt.c_str(), (int)prompt.size(), nullptr, 0, true, true);
        prompt_tokens.resize((size_t) n_est);
        const int n = llama_tokenize(model, prompt.c_str(), (int)prompt.size(), prompt_tokens.data(), (int)prompt_tokens.size(), true, true);
        if (n < 0) {
            std::fprintf(stderr, "{\"error\":\"tokenize failed\"}\n");
            llama_free_model(model);
            return 1;
        }
        prompt_tokens.resize((size_t) n);
        const int64_t t_tok1 = now_us();
        run.phases_us["tokenize"] = (double)(t_tok1 - t_tok0);
    }

    // Context init
    CallbackState cb;
    cb.dump_raw_ops = args.dump_raw_ops;
    cb.prefill = &run.prefill;
    cb.decode = &run.decode;

    llama_context * ctx = nullptr;
    {
        const int64_t t_ctx0 = now_us();
        llama_context_params cparams = llama_context_default_params();
        cparams.seed = (uint32_t) args.seed;
        cparams.n_ctx = (uint32_t) args.n_ctx;
        cparams.n_batch = (uint32_t) std::max<int>(1, (int)prompt_tokens.size());
        cparams.n_ubatch = 512;
        cparams.n_threads = (uint32_t) args.n_threads;
        cparams.n_threads_batch = (uint32_t) args.n_threads;
        cparams.cb_eval = eval_cb;
        cparams.cb_eval_user_data = &cb;

        ctx = llama_new_context_with_model(model, cparams);
        const int64_t t_ctx1 = now_us();
        run.phases_us["context_init"] = (double)(t_ctx1 - t_ctx0);
    }
    if (!ctx) {
        std::fprintf(stderr, "{\"error\":\"failed to create context\"}\n");
        llama_free_model(model);
        return 1;
    }

    run.mem_mb["rss_after_context"] = read_rss_mb();

    // Prefill
    cb.enabled.store(true, std::memory_order_relaxed);
    cb.phase.store((int)CurrentPhase::prefill, std::memory_order_relaxed);
    cb.decode_step.store(-1, std::memory_order_relaxed);
    cb.last_ts_us = 0;
    cb.last_op.clear();
    cb.last_name.clear();
    run.prefill.total_us = 0;
    run.prefill.n_tokens = (int64_t)prompt_tokens.size();

    {
        const int64_t t_pf0 = now_us();
        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(), (int)prompt_tokens.size(), 0, 0);
        const int rc = llama_decode(ctx, batch);
        const int64_t t_pf1 = now_us();
        run.phases_us["prefill"] = (double)(t_pf1 - t_pf0);

        // Flush tail time (time since last callback) into last op
        if (cb.last_ts_us != 0) {
            const int64_t ts = now_us();
            const int64_t delta = ts - cb.last_ts_us;
            if (!cb.last_op.empty()) {
                run.prefill.total_us += (double)delta;
                run.prefill.per_op_us[cb.last_op] += (double)delta;
                run.prefill.per_sublayer_us[cb.last_sublayer] += (double)delta;
            }
        }

        if (rc != 0) {
            std::fprintf(stderr, "{\"error\":\"decode failed (prefill)\",\"rc\":%d}\n", rc);
            cb.enabled.store(false, std::memory_order_relaxed);
            llama_free(ctx);
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
    }

    run.mem_mb["rss_after_prefill"] = read_rss_mb();
    run.mem_mb["rss_peak"] = std::max(run.mem_mb["rss_after_prefill"], run.mem_mb["rss_after_context"]);

    // Greedy decode: sample from current logits, then decode that token at n_past (same pattern as
    // examples/save-load-state/save-load-state.cpp).
    cb.phase.store((int)CurrentPhase::decode, std::memory_order_relaxed);
    run.decode.n_tokens = args.n_gen;
    run.decode.total_us = 0;
    run.decode.per_step_us.clear();
    run.decode.per_step_us.reserve((size_t)args.n_gen);

    std::vector<llama_token> generated;
    generated.reserve((size_t)args.n_gen);

    const int32_t n_vocab = llama_n_vocab(model);
    std::vector<llama_token_data> candidates;
    candidates.reserve((size_t)n_vocab);

    int32_t n_past = (int32_t)prompt_tokens.size();

    {
        const int64_t t_dec0 = now_us();
        for (int i = 0; i < args.n_gen; i++) {
            cb.decode_step.store(i, std::memory_order_relaxed);
            cb.last_ts_us = 0;
            cb.last_op.clear();
            cb.last_name.clear();

            // Sample (wall time for sampling only)
            const int64_t t_s0 = now_us();
            float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) {
                std::fprintf(stderr, "{\"error\":\"no logits at step %d\"}\n", i);
                break;
            }
            candidates.clear();
            for (llama_token tid = 0; tid < n_vocab; tid++) {
                candidates.push_back(llama_token_data{ tid, logits[tid], 0.0f });
            }
            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
            const llama_token new_token = llama_sample_token_greedy(ctx, &candidates_p);
            const int64_t t_s1 = now_us();
            run.sample_us += (double)(t_s1 - t_s0);
            run.sample_calls += 1;

            if (llama_token_is_eog(model, new_token)) {
                break;
            }
            generated.push_back(new_token);

            const int64_t t_step0 = now_us();
            llama_batch batch = llama_batch_get_one(&generated.back(), 1, n_past, 0);
            const int rc = llama_decode(ctx, batch);
            const int64_t t_step1 = now_us();

            // Flush tail for the step (op timings from eval callback)
            if (cb.last_ts_us != 0) {
                const int64_t ts = now_us();
                const int64_t delta = ts - cb.last_ts_us;
                if (!cb.last_op.empty()) {
                    run.decode.total_us += (double)delta;
                    run.decode.per_op_us[cb.last_op] += (double)delta;
                    run.decode.per_sublayer_us[cb.last_sublayer] += (double)delta;
                }
            }

            run.decode.per_step_us.push_back((double)(t_step1 - t_step0));

            if (rc != 0) {
                std::fprintf(stderr, "{\"error\":\"decode failed (step)\",\"step\":%d,\"rc\":%d}\n", i, rc);
                break;
            }
            n_past += 1;
        }
        const int64_t t_dec1 = now_us();
        run.phases_us["decode"] = (double)(t_dec1 - t_dec0);
    }

    // detokenize (just measure piece conversion cost)
    {
        const int64_t t_d0 = now_us();
        char buf[256];
        for (llama_token tok : generated) {
            (void) llama_token_to_piece(model, tok, buf, (int32_t)sizeof(buf), true);
            run.detokenize_calls += 1;
        }
        const int64_t t_d1 = now_us();
        run.phases_us["detokenize"] = (double)(t_d1 - t_d0);
        run.detokenize_us = (double)(t_d1 - t_d0);
    }

    cb.enabled.store(false, std::memory_order_relaxed);

    // Snapshot model metadata BEFORE teardown (the handle is freed below).
    const int64_t meta_n_layers = (int64_t) llama_n_layer(model);
    const int64_t meta_n_embd   = (int64_t) llama_n_embd(model);
    const int64_t meta_n_params = (int64_t) llama_model_n_params(model);

    // teardown
    const int64_t t_td0 = now_us();
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    const int64_t t_td1 = now_us();
    run.phases_us["teardown"] = (double)(t_td1 - t_td0);

    run.phases_us["sample"] = run.sample_us;

    const int64_t t1 = now_us();
    run.phases_us["total"] = (double)(t1 - t0);

    // Output JSON to stdout
    std::ostringstream out;
    out << "{";
    bool first = true;

    // meta
    {
        if (!first) out << ",";
        first = false;
        json_escape_and_write(out, "meta");
        out << ":{";
        bool mf = true;
        write_kv_str(out, "model_file", args.model_path, mf);
        write_kv_int(out, "file_size_bytes", (int64_t)file_size_bytes(args.model_path), mf);
        write_kv_int(out, "n_threads", args.n_threads, mf);
        write_kv_int(out, "n_prompt", (int64_t)prompt_tokens.size(), mf);
        write_kv_int(out, "n_gen", args.n_gen, mf);
        write_kv_int(out, "seed", args.seed, mf);

        write_kv_int(out, "n_layers", meta_n_layers, mf);
        write_kv_int(out, "n_embd", meta_n_embd, mf);
        write_kv_int(out, "n_params", meta_n_params, mf);
        out << "}";
    }

    // phases_ms
    {
        out << ",";
        json_escape_and_write(out, "phases_ms");
        out << ":{";
        bool pf = true;
        for (const auto & [k, v] : run.phases_us) {
            write_kv_num(out, k, us_to_ms((int64_t)v), pf);
        }
        out << "}";
    }

    // memory_mb
    {
        out << ",";
        json_escape_and_write(out, "memory_mb");
        out << ":";
        write_map_mb(out, run.mem_mb);
    }

    // prefill
    {
        out << ",";
        json_escape_and_write(out, "prefill");
        out << ":{";
        bool pf = true;
        write_kv_int(out, "n_tokens", run.prefill.n_tokens, pf);
        write_kv_num(out, "total_ms", us_to_ms((int64_t)run.phases_us["prefill"]), pf);
        const double tps = run.prefill.n_tokens > 0 && run.phases_us["prefill"] > 0
            ? (run.prefill.n_tokens / (run.phases_us["prefill"] / 1e6))
            : 0.0;
        write_kv_num(out, "tokens_per_sec", tps, pf);

        out << ",";
        json_escape_and_write(out, "per_op_ms");
        out << ":";
        write_map_num(out, run.prefill.per_op_us);

        out << ",";
        json_escape_and_write(out, "per_sublayer_ms");
        out << ":";
        write_map_num(out, run.prefill.per_sublayer_us);

        out << "}";
    }

    // decode
    {
        out << ",";
        json_escape_and_write(out, "decode");
        out << ":{";
        bool df = true;
        write_kv_int(out, "n_tokens", (int64_t)generated.size(), df);
        write_kv_num(out, "total_ms", us_to_ms((int64_t)run.phases_us["decode"]), df);
        const double tps = generated.size() > 0 && run.phases_us["decode"] > 0
            ? (generated.size() / (run.phases_us["decode"] / 1e6))
            : 0.0;
        write_kv_num(out, "tokens_per_sec", tps, df);
        if (!run.decode.per_step_us.empty()) {
            write_kv_num(out, "first_token_ms", us_to_ms((int64_t)run.decode.per_step_us.front()), df);
        }

        out << ",";
        json_escape_and_write(out, "per_step_ms");
        out << ":[";
        for (size_t i = 0; i < run.decode.per_step_us.size(); i++) {
            if (i) out << ",";
            out << us_to_ms((int64_t)run.decode.per_step_us[i]);
        }
        out << "]";

        out << ",";
        json_escape_and_write(out, "per_op_ms");
        out << ":";
        write_map_num(out, run.decode.per_op_us);

        out << ",";
        json_escape_and_write(out, "per_sublayer_ms");
        out << ":";
        write_map_num(out, run.decode.per_sublayer_us);

        out << "}";
    }

    // sample
    {
        out << ",";
        json_escape_and_write(out, "sample");
        out << ":{";
        bool sf = true;
        write_kv_int(out, "n_calls", run.sample_calls, sf);
        write_kv_num(out, "total_ms", us_to_ms((int64_t)run.sample_us), sf);
        const double avg_us = run.sample_calls > 0 ? (run.sample_us / run.sample_calls) : 0.0;
        write_kv_num(out, "avg_us", avg_us, sf);
        out << "}";
    }

    if (args.dump_raw_ops) {
        out << ",";
        json_escape_and_write(out, "raw_ops");
        out << ":[";
        for (size_t i = 0; i < cb.raw_ops.size(); i++) {
            if (i) out << ",";
            const auto & ev = cb.raw_ops[i];
            out << "{";
            bool rf = true;
            write_kv_str(out, "phase", ev.phase, rf);
            write_kv_int(out, "step", ev.step, rf);
            write_kv_str(out, "op", ev.op, rf);
            write_kv_str(out, "name", ev.name, rf);
            write_kv_int(out, "layer", ev.layer, rf);
            write_kv_str(out, "sublayer", ev.sublayer, rf);
            write_kv_int(out, "time_us", ev.time_us, rf);
            out << "}";
        }
        out << "]";
    }

    out << "}";

    std::printf("%s\n", out.str().c_str());
    return 0;
}

