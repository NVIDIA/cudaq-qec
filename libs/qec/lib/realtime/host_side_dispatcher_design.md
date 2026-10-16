# Host-Side Spin-Polling Dispatcher with Dynamic Worker Pool

## Design Specification

**Component**: `cudaq-qec` Realtime Decoding Subsystem
**Status**: Implemented
**Supersedes**: Device-side persistent kernel dispatcher (`dispatch_kernel_with_graph`) and Statically-mapped Host Dispatcher
**Target Platforms**: NVIDIA Grace Hopper (GH200), Grace Blackwell (GB200)
**Shared-Memory Model**: libcu++ `cuda::std::atomic` with `thread_scope_system`
**Last Updated**: 2026-04-01

---

## 1. System Context & Motivation

### 1.1 The Pipeline
The system performs real-time quantum error correction (QEC). An FPGA streams syndrome measurements into a host-device shared ring buffer continuously (~104 µs cadence for d=13, T=104). 
1. **Predecoding (GPU)**: TensorRT neural network inference (~88 µs pure GPU compute for d=13/T=104 with FP16; ~146 µs p50 in pipeline with DMA and dispatch overhead).
2. **Global Decoding (CPU)**: PyMatching (MWPM) (~224 µs average for d=13/T=104 with full 17,472-detector parity check matrix).

### 1.2 The Problem
The legacy architecture used a persistent GPU kernel to launch child CUDA graphs using `cudaStreamGraphFireAndForget`. This hit a hardcoded CUDA runtime limit of 128 cumulative launches, causing fatal crashes. A naive host-side port mapping FPGA slots 1:1 to GPU streams caused **Head-of-Line (HOL) blocking**: a single slow PyMatching decode would stall the sequential dispatcher, backing up the ring buffer and violating strict quantum coherence latency budgets.

### 1.3 The Solution
This document defines a **Host-Side Dispatcher with a Dynamic Worker Pool**. 
* The dispatcher runs on a dedicated CPU core.
* Predecoder streams and CPU workers act as an interchangeable pool.
* Inflight jobs are tagged with their origin slot, allowing out-of-order execution and completion.
* Synchronization relies exclusively on Grace Blackwell's NVLink-C2C hardware using libcu++ system-scope atomics.
* **Decoupled architecture**: PyMatching decode runs in a separate thread pool from the predecoder workers, allowing GPU streams to be released immediately after inference completion rather than blocking on CPU decode.

---

## 2. Core Architecture: Dynamic Worker Pool

Instead of mapping predecoder streams statically to incoming data, the host dispatcher maintains a bitmask of available workers (`idle_mask`).

1. **Allocate**: When `rx_flags[slot]` indicates new data, the dispatcher finds the first available worker stream using a hardware bit-scan (`__builtin_ffsll`).
2. **Tag**: The dispatcher records the original `slot` in a tracking array (`inflight_slot_tags[worker_id]`) so the response can be routed correctly.
3. **Pre-launch DMA**: If a `pre_launch_fn` callback is registered on the worker, the dispatcher calls it to issue a `cudaMemcpyAsync` (DMA engine copy) of the input payload from the ring buffer to the TRT input buffer before graph launch.
4. **Dispatch**: The dispatcher launches the CUDA graph on the assigned worker's stream and clears its availability bit.
5. **Free**: The predecoder worker thread (not the PyMatching thread) restores the worker's availability bit in the `idle_mask` after copying inference output and enqueuing the PyMatching job. Slot completion is deferred to the PyMatching thread pool.

---

## 3. Memory & Synchronization Model

**CRITICAL DIRECTIVE**: The ARM Neoverse architecture (Grace) is **weakly ordered**. Code generated from this document MUST NOT use `volatile`, `__threadfence_system()`, or `std::atomic_thread_fence`. 

All shared state must use **libcu++ system-scope atomics** allocated in mapped pinned memory (`cudaHostAllocMapped`).

### 3.1 Shared State Variables

| Variable | Type | Memory Location | Purpose |
| :--- | :--- | :--- | :--- |
| `rx_flags[NUM_SLOTS]` | `atomic<uint64_t, thread_scope_system>` | Mapped Pinned | FPGA writes data ptr; CPU polls (Acquire). |
| `tx_flags[NUM_SLOTS]` | `atomic<uint64_t, thread_scope_system>` | Mapped Pinned | Dispatcher writes IN_FLIGHT; PyMatch writes response; Consumer polls (Release/Acquire). Not used for producer backpressure. |
| `ready_flags[1]` | `atomic<int, thread_scope_system>` | Mapped Pinned | GPU signals TRT done; CPU polls (Release/Acquire). Queue depth = 1. |
| `idle_mask` | `atomic<uint64_t, thread_scope_system>` | Host CPU Mem | Bitmask of free workers. 1 = free, 0 = busy. |
| `inflight_slot_tags[NUM_WORKERS]`| `int` (Plain array) | Host CPU Mem | Maps `worker_id` -> original FPGA `slot`. |
| `mailbox_bank[NUM_WORKERS]` | `void*` (Plain array) | Mapped Pinned | Dispatcher writes device ptr for pre-launch callback. |
| `h_ring_ptrs[1]` | `void*` (Mapped Pinned) | Mapped Pinned | Pre-launch callback writes slot device ptr for CPU worker readback. |
| `h_predecoder_outputs_[1]` | `void*` (Mapped Pinned) | Mapped Pinned | GPU output copied here via DMA; CPU worker reads inference results. |

**NUM_SLOTS**: 16 (ring buffer capacity).
**NUM_WORKERS**: 8 (predecoder streams, each with a dedicated CPU poller thread).
**Queue depth**: 1 per predecoder (single in-flight inference per stream).

---

## 4. Host Dispatcher Thread (Producer)

The dispatcher loop is a tight spin-polling loop running on a dedicated CPU core. It is implemented in `realtime/lib/daemon/dispatcher/host_dispatcher.cu` as `cudaq_host_dispatcher_loop()`.

### 4.1 cudaq_host_dispatch_worker_t Structure

Each worker in the pool has the following fields:

```cpp
typedef struct {
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    uint32_t function_id;
    void (*pre_launch_fn)(void* user_data, void* slot_dev, cudaStream_t stream);
    void* pre_launch_data;
    void (*post_launch_fn)(void* user_data, void* slot_dev, cudaStream_t stream);
    void* post_launch_data;
} cudaq_host_dispatch_worker_t;
```

The `pre_launch_fn` callback enables the dispatcher to issue a `cudaMemcpyAsync` (using the DMA copy engine) for the input payload before each graph launch, without baking application-specific logic into the generic dispatcher. The `post_launch_fn` callback is used in GPU-only mode to enqueue a `cudaLaunchHostFunc` that signals slot completion without CPU worker threads.

### 4.2 Dispatcher Config Structure

The dispatcher loop takes a `cudaq_host_dispatch_loop_ctx_t` that composes three public API structs plus runtime state:

```cpp
typedef struct {
    cudaq_ringbuffer_t ringbuffer;         // ring buffer pointers & strides
    cudaq_dispatcher_config_t config;      // num_slots, slot_size, dispatch_path
    cudaq_function_table_t function_table; // entries pointer + count

    cudaq_host_dispatch_worker_t *workers;
    size_t num_workers;
    void **h_mailbox_bank;
    void *shutdown_flag;       // opaque cuda::std::atomic<int>*
    uint64_t *stats_counter;
    void *live_dispatched;     // opaque cuda::std::atomic<uint64_t>*
    void *idle_mask;           // opaque cuda::std::atomic<uint64_t>*, 1=free 0=busy
    int *inflight_slot_tags;   // worker_id -> origin FPGA slot
    void *io_ctxs_host;       // NULL for legacy mode
    void *io_ctxs_dev;        // NULL for legacy mode
} cudaq_host_dispatch_loop_ctx_t;
```

### 4.3 Function-Aware Worker Selection

When a function table is present, the dispatcher routes each request to a worker whose `function_id` matches the RPC header's `function_id`. This is correct for mixed function tables (e.g., different model types). For pools of interchangeable workers running the same model, all workers **must** share a single `function_id` to avoid head-of-line blocking — otherwise the dispatcher waits for the one specific worker that matches even when other workers are idle.

### 4.4 Dispatcher Logic (Pseudocode)
```cpp
void cudaq_host_dispatcher_loop(const cudaq_host_dispatch_loop_ctx_t *ctx) {
    size_t current_slot = 0;

    while (ctx->shutdown_flag->load(acquire) == 0) {
        uint64_t rx_value = ctx->ringbuffer.rx_flags_host[current_slot].load(acquire);
        if (rx_value == 0) { CUDAQ_REALTIME_CPU_RELAX(); continue; }

        void* slot_host = reinterpret_cast<void*>(rx_value);

        // Parse RPC header and lookup function table
        if (use_function_table) {
            ParsedSlot parsed = parse_slot_with_function_table(slot_host, ctx);
            if (parsed.drop) { clear_and_advance(); continue; }
        }

        // Wait for an available worker with matching function_id
        int worker_id = acquire_graph_worker(ctx, ...);
        if (worker_id < 0) { CUDAQ_REALTIME_CPU_RELAX(); continue; }

        // Mark worker busy, tag with origin slot
        ctx->idle_mask->fetch_and(~(1ULL << worker_id), release);
        ctx->inflight_slot_tags[worker_id] = current_slot;

        // Translate host ptr to device ptr, write to mailbox
        ptrdiff_t offset = (uint8_t*)slot_host - ctx->ringbuffer.rx_data_host;
        void* data_dev = ctx->ringbuffer.rx_data + offset;
        ctx->h_mailbox_bank[worker_id] = data_dev;
        __sync_synchronize();

        // Pre-launch callback: DMA copy input to TRT buffer
        if (worker.pre_launch_fn)
            worker.pre_launch_fn(worker.pre_launch_data, data_dev, worker.stream);

        // Launch graph
        cudaError_t err = cudaGraphLaunch(worker.graph_exec, worker.stream);
        if (err != cudaSuccess) {
            tx_flags_host[current_slot].store(CUDAQ_TX_FLAG_ERROR_TAG<<48|err, release);
            idle_mask->fetch_or(1ULL << worker_id, release);
        } else {
            if (worker.post_launch_fn)
                worker.post_launch_fn(worker.post_launch_data, data_dev, worker.stream);
            // NOTE: tx_flags IN_FLIGHT is still written by the dispatcher for the
            // consumer's benefit (so it can distinguish in-flight from completed slots),
            // but the producer does NOT check tx_flags for backpressure — it uses
            // slot_occupied[] instead (hsb-compatible model).
            tx_flags_host[current_slot].store(CUDAQ_TX_FLAG_IN_FLIGHT, release);
        }

        // Consume slot and advance
        rx_flags_host[current_slot].store(0, release);
        current_slot = (current_slot + 1) % num_slots;
    }
    for (auto& w : ctx->workers) cudaStreamSynchronize(w.stream);
}
```

---

## 5. GPU Graph Composition & Data Transfer

### 5.1 DMA-Based Data Movement

Data copies between the ring buffer and TRT inference buffers use the GPU's DMA copy engine rather than SM-based kernels, freeing compute resources for inference.

**Input copy (ring buffer -> TRT input)**: Issued by the host dispatcher via `pre_launch_fn` callback as a `cudaMemcpyAsync(HostToDevice)` on the worker's stream *before* `cudaGraphLaunch`. The dispatcher passes a device pointer (`slot_dev`) derived from the ring buffer's mapped memory; the callback converts it back to a host pointer using offset arithmetic and issues an H2D copy. Using the host pointer (rather than the mapped device pointer with D2D) enables **multi-GPU support**: pinned host memory is DMA-accessible from any GPU, so predecoders on different devices can all pull input from the same ring buffer. The source address is dynamic (determined at dispatch time from the ring buffer slot at offset `CUDAQ_RPC_HEADER_SIZE` = 24 bytes), so it cannot be baked into the captured graph.

**Output copy (TRT output -> host-mapped outputs)**: Captured inside the CUDA graph as a `cudaMemcpyAsync(DeviceToDevice)`. Both source (`d_trt_output_`) and destination (`d_predecoder_outputs_`) are fixed addresses, so this is captured at graph instantiation time.

### 5.2 Captured CUDA Graph Contents

The CUDA graph for each predecoder contains (in order):

1. **TRT inference** (`context_->enqueueV3(stream)`) -- or `passthrough_copy_kernel` if `SKIP_TRT` is set.
2. **Output DMA copy** (`cudaMemcpyAsync` D2D) -- copies TRT output to host-mapped predecoder output buffer (`h_predecoder_outputs_`).
3. **Signal kernel** (`predecoder_signal_ready_kernel<<<1,1>>>`) -- a single-thread kernel that performs `d_ready_flags[0].store(1, release)` to notify the CPU worker.

The graph is instantiated with `cudaGraphInstantiate(&graph_exec_, graph, 0)` for host-launch mode. Input data arrives exclusively via the pre-launch DMA copy callback; no input-copy kernel exists in the graph.

### 5.3 Source Files

The `ai_predecoder_service.cu` implementation contains only two device kernels:

- `predecoder_signal_ready_kernel` -- single-thread kernel that atomically stores `1` to the ready flag with system-scope release semantics.
- `passthrough_copy_kernel` -- vectorized identity copy (`uint4` 16-byte loads/stores, 256 threads) used when `SKIP_TRT` is set, substituting for TRT inference.

### 5.4 Passthrough Copy Kernel (SKIP_TRT mode)

When `SKIP_TRT` is set, the `passthrough_copy_kernel` substitutes for TRT inference, providing a deterministic identity function for testing and benchmarking the infrastructure overhead. In SKIP_TRT mode, the `ai_decoder_service` constructor sets `input_size_ = output_size_ = 1600 * sizeof(float)` (6400 bytes) without loading any model file.

---

## 6. Decoupled Worker Architecture

The CPU-side processing uses a **two-tier decoupled architecture** that separates GPU polling from CPU-intensive decode:

### 6.1 Tier 1: Predecoder Workers (GPU Polling + Copy)

Each predecoder has a dedicated worker thread in the `realtime_pipeline`. These threads:

1. **Poll** `ready_flags[0]` via `compare_exchange_strong(1, 2, acquire, relaxed)` (CAS claiming).
2. **Copy** inference output from `h_predecoder_outputs_` to a per-slot buffer (`deferred_outputs[origin_slot]`).
3. **Compute** syndrome density metrics (input vs. output nonzero detector counts).
4. **Release** the GPU predecoder slot via `release_job(slot_idx)` → `ready_flags[0].store(0, release)`.
5. **Enqueue** a `PyMatchJob` to the `PyMatchQueue`.
6. **Return** `DEFERRED_COMPLETION` to the pipeline, which releases `idle_mask` but does NOT set `tx_flags`.

### 6.2 Tier 2: PyMatching Workers (CPU Decode + Completion)

A separate thread pool (16 workers for d13_r104) dequeues from `PyMatchQueue` and:

1. **Decode** using a per-thread PyMatching decoder instance (lock-free `thread_local` acquisition).
2. **Write** the RPC response (`DecodeResponse`) directly into the ring buffer slot.
3. **Signal** slot completion via `pipeline.complete_deferred(origin_slot)`, which stores the slot host address into `tx_flags[origin_slot]`.

### 6.3 Why Decouple?

With the coupled architecture, a single slow PyMatching decode (up to 139 ms tail latency) would hold the predecoder worker busy, preventing the GPU stream from being dispatched new work. This caused:
- Severe head-of-line blocking on `idle_mask`
- ~41M backpressure stalls
- Tail latencies: p90 = 970 µs, p99 = 1,767 µs

The decoupled architecture reduces predecoder worker hold time from ~214 µs to ~10 µs, dropping:
- Backpressure stalls from 41M to 6.2M (85% reduction)
- p90 from 970 µs to 515 µs (47% reduction)
- p99 from 1,767 µs to 1,249 µs (29% reduction)

### 6.4 DEFERRED_COMPLETION Protocol

```
Pipeline Worker Thread:         PyMatching Thread:
  poll_next_job() → CAS 1→2      (blocked on queue)
  copy output to deferred_buf          |
  release_job() → store 0             |
  enqueue PyMatchJob ──────────►  pop PyMatchJob
  return DEFERRED_COMPLETION      decode with PyMatching
  pipeline sets idle_mask ✓       write RPC response
  pipeline skips tx_flags ✗       complete_deferred(slot)
                                   └──► tx_flags[slot].store(addr)
```

### 6.5 PyMatchQueue

Thread-safe MPSC queue using `std::mutex` + `std::condition_variable`:

```cpp
struct PyMatchJob {
    int origin_slot;
    uint64_t request_id;
    void *ring_buffer_ptr;
};

class PyMatchQueue {
    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<PyMatchJob> jobs_;
    bool stop_ = false;
public:
    void push(PyMatchJob &&j);
    bool pop(PyMatchJob &out);  // blocks until job available or shutdown
    void shutdown();
};
```

### 6.6 Ready-Flag State Machine (Atomic Claiming)

With queue depth 1, the poller must **claim** each completion exactly once.

**States** (per-worker ready flag):

| Value | State      | Meaning |
| :---  | :---       | :---    |
| 0     | Idle       | Waiting for GPU, or worker has called `release_job`. |
| 1     | Ready      | GPU finished; signal kernel stored 1. |
| 2     | Processing | CPU poller claimed the job; copying output. |

**Poller**: Use `compare_exchange_strong(expected=1, desired=2, memory_order_acquire, memory_order_relaxed)`. Only the thread that wins the CAS proceeds. Use **relaxed on failure** so spin-polling does not add barriers that delay seeing the GPU's store(1).

**Worker**: When output is copied and job is enqueued, call `release_job(slot_idx)` which does `ready_flags[0].store(0, release)` so the slot is Idle for the next launch.

---

## 7. Out-of-Order Consumer

The consumer thread harvests completions **out-of-order** by scanning all active slots on every iteration, rather than waiting for a sequential `next_harvest` counter. This eliminates head-of-line blocking where a slow request in slot N would prevent harvesting faster completions in slot N+1.

### 7.1 Consumer Logic (Pseudocode)
```cpp
while (!consumer_stop) {
    bool found_any = false;
    for (uint32_t s = 0; s < NUM_SLOTS; ++s) {
        if (!slot_occupied[s]) continue;

        cudaq_tx_status_t status = ring->poll_tx(s, &err);

        if (status == CUDAQ_TX_READY) {
            int rid = slot_request[s];
            complete_ts[rid] = now();
            completed[rid] = true;
            total_completed++;

            slot_occupied[s] = 0;      // Reset occupancy FIRST
            __sync_synchronize();       // ARM memory fence
            ring->clear_slot(s);        // Then clear rx/tx flags
            found_any = true;
        }
    }
    if (!found_any) CUDAQ_REALTIME_CPU_RELAX();
}
```

The consumer uses `slot_occupied[]` (set by the producer, cleared by the consumer) to know which slots are active, and `poll_tx()` to distinguish in-flight from completed results.

### 7.2 Consumer-Producer Race Fix

On ARM's weakly ordered memory model, the consumer must reset `slot_occupied[s] = 0` **before** clearing the ring buffer flags (via `clear_slot`), with a `__sync_synchronize()` fence between them. Without this ordering:
1. Consumer clears flags (slot appears free to producer)
2. Producer writes new `slot_occupied[s] = 1` 
3. Consumer's delayed `slot_occupied[s] = 0` clobbers the producer's write

This race caused exactly one request to get "stuck" indefinitely, eventually stalling the entire pipeline through backpressure.

---

## 8. realtime_pipeline Scaffolding

The low-level dispatcher, consumer, and worker threads are wrapped by a higher-level `realtime_pipeline` class (`libs/qec/include/cudaq/qec/realtime/pipeline.h`) that hides all ring buffer management, atomics, and thread lifecycle. Application code provides three callbacks:

1. **GPU stage factory** (`gpu_stage_factory`): Called once per worker during `start()`. Returns the `cudaGraphExec_t`, `cudaStream_t`, `pre_launch_fn`, `post_launch_fn`, `function_id`, and an opaque `user_context` for each worker.
2. **CPU stage callback** (`cpu_stage_callback`): Called by each worker thread when GPU inference completes. Receives `cpu_stage_context` containing `gpu_output`, `gpu_output_size`, `response_buffer`, and the `user_context`. Returns the number of bytes written, `0` if no result ready (poll again), or `DEFERRED_COMPLETION` to release the worker without signaling slot completion.
3. **Completion callback** (`completion_callback`): Called by the consumer thread for each completed (or errored) request with a `completion` struct.

```cpp
realtime_pipeline pipeline(config);
pipeline.set_gpu_stage([&](int worker_id) -> gpu_worker_resources { ... });
pipeline.set_cpu_stage([&](const cpu_stage_context& ctx) -> size_t { ... });
pipeline.set_completion_handler([&](const completion& c) { ... });
auto injector = pipeline.create_injector();
pipeline.start();
injector.submit(function_id, payload, payload_size, request_id);
// ...
pipeline.stop();
```

### 8.1 DEFERRED_COMPLETION

When the CPU stage callback returns `DEFERRED_COMPLETION` (= `SIZE_MAX`), the pipeline:
- Sets the worker's bit in `idle_mask` (worker is free for next dispatch)
- Does NOT write to `tx_flags[origin_slot]` (slot stays IN_FLIGHT)

The caller is responsible for eventually calling `pipeline.complete_deferred(slot)`, which stores the slot host address into `tx_flags[slot]` with release semantics, making the completion visible to the consumer.

### 8.2 GPU-Only Mode

If no `cpu_stage_callback` is registered, the pipeline operates in **GPU-only mode**: no CPU worker threads are spawned. Instead, the dispatcher's `post_launch_fn` enqueues a `cudaLaunchHostFunc` on each worker stream. When the GPU finishes, the CUDA runtime calls the host function, which stores into `tx_flags` and restores the `idle_mask` bit — all from the CUDA callback thread.

### 8.3 ring_buffer_injector

The `ring_buffer_injector` class (created via `pipeline.create_injector()`) encapsulates the host-side submission logic for testing without FPGA hardware. It provides:

- `try_submit()`: Non-blocking, returns false on backpressure. Checks `slot_occupied[slot]` to determine if the next slot is free — this mirrors the hsb model where the FPGA checks `ring_flag` (mapped to `rx_flags`) rather than a separate tx_flags sentinel.
- `submit()`: Blocking spin-wait until a slot becomes available.
- `backpressure_stalls()`: Counter of spin iterations during backpressure.

The injector uses a round-robin slot selection with atomic CAS for thread safety.

The `pipeline_stage_config` allows configuring `num_workers`, `num_slots`, `slot_size`, and optional `core_pinning` for dispatcher, consumer, and worker threads.

---

## 9. Step-by-Step Data Flow Trace

1. **Producer** writes uint8 measurements into `payload_buf` from Stim test data.
2. **Producer** calls `injector.submit(fid, payload, size, request_id)`.
3. **ring_buffer_injector** writes RPC header (`RPCHeader`: magic, function_id, arg_len, request_id, ptp_timestamp = 24 bytes) + payload into `rx_data[slot]`.
4. **ring_buffer_injector** sets `rx_flags[slot] = host_ptr` (release).
5. **Host Dispatcher** reads `rx_flags[slot]`, sees data.
6. **Host Dispatcher** parses RPC header, looks up function in the function table.
7. **Host Dispatcher** scans `idle_mask`, finds `worker_id = 2` is free.
8. **Host Dispatcher** marks bit 2 busy in `idle_mask`.
9. **Host Dispatcher** saves `inflight_slot_tags[2] = slot`.
10. **Host Dispatcher** translates `host_ptr` to `dev_ptr`, writes to `mailbox_bank[2]`.
11. **Host Dispatcher** calls `pre_launch_fn`: writes `h_ring_ptrs[0] = dev_ptr`, derives host pointer via offset arithmetic, issues `cudaMemcpyAsync(d_trt_input, host_ptr + 24, input_size, H2D, stream[2])`.
12. **Host Dispatcher** calls `cudaGraphLaunch(..., stream[2])`.
13. **Host Dispatcher** sets `tx_flags[slot] = IN_FLIGHT` (for consumer's benefit), then clears `rx_flags[slot] = 0` and advances to next slot. The producer does not check `tx_flags`; backpressure is via `slot_occupied[]`.
14. **GPU DMA engine** copies input payload from ring buffer to TRT input buffer.
15. **GPU** executes TRT inference (or passthrough copy in SKIP_TRT mode).
16. **GPU DMA engine** copies TRT output to host-mapped `h_predecoder_outputs_`.
17. **GPU signal kernel** sets `ready_flags[0] = 1` (system-scope atomic release).
18. **Predecoder Worker** CAS(1, 2) on `ready_flags[0]`, wins, reads inference output.
19. **Predecoder Worker** copies output to `deferred_outputs[origin_slot]`.
20. **Predecoder Worker** computes syndrome density metrics.
21. **Predecoder Worker** calls `release_job(0)` → `ready_flags[0].store(0, release)`.
22. **Predecoder Worker** extracts `request_id` from RPC header, enqueues `PyMatchJob`.
23. **Predecoder Worker** returns `DEFERRED_COMPLETION`.
24. **Pipeline** restores bit 2 in `idle_mask` (worker free for next dispatch). Does NOT touch `tx_flags`.
25. **PyMatching Worker** pops `PyMatchJob` from queue, acquires per-thread decoder.
26. **PyMatching Worker** runs PyMatching MWPM decode over full parity check matrix.
27. **PyMatching Worker** writes `RPCResponse + DecodeResponse` into ring buffer slot.
28. **PyMatching Worker** calls `pipeline.complete_deferred(slot)` → `tx_flags[slot].store(host_addr, release)`.
29. **Consumer** scans all slots where `slot_occupied[s]` is set, calls `poll_tx(s)` to check if result is ready (i.e., `tx_flags[slot]` is a valid address, not IN_FLIGHT).
30. **Consumer** calls `completion_handler(request_id, slot, success)`.
31. **Consumer** sets `slot_occupied[slot] = 0`, `__sync_synchronize()`, then calls `clear_slot(slot)` (clears rx/tx flags). Producer may now reuse slot.

---

## 10. RPC Protocol & Ring Buffer

### 10.1 RPC Header

```cpp
struct RPCHeader {
    uint32_t magic;        // RPC_MAGIC_REQUEST
    uint32_t function_id;  // FNV-1a hash of function name
    uint32_t arg_len;      // payload length in bytes
    uint32_t request_id;   // unique request identifier
    uint64_t ptp_timestamp; // PTP timestamp (optional)
};
// sizeof(RPCHeader) == 24
#define CUDAQ_RPC_HEADER_SIZE 24u
```

### 10.2 Backpressure Model (HSB-Compatible)

The producer does **not** check `tx_flags` for backpressure. Instead, it checks a host-side `slot_occupied[]` byte vector — set to 1 by the producer on submission, cleared to 0 by the consumer after harvesting. This mirrors the hsb FPGA model where `ring_flag` (mapped to `rx_flags`) is the sole backpressure mechanism between sender and receiver.

The dispatcher still writes `tx_flags[slot] = CUDAQ_TX_FLAG_IN_FLIGHT` after a successful `cudaGraphLaunch` and before clearing `rx_flags[slot]`. This is used internally by the consumer to distinguish in-flight slots from completed results, but it is **not** part of the producer's backpressure contract.

1. **Dispatcher**: On successful launch, writes `tx_flags[current_slot] = IN_FLIGHT` then clears `rx_flags`. On failure, writes `CUDAQ_TX_FLAG_ERROR_TAG<<48 | err`.
2. **Producer**: Reuses a slot only when `slot_occupied[slot] == 0`. Does not read `tx_flags`.
3. **Consumer**: Scans slots where `slot_occupied[s]` is set, polls `tx_flags` to distinguish IN_FLIGHT from completed, and harvests completed results.

**Slot lifecycle**: Idle (`slot_occupied=0`) → Written (`slot_occupied=1`, `rx_flags=ptr`) → Dispatched (`rx_flags=0`, GPU processing) → Completed (`tx_flags=response addr`) → Consumer harvests, `slot_occupied=0`, flags cleared → Idle.

---

## 11. Dynamic Batch Handling for ONNX Models

When building a TensorRT engine from an ONNX model with dynamic batch dimensions (dim 0 <= 0), `ai_decoder_service.cu` automatically creates an optimization profile that pins all dynamic dimensions to 1. This enables building engines from models like `predecoder_memory_d13_T13_X.onnx` which use a symbolic `batch` dimension.

---

## 12. Test Suite

A GTest-based test suite (`libs/qec/unittests/test_realtime_pipeline.cu`) validates the pipeline using `SKIP_TRT` passthrough mode (no TensorRT dependency at runtime). The tests are organized into three categories:

### 12.1 Unit Tests (8 tests)
- **ai_decoder_service**: Verify SKIP_TRT buffer sizes (1600 floats = 6400 bytes), allocation, and graph capture.
- **ai_predecoder_service**: Verify mapped pinned memory allocation, `poll_next_job` / `release_job` state machine, and host-launchable graph.

### 12.2 Correctness Tests (5 tests)
Data-integrity tests that verify known payloads survive the full CUDA graph round-trip bitwise-identical (memcmp, not epsilon):
- **Zeros, Known Pattern, Random Data, Extreme Float Values**: Single-request verification with different payload patterns (including `FLT_MAX`, `NaN`, `INFINITY`).
- **Multiple Requests (5,000 iterations)**: Pushes 5,000 random 6.4 KB payloads through the pipeline and verifies bitwise identity on every one. Confirms no cross-contamination or data corruption over sustained use.

### 12.3 Integration Tests (8 tests)
- **Dispatcher lifecycle**: Shutdown semantics, stats counter accuracy, invalid RPC magic rejection, slot wraparound.
- **Single Request Round-Trip**: Full dispatcher -> graph -> poll -> verify data path.
- **Multi-Predecoder Concurrency**: 4 predecoders on 4 streams, simultaneous dispatch, per-predecoder data verification.
- **Sustained Throughput (200 requests)**: Regression test for the 128-launch-limit fix. Proves indefinite stability of the host-side dispatcher.

### 12.4 End-to-End Benchmark (test_realtime_predecoder_w_pymatching)
- Configurable surface code distance and round count: d7, d13, d13_r104, d21, d31.
- Loads Stim-generated test data (detectors, observables, parity check matrix, priors).
- Streams syndrome data at configurable rate with correctness verification (LER).
- Reports latency percentiles, throughput, backpressure stalls, syndrome density reduction.
- `--num-gpus N` is accepted but currently clamped to 1 (multi-GPU dispatch is not yet supported; see Section 15).

---

## 13. Shutdown and Grace Period

- **Grace period**: After the producer stops submitting, the pipeline waits up to 5 seconds for `total_completed >= total_submitted`.
- **Consumer exit**: The consumer thread normally exits when `producer_stop && total_completed >= total_submitted`. To avoid hanging forever if some in-flight requests never complete, set a **consumer_stop** flag after the grace period; the consumer loop checks this and exits so `consumer.join()` returns and the process can print the final report and exit cleanly.
- **Dispatcher shutdown**: Set `shutdown_flag = 1` after the consumer exits, then join the dispatcher thread. The dispatcher synchronizes all worker streams before returning.
- **PyMatching thread pool**: Call `pymatch_queue.shutdown()` to unblock all waiting threads, then join all PyMatching worker threads.

---

## 14. Performance Results (d=13, T=104, 104 µs rate, 20s)

### 14.1 Single-GPU (current, shared function_id + H2D copy)

Measured on Grace Blackwell (GB200) with `predecoder_memory_d13_T104_X.onnx` (FP16), 8 predecoders on 1 GPU, 16 PyMatching workers, 16 slots:

| Metric | Value |
| :--- | :--- |
| Throughput | 9,610 req/s |
| Submitted / Completed | 192,309 / 192,309 |
| Backpressure stalls | 4,821,740 |
| p50 latency | 334.9 µs |
| Mean latency | 373.8 µs |
| p90 latency | 492.8 µs |
| p95 latency | 572.8 µs |
| p99 latency | 1,213.2 µs |
| Max latency | 3,757.4 µs |
| PyMatching decode (avg) | 221.3 µs |
| Syndrome density reduction | 98.3% |
| Pipeline LER | 0.0020 |

### 14.2 Multi-GPU

Multi-GPU dispatch is **not currently supported**. The host dispatcher thread does not call `cudaSetDevice()` before `cudaGraphLaunch()` or `cudaStreamQuery()`, causing hangs when workers span multiple devices. The `--num-gpus` flag is accepted but clamped to 1 with a warning. See Section 15 for details.

### 14.3 Raw TRT Baseline (trtexec)

| Mode | GPU Compute | Total Host Latency |
| :--- | :--- | :--- |
| Default | 107 µs | 119 µs |
| CUDA Graph + SpinWait | 90 µs | 99 µs |
| CUDA Graph + SpinWait + No Transfers | 88 µs | 88 µs |

---

## 15. Multi-GPU Support (Not Yet Implemented)

Multi-GPU dispatch is **disabled** pending a fix to the host dispatcher. The `--num-gpus` flag is accepted but clamped to 1.

### 15.1 Known Issue

The host dispatcher thread (`cudaq_host_dispatcher_loop`) does not call `cudaSetDevice()` before `cudaGraphLaunch()` or `cudaStreamQuery()`. CUDA streams and graphs are bound to the device on which they were created, but `cudaGraphLaunch` dispatches to the **calling thread's current device**, not the stream's device. When the dispatcher thread's current device is GPU 0 but it launches a graph captured on GPU 1 with a stream created on GPU 1, the call hangs or silently fails.

### 15.2 Fix Required

To support multi-GPU, each `cudaq_host_dispatch_worker_t` needs a `gpu_id` field, and the dispatcher must call `cudaSetDevice(worker.gpu_id)` before `cudaGraphLaunch` and `cudaStreamQuery`.

---

## 16. LLM Implementation Directives (Constraints Checklist)

When generating code from this specification, the LLM **MUST** strictly adhere to the following constraints:

- [ ] **NO CUDA STREAM QUERYING**: Do not use `cudaStreamQuery()` for backpressure or completion checking. It incurs severe driver latency. Rely strictly on `idle_mask` and `ready_flags`.
- [ ] **NO WEAK ORDERING BUGS**: Do not use `volatile`. Do not use `__threadfence_system()`. You must use `cuda::std::atomic<T, cuda::thread_scope_system>` (or `<cuda/atomic>` with `thread_scope_system`) for all cross-device synchronization.
- [ ] **NO HEAD OF LINE BLOCKING**: The host dispatcher MUST NOT statically map slots to predecoders. It must dynamically allocate via `idle_mask`. The consumer MUST harvest out-of-order by scanning all active slots. When all workers run the same model, they MUST share a single `function_id` so the dispatcher can pick any idle worker; per-worker unique IDs cause function-aware routing to degenerate into 1:1 static mapping.
- [ ] **NO DATA LOSS**: If `idle_mask == 0` (all workers busy), the dispatcher MUST spin on the current slot (`CUDAQ_REALTIME_CPU_RELAX()`). It MUST NOT advance `current_slot` until a worker is allocated and the graph is launched.
- [ ] **NO RACE CONDITIONS ON TAGS**: `inflight_slot_tags` does not need to be atomic because index `[worker_id]` is exclusively owned by the active flow once the dispatcher clears the bit in `idle_mask`, until the worker thread restores the bit.
- [ ] **READY FLAG CLAIMING**: The CPU poller MUST claim each completion exactly once using compare_exchange_strong(1, 2) on the ready flag; use relaxed memory order on CAS failure. The worker MUST clear the flag (store 0) in `release_job`.
- [ ] **BACKPRESSURE MODEL**: The producer (ring_buffer_injector) MUST check `slot_occupied[slot]` for backpressure, NOT `tx_flags`. This mirrors the hsb model where `ring_flag`/`rx_flags` is the sole sender-receiver backpressure mechanism. The dispatcher still writes `tx_flags = IN_FLIGHT` for the consumer's benefit, but the producer does not read `tx_flags`. The consumer uses `poll_tx()` to distinguish in-flight from completed results.
- [ ] **CONSUMER MEMORY ORDERING**: The consumer MUST set `slot_occupied[s] = 0` BEFORE calling `cudaq_host_ringbuffer_clear_slot`, with a `__sync_synchronize()` fence between them, to prevent the producer-consumer race on ARM.
- [ ] **DMA DATA MOVEMENT**: Use `cudaMemcpyAsync` (DMA engine) for data copies. Input copy is issued via `pre_launch_fn` callback before graph launch at offset `CUDAQ_RPC_HEADER_SIZE` (24 bytes) using `cudaMemcpyHostToDevice` from the pinned ring buffer host pointer. Output copy is captured inside the graph. Do not use SM-based byte-copy kernels for fixed-address transfers.
- [ ] **NO INPUT KERNEL IN GRAPH**: The captured CUDA graph must NOT contain an input-copy kernel. All input data movement is handled by the `pre_launch_fn` DMA callback issued on the worker stream before `cudaGraphLaunch`.
- [ ] **DEFERRED COMPLETION**: When the CPU stage returns `DEFERRED_COMPLETION`, the pipeline MUST release `idle_mask` but MUST NOT write `tx_flags`. The external caller MUST call `complete_deferred(slot)` to signal completion.
- [ ] **SHUTDOWN**: Use a `consumer_stop` (or equivalent) flag so the consumer thread can exit after a grace period even when `total_completed < total_submitted`; join the consumer after setting the flag so the process exits cleanly. Shut down the PyMatching queue before stopping the pipeline.
