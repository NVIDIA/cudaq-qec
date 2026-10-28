# Host-Side Spin-Polling Dispatcher with Dynamic Worker Pool

## Design Specification

**Component**: `cudaq-qec` Realtime Decoding Subsystem
**Status**: Implemented (dispatcher owned by `libcudaq-realtime-pipeline`)
**Supersedes**: Device-side persistent kernel dispatcher (`dispatch_kernel_with_graph`), Statically-mapped Host Dispatcher, and CUDA-Q realtime's host graph-launch dispatcher (`cudaq_host_dispatcher_loop`, removed in CUDA-Q PR 5736)
**Target Platforms**: NVIDIA Grace Hopper (GH200), Grace Blackwell (GB200)
**Shared-Memory Model**: libcu++ `cuda::std::atomic` with `thread_scope_system`
**Last Updated**: 2026-10-27

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
* The dispatcher is a thread owned by `realtime_pipeline`, running on a dedicated CPU core.
* Predecoder streams and CPU workers act as an interchangeable pool.
* Inflight jobs are tagged with their origin slot, allowing out-of-order execution and completion.
* Synchronization relies exclusively on Grace Blackwell's NVLink-C2C hardware using libcu++ system-scope atomics.
* **Decoupled architecture**: PyMatching decode runs in a separate thread pool from the predecoder workers, allowing GPU streams to be released immediately after inference completion rather than blocking on CPU decode.

---

## 2. Core Architecture: Dynamic Worker Pool

Instead of mapping predecoder streams statically to incoming data, the host dispatcher maintains a bitmask of available workers (`idle_mask`).

1. **Allocate**: When `rx_flags[slot]` indicates new data, the dispatcher finds the first available worker stream using a hardware bit-scan (`__builtin_ffsll`).
2. **Tag**: The dispatcher records the original `slot` in a tracking array (`inflight_slot_tags[worker_id]`) before it clears the worker's availability bit, so the response can be routed correctly.
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
| `tx_flags[NUM_SLOTS]` | `atomic<uint64_t, thread_scope_system>` | Mapped Pinned | 0 = no response; non-zero = host address of the TX slot holding the response, stored (Release) by `complete_deferred`, by the worker thread on a non-deferred completion, or by the GPU-only callback. Internal ring: the consumer polls (Acquire) and clears. External ring: the transport's TX kernel consumes and clears. Not used for producer backpressure. |
| `slot_occupied[NUM_SLOTS]`, `slot_cuda_error[NUM_SLOTS]` | `uint8_t`, `int` (plain arrays) | Host CPU Mem | Pipeline-private per-slot in-flight state. `slot_occupied` is set by the producer and cleared by the consumer (internal ring); `slot_cuda_error` carries a failed `cudaGraphLaunch` from the dispatcher to the completion callback. |
| `ready_flags[1]` | `atomic<int, thread_scope_system>` | Mapped Pinned | GPU signals TRT done; CPU polls (Release/Acquire). Queue depth = 1. |
| `idle_mask` | `atomic<uint64_t, thread_scope_system>` | Host CPU Mem | Bitmask of free workers. 1 = free, 0 = busy. |
| `inflight_slot_tags[NUM_WORKERS]`| `int` (Release store / Acquire load) | Host CPU Mem | Maps `worker_id` -> original FPGA `slot`. Written by the dispatcher before the launch, read by the worker thread. |
| `worker_launch_failed[NUM_WORKERS]` | `uint8_t` (Release store / Acq-Rel exchange) | Host CPU Mem | Dispatcher -> worker thread: a failed `cudaGraphLaunch`; the worker thread returns its own `idle_mask` bit. |
| `mailbox_bank[NUM_WORKERS]` | `void*` (Plain array) | Mapped Pinned | Allocated by the application for the `ai_predecoder_service` constructors; the predecoder graph does not read it (input arrives through `pre_launch_fn`). |
| `h_ring_ptrs[1]` | `void*` (Mapped Pinned) | Mapped Pinned | Pre-launch callback records the slot device ptr (service bookkeeping; the CPU stage reads the request from the host RX slot). |
| `h_predecoder_outputs_[1]` | `void*` (Mapped Pinned) | Mapped Pinned | GPU output copied here via DMA; CPU worker reads inference results. |

**NUM_SLOTS**: `pipeline_stage_config::num_slots`, used verbatim by the pipeline (16 in the PyMatching benchmark; the GpuRoce bridge passes the transceiver's page count).
**NUM_WORKERS**: 8 (predecoder streams, each with a dedicated CPU poller thread).
**Queue depth**: 1 per predecoder (single in-flight inference per stream).

---

## 4. Host Dispatcher Thread (pipeline-owned)

The dispatcher loop is a tight spin-polling loop running on a dedicated CPU core. It is implemented in `libs/qec/lib/realtime/realtime_pipeline.cu` as `realtime_pipeline::Impl::dispatcher_loop()` and started by `realtime_pipeline::start()`. CUDA-Q realtime contributes only the ring layout (`cudaq_ringbuffer_t`), the `cudaq_host_ringbuffer_*` slot helpers exported by `libcudaq-realtime`, the `RPCHeader` / `RPCResponse` wire structs (`dispatch_kernel_launch.h`) and `CUDAQ_REALTIME_CPU_RELAX` (`cpu_relax.h`).

### 4.1 Worker Resources

Each worker in the pool is described by the public `gpu_worker_resources` struct (`pipeline.h`), returned by the application's `gpu_stage_factory` once per worker during `start()`:

```cpp
struct gpu_worker_resources {
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    void (*pre_launch_fn)(void *user_data, void *slot_dev, cudaStream_t stream);
    void *pre_launch_data;
    void (*post_launch_fn)(void *user_data, void *slot_dev, cudaStream_t stream);
    void *post_launch_data;
    uint32_t function_id;
    void *user_context;
};
```

The `pre_launch_fn` callback enables the dispatcher to issue a `cudaMemcpyAsync` (using the DMA copy engine) for the input payload before each graph launch, without baking application-specific logic into the dispatcher. `post_launch_fn` is called by the dispatcher after a successful launch; in GPU-only mode the dispatcher itself then enqueues the `cudaLaunchHostFunc` completion callback (Section 8.2).

Per worker, the pipeline keeps private state: its bit in `idle_mask`, `inflight_slot_tags[w]` (origin slot of the request in flight), `worker_launch_failed[w]` (a failed launch to be acknowledged by the worker thread) and, in GPU-only mode, the host-callback context `{tx_flags_host, idle_mask*, worker_id, origin_slot, tx_value}`.

### 4.2 Dispatcher State

The loop runs over members of `realtime_pipeline::Impl`; nothing crosses a library boundary (abridged):

```cpp
cudaq_ringbuffer_t active_rb_;      // ring views: rx_flags_host, tx_flags_host, rx_data,
                                    // rx_data_host, tx_data_host, rx/tx_stride_sz
bool external_ring_;                // transport-owned ring (Section 7.3)
gpu_worker_resources worker_resources[NUM_WORKERS];
atomic<uint64_t> idle_mask;         // 1 = free, 0 = busy; only the dispatcher clears bits
int      inflight_slot_tags[NUM_WORKERS];   // worker_id -> origin slot (release/acquire)
uint8_t  worker_launch_failed[NUM_WORKERS]; // dispatcher -> worker thread
int      slot_cuda_error[NUM_SLOTS];        // dispatcher -> completion callback
uint8_t  slot_occupied[NUM_SLOTS];          // producer -> consumer (internal ring)
uint32_t slot_request[NUM_SLOTS];           // request_id captured per slot
atomic<uint64_t> live_dispatched;   // launch attempts (Stats::dispatched)
atomic<int> shutdown_flag;          // stop() -> dispatcher
int device_id;                      // cudaGetDevice() on the thread calling start()
```

The dispatcher thread and every worker thread call `cudaSetDevice(device_id)` first, so the factory's streams and graphs must belong to the device that is current on the thread calling `start()`.

### 4.3 Function-Aware Worker Selection

The dispatcher routes each request to a worker whose `function_id` matches the RPC header's `function_id`; a request whose `function_id` no worker serves is dropped (its RX flag is cleared and nothing is counted). This is correct for mixed worker pools (e.g., different model types). For pools of interchangeable workers running the same model, all workers **must** share a single `function_id` to avoid head-of-line blocking — otherwise the dispatcher waits for the one specific worker that matches even when other workers are idle.

### 4.4 Dispatcher Logic (Pseudocode)
```cpp
void realtime_pipeline::Impl::dispatcher_loop() {
    cudaSetDevice(device_id);
    const cudaq_ringbuffer_t &rb = active_rb_;
    size_t cursor = 0;

    while (shutdown_flag.load(acquire) == 0) {
        uint64_t rx_value = atomic_load(&rb.rx_flags_host[cursor], acquire);
        if (rx_value == 0) { CUDAQ_REALTIME_CPU_RELAX(); continue; }

        // Producer convention: a non-zero rx_flag is the host address of the RX slot
        auto *slot_host = reinterpret_cast<uint8_t*>(rx_value);
        auto *hdr = reinterpret_cast<const RPCHeader*>(slot_host);
        if (hdr->magic != RPC_MAGIC_REQUEST) {                 // drop, not counted
            atomic_store(&rb.rx_flags_host[cursor], 0, release); advance(); continue;
        }
        // Copy the header fields before anything can overwrite the slot
        uint32_t fid = hdr->function_id, rid = hdr->request_id;
        uint64_t ptp = hdr->ptp_timestamp;
        if (no worker_resources[i].function_id == fid) {        // drop, not counted
            atomic_store(&rb.rx_flags_host[cursor], 0, release); advance(); continue;
        }

        // First idle worker (ffsll over idle_mask) whose function_id matches
        int w = acquire_worker(fid);
        if (w < 0) { CUDAQ_REALTIME_CPU_RELAX(); continue; }   // backpressure: stay on this slot
        if (external_ring_) slot_request[cursor] = rid;        // internal ring: the injector recorded the id (Section 9, step 3)
        atomic_store(&inflight_slot_tags[w], cursor, release); // tag first ...
        idle_mask.fetch_and(~(1ULL << w), acq_rel);            // ... then take the worker

        // Device view of the RX slot, host view of the TX slot
        void    *slot_dev = rb.rx_data + (slot_host - rb.rx_data_host);
        uint8_t *tx_host  = rb.tx_data_host + cursor * rb.tx_stride_sz;  // == slot_host on the in-place internal ring
        auto &wr = worker_resources[w];

        if (wr.pre_launch_fn) wr.pre_launch_fn(wr.pre_launch_data, slot_dev, wr.stream);
        cudaError_t err = cudaGraphLaunch(wr.graph_exec, wr.stream);
        if (err == cudaSuccess && wr.post_launch_fn) wr.post_launch_fn(wr.post_launch_data, slot_dev, wr.stream);
        live_dispatched.fetch_add(1, relaxed);                   // launch attempts; drops are not counted

        // Consume the RX slot before publishing anything for it from this thread, so a
        // follow-up request in the same slot never loses its rx_flag to this clear. On an
        // external ring the transport may refill the page while the pre-launch H2D copy
        // is still queued on the stream.
        atomic_store(&rb.rx_flags_host[cursor], 0, release);

        if (err != cudaSuccess) {
            (void)cudaGetLastError();
            write RPCResponse{RPC_MAGIC_RESPONSE, status = err, result_len = 0, rid, ptp} at tx_host;
            slot_cuda_error[cursor] = err;
            publish(cursor, tx_host);                            // Section 7.3: tx_flags = TX slot address
            if (gpu_only) idle_mask.fetch_or(1ULL << w, release);             // no callback was enqueued
            else          atomic_store(&worker_launch_failed[w], 1, release); // worker thread frees itself
        } else if (gpu_only) {
            gpu_only_ctxs[w].origin_slot = cursor; gpu_only_ctxs[w].tx_value = tx_host;
            cudaLaunchHostFunc(wr.stream, gpu_only_host_callback, &gpu_only_ctxs[w]);
            // on enqueue failure: cudaStreamSynchronize(wr.stream); gpu_only_host_callback(&gpu_only_ctxs[w]);
        }
        // Otherwise nothing is written to tx_flags: the slot is in flight in pipeline-private
        // state (slot_occupied, inflight_slot_tags) until a response is published.
        cursor = (cursor + 1) % num_slots;
    }
    for (auto &wr : worker_resources) cudaStreamSynchronize(wr.stream);  // drains graphs and host-func callbacks
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
2. **Write** the RPC response (`RPCResponse` + `DecodeResponse`) into the TX slot handed over as `cpu_stage_context::response_buffer` (`PyMatchJob::response_buffer`; the RX slot itself on the in-place internal ring).
3. **Signal** slot completion via `pipeline.complete_deferred(origin_slot)`, which stores the TX slot host address into `tx_flags[origin_slot]`.

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
                                   └──► tx_flags[slot].store(tx_slot_addr)
```

### 6.5 PyMatchQueue

Thread-safe MPSC queue using `std::mutex` + `std::condition_variable`:

```cpp
struct PyMatchJob {
    int origin_slot;
    uint64_t request_id;
    void *response_buffer;   // TX slot host pointer from cpu_stage_context
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

## 7. Out-of-Order Consumer (internal ring)

The consumer thread harvests completions **out-of-order** by scanning all active slots on every iteration, rather than waiting for a sequential `next_harvest` counter. This eliminates head-of-line blocking where a slow request in slot N would prevent harvesting faster completions in slot N+1.

### 7.1 Consumer Logic (Pseudocode)
```cpp
while (!consumer_stop) {
    bool found_any = false;
    for (uint32_t s = 0; s < NUM_SLOTS; ++s) {
        if (!slot_occupied[s]) continue;

        if (ring->poll_tx(s) == CUDAQ_TX_READY) {   // any non-zero tx_flags value
            int err = slot_cuda_error[s];           // set by the dispatcher on a failed launch
            completion_handler({slot_request[s], s, err == 0, err});
            slot_cuda_error[s] = 0;
            total_completed++;

            slot_occupied[s] = 0;      // Reset occupancy FIRST
            __sync_synchronize();       // ARM memory fence
            ring->clear_slot(s);        // Then clear tx_flags (rx_flags was cleared by the dispatcher)
            found_any = true;
        }
    }
    if (!found_any) CUDAQ_REALTIME_CPU_RELAX();
}
```

The consumer uses `slot_occupied[]` (set by the producer, cleared by the consumer) to know which slots are active, and `poll_tx()`, which reports `READY` for any non-zero `tx_flags` value, to find published responses. An occupied slot with `tx_flags == 0` is in flight.

### 7.2 Consumer-Producer Race Fix

On ARM's weakly ordered memory model, the consumer must reset `slot_occupied[s] = 0` **before** clearing the ring buffer flags (via `clear_slot`), with a `__sync_synchronize()` fence between them. Without this ordering:
1. Consumer clears flags (slot appears free to producer)
2. Producer writes new `slot_occupied[s] = 1` 
3. Consumer's delayed `slot_occupied[s] = 0` clobbers the producer's write

This race caused exactly one request to get "stuck" indefinitely, eventually stalling the entire pipeline through backpressure.

### 7.3 Publishing and External Rings

Every completion path ends in the pipeline's `publish(slot, tx_host)`: a release store of the TX slot host address into `tx_flags[slot]`. The callers are the worker thread (byte count returned by the CPU stage), `complete_deferred()` (deferred completions), the GPU-only host callback and the dispatcher's launch-failure path.

On the pipeline-owned **internal ring** the consumer thread above harvests the published slot. On an **external ring** (`pipeline_stage_config::external_ringbuffer`, e.g. the GpuRoce transceiver's rings) the transport owns `tx_flags`: its TX kernel treats any non-zero value as an address inside the TX ring, sends `published_addr - tx_ring_base` and zeroes the flag itself. The pipeline therefore never polls or clears `tx_flags` on an external ring and starts no consumer thread; `publish()` calls the completion handler and increments `total_completed` right after the flag store, on the publishing thread (worker thread, `complete_deferred` caller or dispatcher). The request id comes from `slot_request[slot]`, captured by the dispatcher before the response overwrote the slot.

---

## 8. realtime_pipeline Scaffolding

The low-level dispatcher, consumer, and worker threads are wrapped by a higher-level `realtime_pipeline` class (`libs/qec/include/cudaq/qec/realtime/pipeline.h`) that hides all ring buffer management, atomics, and thread lifecycle. Application code provides three callbacks:

1. **GPU stage factory** (`gpu_stage_factory`): Called once per worker during `start()`. Returns the `cudaGraphExec_t`, `cudaStream_t`, `pre_launch_fn`, `post_launch_fn`, `function_id`, and an opaque `user_context` for each worker.
2. **CPU stage callback** (`cpu_stage_callback`): Called by each worker thread when GPU inference completes. Receives `cpu_stage_context` containing `gpu_output`, `gpu_output_size`, `response_buffer` (the TX slot host pointer; `max_response_size` is the TX stride), and the `user_context`. Returns the number of bytes written, `0` if no result ready (poll again), or `DEFERRED_COMPLETION` to release the worker without signaling slot completion.
3. **Completion callback** (`completion_callback`): Called for each completed (or errored) request with a `completion` struct: by the consumer thread on the internal ring, or on the publishing thread (Section 7.3) on an external ring. `completion::success` is false and `cuda_error` holds the `cudaError_t` when `cudaGraphLaunch` failed.

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
- Does NOT write to `tx_flags[origin_slot]`: no response is published and the slot stays occupied in pipeline-private state

The caller is responsible for eventually calling `pipeline.complete_deferred(slot)`, which stores the TX slot host address (== the RX slot on the in-place internal ring) into `tx_flags[slot]` with release semantics, making the completion visible to the consumer (internal ring) or to the transport's TX kernel (external ring). `complete_deferred()` stays callable after `stop()`; ring memory is released in the destructor.

### 8.2 GPU-Only Mode

If no `cpu_stage_callback` is registered, the pipeline operates in **GPU-only mode**: no CPU worker threads are spawned. After a successful launch (and the application's `post_launch_fn`, if any) the dispatcher enqueues a `cudaLaunchHostFunc` on the worker stream. When the GPU finishes, the CUDA runtime calls the host function, which publishes the TX slot address into `tx_flags` and then restores the `idle_mask` bit — all from the CUDA callback thread, with no CUDA calls inside. The pipeline writes no `RPCResponse` in this mode; the graph and `post_launch_fn` own the TX slot content, which is why GPU-only mode is rejected on an external ring (nobody frames a response for the transport).

### 8.3 ring_buffer_injector

The `ring_buffer_injector` class (created via `pipeline.create_injector()`) encapsulates the host-side submission logic for testing without FPGA hardware. It provides:

- `try_submit()`: Non-blocking, returns false on backpressure. Checks `slot_occupied[slot]` to determine if the next slot is free — this mirrors the hsb model where the FPGA checks `ring_flag` (mapped to `rx_flags`) rather than a separate tx_flags sentinel.
- `submit()`: Blocking spin-wait until a slot becomes available.
- `backpressure_stalls()`: Counter of spin iterations during backpressure.

The injector uses a round-robin slot selection with atomic CAS for thread safety.

The `pipeline_stage_config` allows configuring `num_workers`, `num_slots` (used verbatim), `slot_size`, optional `core_pinning` for dispatcher, consumer, and worker threads, and `external_ringbuffer`, which hands the pipeline a transport-owned ring (Section 7.3).

---

## 9. Step-by-Step Data Flow Trace

1. **Producer** writes uint8 measurements into `payload_buf` from Stim test data.
2. **Producer** calls `injector.submit(fid, payload, size, request_id)`.
3. **ring_buffer_injector** records `slot_request[slot] = request_id`, sets `slot_occupied[slot] = 1`, then writes RPC header (`RPCHeader`: magic, function_id, arg_len, request_id, ptp_timestamp = 24 bytes) + payload into `rx_data[slot]`.
4. **ring_buffer_injector** sets `rx_flags[slot] = host_ptr` (release).
5. **Host Dispatcher** reads `rx_flags[slot]`, sees data.
6. **Host Dispatcher** checks the RPC magic and copies `function_id`, `request_id` and `ptp_timestamp` out of the header.
7. **Host Dispatcher** scans `idle_mask`, finds `worker_id = 2` free with a matching `function_id`.
8. **Host Dispatcher** saves `inflight_slot_tags[2] = slot`.
9. **Host Dispatcher** marks bit 2 busy in `idle_mask`.
10. **Host Dispatcher** translates `host_ptr` to the slot device pointer `dev_ptr` and derives the TX slot host pointer (`tx_data_host + slot * tx_stride`).
11. **Host Dispatcher** calls `pre_launch_fn`: writes `h_ring_ptrs[0] = dev_ptr`, derives host pointer via offset arithmetic, issues `cudaMemcpyAsync(d_trt_input, host_ptr + 24, input_size, H2D, stream[2])`.
12. **Host Dispatcher** calls `cudaGraphLaunch(..., stream[2])`.
13. **Host Dispatcher** clears `rx_flags[slot] = 0` and advances to the next slot. Nothing is written to `tx_flags`; the slot is in flight in pipeline-private state (`slot_occupied[slot]`, `inflight_slot_tags[2]`). The producer does not check `tx_flags`; backpressure is via `slot_occupied[]`.
14. **GPU DMA engine** copies input payload from ring buffer to TRT input buffer.
15. **GPU** executes TRT inference (or passthrough copy in SKIP_TRT mode).
16. **GPU DMA engine** copies TRT output to host-mapped `h_predecoder_outputs_`.
17. **GPU signal kernel** sets `ready_flags[0] = 1` (system-scope atomic release).
18. **Predecoder Worker** CAS(1, 2) on `ready_flags[0]`, wins, reads inference output.
19. **Predecoder Worker** copies output to `deferred_outputs[origin_slot]`.
20. **Predecoder Worker** computes syndrome density metrics.
21. **Predecoder Worker** calls `release_job(0)` → `ready_flags[0].store(0, release)`.
22. **Predecoder Worker** reads `request_id` from the RPC header in the host RX slot, enqueues `PyMatchJob` carrying `ctx.response_buffer`.
23. **Predecoder Worker** returns `DEFERRED_COMPLETION`.
24. **Pipeline** restores bit 2 in `idle_mask` (worker free for next dispatch). Does NOT touch `tx_flags`.
25. **PyMatching Worker** pops `PyMatchJob` from queue, acquires per-thread decoder.
26. **PyMatching Worker** runs PyMatching MWPM decode over full parity check matrix.
27. **PyMatching Worker** writes `RPCResponse + DecodeResponse` into the TX slot (`PyMatchJob::response_buffer`).
28. **PyMatching Worker** calls `pipeline.complete_deferred(slot)` → `tx_flags[slot].store(tx_slot_addr, release)`.
29. **Consumer** scans all slots where `slot_occupied[s]` is set, calls `poll_tx(s)`, which reports `READY` for any non-zero `tx_flags` value.
30. **Consumer** calls `completion_handler(request_id, slot, success = slot_cuda_error[slot] == 0, cuda_error)`.
31. **Consumer** sets `slot_occupied[slot] = 0`, `__sync_synchronize()`, then calls `clear_slot(slot)` (clears `tx_flags` only; `rx_flags` was cleared in step 13). Producer may now reuse slot.

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

1. **Dispatcher**: After the launch, clears `rx_flags`; `tx_flags` stays 0 and the slot is tracked in flight by pipeline-private state (`slot_occupied`, `inflight_slot_tags`). On a failed `cudaGraphLaunch` it writes an `RPCResponse` with the error status into the TX slot, records the error in `slot_cuda_error[slot]` and publishes the TX slot address.
2. **Producer**: Reuses a slot only when `slot_occupied[slot] == 0`. Does not read `tx_flags`.
3. **Consumer**: Scans slots where `slot_occupied[s]` is set; any non-zero `tx_flags` is a published response, harvested with `success = (slot_cuda_error[s] == 0)`.

**Slot lifecycle**: Idle (`slot_occupied=0`) → Written (`slot_occupied=1`, `rx_flags=ptr`) → Dispatched (`rx_flags=0`, `tx_flags=0`, pipeline-tracked, GPU processing) → Completed (`tx_flags=TX slot addr`) → Consumer harvests, `slot_occupied=0`, `tx_flags` cleared → Idle. **Launch error**: Dispatched → Completed with `completion::success=false` and `cuda_error`, through the same path.

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

### 12.3 Integration Tests (`PipelineDispatchTest`, 11 tests)
Driven through the public `realtime_pipeline` API with passthrough predecoders:
- **Dispatcher lifecycle**: Shutdown semantics, stats counter accuracy, invalid RPC magic rejection (on an external ring, where the drop is observable), slot wraparound.
- **Single Request Round-Trip**: Full dispatcher -> graph -> CPU stage -> completion data path.
- **Multi-Predecoder Concurrency**: 4 predecoders on 4 streams with distinct function ids, simultaneous dispatch, per-predecoder data verification.
- **Sustained Throughput (200 requests)**: 2 workers sharing one function id over 8 slots; exercises injector backpressure and proves indefinite stability of the host-side dispatcher.
- **Deferred Completion**: `DEFERRED_COMPLETION` followed by `complete_deferred()` from the test thread.
- **GPU-Only Round-Trip**: no CPU stage; completion comes from the host callback.
- **Launch Failure Reports Error**: null `graph_exec` -> completion with `success=false` and the CUDA error, an `RPCResponse` with non-zero status in the TX slot, and a worker that is reusable afterwards.

### 12.4 End-to-End Benchmark (test_realtime_predecoder_w_pymatching)
- Configurable surface code distance and round count: d7, d13, d13_r104, d21, d31.
- Loads Stim-generated test data (detectors, observables, parity check matrix, priors).
- Streams syndrome data at configurable rate with correctness verification (LER).
- Reports latency percentiles, throughput, backpressure stalls, syndrome density reduction.
- `--num-gpus N` is accepted but currently clamped to 1 (multi-GPU dispatch is not yet supported; see Section 15).

---

## 13. Shutdown and Grace Period

- **Grace period**: After the producer stops submitting, `stop()` waits up to 5 seconds for `total_completed >= total_submitted` (internal ring only; an external ring cannot be drained by count because the transport owns the request stream).
- **Dispatcher shutdown**: `stop()` then sets `shutdown_flag = 1` and joins the dispatcher thread. The dispatcher synchronizes all worker streams before returning, so every graph and every GPU-only host callback has finished and published.
- **Consumer and workers**: `consumer_stop` is set after the dispatcher has joined, so the consumer thread (internal ring) exits even when `total_completed < total_submitted`; the worker threads are joined last. This keeps `stop()` from hanging on requests that never complete.
- **PyMatching thread pool**: Shut the queue down (`pymatch_queue.shutdown()` unblocks all waiting threads), then stop the pipeline and join the PyMatching worker threads. The reverse order is also safe: `complete_deferred()` stays callable after `stop()`, so a decode that finishes late still publishes into ring memory that lives until the pipeline is destroyed.

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

Multi-GPU dispatch is **not currently supported**. The pipeline's dispatcher thread launches every worker's graph on one device (the device current on the thread that called `start()`), so workers cannot span multiple devices. The `--num-gpus` flag is accepted but clamped to 1 with a warning. See Section 15 for details.

### 14.3 Raw TRT Baseline (trtexec)

| Mode | GPU Compute | Total Host Latency |
| :--- | :--- | :--- |
| Default | 107 µs | 119 µs |
| CUDA Graph + SpinWait | 90 µs | 99 µs |
| CUDA Graph + SpinWait + No Transfers | 88 µs | 88 µs |

---

## 15. Multi-GPU Support (Not Yet Implemented)

Multi-GPU dispatch is **disabled** pending an extension of the pipeline's dispatcher. The `--num-gpus` flag is accepted but clamped to 1.

### 15.1 Known Issue

The pipeline's dispatcher thread (`realtime_pipeline::Impl::dispatcher_loop`) calls `cudaSetDevice(device_id)` once, with the device captured on the thread that called `start()`, and every worker thread does the same. CUDA streams and graphs are bound to the device on which they were created, but `cudaGraphLaunch` dispatches to the **calling thread's current device**, not the stream's device. A graph captured on GPU 1 with a stream created on GPU 1, launched while the dispatcher's current device is GPU 0, hangs or silently fails.

### 15.2 Fix Required

To support multi-GPU, each worker needs a per-worker device id (a `gpu_worker_resources` field or a factory-side table), and the pipeline's dispatcher loop must call `cudaSetDevice(worker_device)` before `pre_launch_fn` and `cudaGraphLaunch`, with the worker thread doing the same before touching that worker's stream.

---

## 16. Invariants

The implementation in `realtime_pipeline.cu` keeps the following invariants; changes to the dispatcher, worker, consumer or publish paths must preserve them.

- **No CUDA stream querying**: `cudaStreamQuery()` is not used for backpressure or completion checking (it incurs severe driver latency). Completion is observed through `idle_mask`, `ready_flags` and `tx_flags` only.
- **No weak-ordering bugs**: No `volatile`, no `__threadfence_system()`. Cross-device synchronization uses `cuda::std::atomic<T, cuda::thread_scope_system>` (or `<cuda/atomic>` with `thread_scope_system`); host-only flags use release/acquire atomics.
- **No head-of-line blocking**: The dispatcher never maps slots to predecoders statically; it allocates dynamically through `idle_mask`. The consumer harvests out-of-order by scanning all occupied slots. Workers running the same model share a single `function_id` so the dispatcher can pick any idle worker; per-worker unique ids degenerate function-aware routing into a 1:1 static mapping.
- **No data loss**: When no idle worker serves the request's `function_id`, the dispatcher spins on the current slot (`CUDAQ_REALTIME_CPU_RELAX()`) with `rx_flags` untouched. It advances only after a worker is allocated and the launch attempted. Only a bad magic or an unknown `function_id` drops a slot, and drops are not counted in `Stats::dispatched`.
- **Tag ownership**: `inflight_slot_tags[w]` is written by the dispatcher (release) before it clears bit `w`, and the worker thread reads it (acquire) only after observing its bit clear, so the tag is the one for the current assignment and stays stable until that thread sets the bit again. Only the dispatcher clears `idle_mask` bits; in CPU-stage mode only worker thread `w` sets bit `w` (a failed launch is handed to it through `worker_launch_failed[w]`), and in GPU-only mode the host callback sets the bit last, after publishing. The tag for a worker therefore cannot change while that worker's flow is active.
- **Ready-flag claiming**: The CPU poller claims each completion exactly once with `compare_exchange_strong(1, 2)` on the ready flag, relaxed on failure; the worker clears the flag (store 0) in `release_job`.
- **Backpressure model**: The producer (`ring_buffer_injector`) checks `slot_occupied[slot]`, never `tx_flags`, mirroring the hsb model where `ring_flag`/`rx_flags` is the sole sender-receiver backpressure mechanism. `slot_request[slot]` and `slot_occupied[slot]` are written before the slot is signalled. Between dispatch and publication `tx_flags` stays 0; the slot is tracked in flight by pipeline-private state, and the consumer treats any non-zero `tx_flags` as a published response.
- **Consumer memory ordering**: The consumer sets `slot_occupied[s] = 0` BEFORE calling `cudaq_host_ringbuffer_clear_slot`, with a `__sync_synchronize()` fence between them, to prevent the producer-consumer race on ARM. `clear_slot` clears `tx_flags` only; `rx_flags` is cleared by the dispatcher when it takes the request.
- **Published value**: `tx_flags[slot]` is either 0 or the host address of the TX slot that holds the `RPCResponse`; no other value is ever stored. On an external ring the pipeline never polls or clears `tx_flags` (the transport's TX kernel does) and reports completion at publish time on the publishing thread.
- **Launch errors**: A failed `cudaGraphLaunch` produces an `RPCResponse` with the `cudaError_t` in `status` in the TX slot, a completion with `success = false` and `cuda_error`, and a worker that is reusable afterwards. Errors never hang the pipeline.
- **In-place aliasing**: On the internal ring the TX slot is the RX slot. Every response writer (dispatcher failure path, CPU stages, PyMatching threads) reads the request fields it needs before writing the response.
- **DMA data movement**: `cudaMemcpyAsync` (DMA engine) moves the data. The input copy is issued by `pre_launch_fn` before the graph launch at offset `CUDAQ_RPC_HEADER_SIZE` (24 bytes), `cudaMemcpyHostToDevice` from the pinned ring buffer host pointer. The output copy is captured inside the graph. No SM-based byte-copy kernels for fixed-address transfers.
- **No input kernel in graph**: The captured CUDA graph contains no input-copy kernel; all input data movement is the `pre_launch_fn` DMA callback issued on the worker stream before `cudaGraphLaunch`.
- **Deferred completion**: When the CPU stage returns `DEFERRED_COMPLETION`, the pipeline releases `idle_mask` and publishes nothing. The caller completes the slot with `complete_deferred(slot)`, which publishes the TX slot address; it stays callable after `stop()`. Because the worker stream is reused immediately, a stage must copy the predecoder output it needs before returning `DEFERRED_COMPLETION`.
- **Shutdown**: `stop()` orders producer stop, grace wait (internal ring only), `shutdown_flag`, dispatcher join (which synchronizes all worker streams), `consumer_stop`, consumer join, worker joins, so the process exits even when `total_completed < total_submitted`. Applications shut the PyMatching queue down and then stop the pipeline; the reverse order is also safe.
- **Device affinity**: The dispatcher and worker threads run on the device current on the thread calling `start()`; factory-created streams and graphs must belong to it.
