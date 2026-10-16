# Hybrid AI Predecoder + PyMatching Global Decoder Pipeline

## Design Document

**Component**: `cudaq-qec` Realtime Decoding Subsystem  
**Status**: Implementation Complete (Test-Validated)  
**Last Updated**: 2026-03-17

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Architecture](#3-architecture)
4. [Component Deep-Dive](#4-component-deep-dive)
   - 4.1 [Ring Buffer & RPC Protocol](#41-ring-buffer--rpc-protocol)
   - 4.2 [Host-Side Dispatcher](#42-host-side-dispatcher)
   - 4.3 [AIDecoderService (Base Class)](#43-aidecoderservice-base-class)
   - 4.4 [AIPreDecoderService (Predecoder + CPU Handoff)](#44-aipredeccoderservice-predecoder--cpu-handoff)
   - 4.5 [Decoupled CPU Worker Architecture](#45-decoupled-cpu-worker-architecture)
5. [Data Flow](#5-data-flow)
6. [Memory Architecture](#6-memory-architecture)
7. [Backpressure Protocol](#7-backpressure-protocol)
8. [Memory Ordering & Synchronization](#8-memory-ordering--synchronization)
9. [CUDA Graph Structure](#9-cuda-graph-structure)
10. [Pipeline Configurations](#10-pipeline-configurations)
11. [File Inventory](#11-file-inventory)
12. [Configuration Parameters](#12-configuration-parameters)
13. [Performance Benchmarking](#13-performance-benchmarking)
14. [Portability](#14-portability)
15. [Limitations & Future Work](#15-limitations--future-work)

---

## 1. Overview

This system implements a **realtime hybrid GPU/CPU pipeline** for quantum error correction (QEC) decoding on the surface code. The pipeline splits the decoding workload into two stages:

| Stage | Location | Algorithm | Data Type |
|-------|----------|-----------|-----------|
| **Predecoding** | GPU | Neural network (TensorRT, from ONNX) | uint8 |
| **Global Decoding** | CPU | PyMatching (MWPM) | float64 |

A **host-side spin-polling dispatcher** monitors a shared ring buffer for incoming syndrome data. When data arrives, the dispatcher finds a free GPU worker from a dynamic bitmask pool (`idle_mask`), copies the input via DMA, and launches a pre-captured CUDA graph containing a TensorRT inference pass. The neural network accepts raw detector values as uint8 tensors and produces residual detectors and a logical frame prediction. The residual detectors are handed off to a **decoupled PyMatching thread pool** via a lock-free queue, where per-thread MWPM decoders compute corrections. Results are written back to the ring buffer and acknowledged.

The system supports multiple surface code distances and round counts via a configurable `PipelineConfig` struct: d=7/T=7, d=13/T=13, d=13/T=104, d=21/T=21, and d=31/T=31. ONNX models are compiled to TensorRT engines on first use and cached to disk as `.engine` files for fast reloading on subsequent runs.

---

## 2. Problem Statement

Surface code QEC requires decoding syndrome measurements within the coherence time of the quantum system. A pure CPU decoder cannot meet this budget at scale. A pure GPU decoder lacks the flexibility to run algorithms like Minimum Weight Perfect Matching (MWPM) efficiently on GPU.

The hybrid approach exploits the strengths of each:

- **GPU**: Massively parallel neural network inference provides fast soft-decision outputs (residual detectors) that reduce the problem complexity for the global decoder. The predecoder achieves **98.3% syndrome density reduction** for d=13/T=104.
- **CPU**: PyMatching solves the residual MWPM problem on the simplified output from the predecoder.

The critical constraint is **zero-copy, zero-allocation** on the hot path. Every buffer is pre-allocated, every kernel is pre-captured into a CUDA Graph, and every transfer uses mapped pinned memory or DMA.

---

## 3. Architecture

### System Diagram

```
 Test Harness (or FPGA DMA)
       │
       │  syndrome data (uint8 detectors)
       ▼
 ┌─────────────────────────────────────────────────────┐
 │           Ring Buffer (Mapped Pinned Memory)         │
 │  ┌──────┐ ┌──────┐ ┌──────┐        ┌──────┐       │
 │  │Slot 0│ │Slot 1│ │Slot 2│  ...   │Slot15│       │
 │  └──┬───┘ └──┬───┘ └──┬───┘        └──┬───┘       │
 │     │        │        │               │            │
 │  rx_flags[0] rx_flags[1] ...   rx_flags[15]        │
 └─────┼────────┼────────┼───────────────┼────────────┘
       │        │        │               │
       ▼        ▼        ▼               ▼
 ┌─────────────────────────────────────────────────────┐
 │          Host-Side Dispatcher Thread                 │
 │                                                     │
 │   Polls rx_flags[] ──► Finds free worker (idle_mask)│
 │   ──► DMA copy (pre_launch_fn) ──► cudaGraphLaunch  │
 └──────────┬──────────┬──────────┬──────────┬─────────┘
            │          │          │          │
            ▼          ▼          ▼          ▼
 ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │ PreDecoder 0 │ │PreDec. 1 │ │   ...    │ │PreDec. 7 │
 │ (CUDA Graph) │ │(CUDAGraph│ │          │ │(CUDAGraph│
 │              │ │          │ │          │ │          │
 │  TRT Infer   │ │   ...    │ │   ...    │ │   ...    │
 │  DMA Output  │ │          │ │          │ │          │
 │  Signal Kern │ │          │ │          │ │          │
 └──────┬───────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
        │              │            │            │
        │   (mapped pinned memory: ready_flags, h_predecoder_outputs_)
        ▼              ▼            ▼            ▼
 ┌─────────────────────────────────────────────────────┐
 │  Predecoder Workers (1:1 with GPU streams)           │
 │  CAS(1,2) on ready_flags → copy output → enqueue    │
 │  Release predecoder → return DEFERRED_COMPLETION     │
 └──────────┬──────────────────────────────────────────┘
            │  PyMatchQueue (mutex + condvar)
            ▼
 ┌──────────────┐ ┌──────────┐       ┌──────────────┐
 │  PyMatch 0   │ │PyMatch 1 │  ...  │  PyMatch 15  │
 │ (thread pool)│ │(thd pool)│       │ (thread pool) │
 │              │ │          │       │               │
 │ PyMatching   │ │PyMatch   │       │ PyMatching    │
 │ (own decoder)│ │(own dec) │       │ (own decoder) │
 │ Write RPC    │ │Write RPC │       │ Write RPC     │
 │ complete_    │ │complete_ │       │ complete_     │
 │  deferred()  │ deferred() │       │  deferred()   │
 └──────┬───────┘ └────┬─────┘       └────┬──────────┘
        │              │                  │
        └──────────────┼──────────────────┘
                       ▼
 ┌─────────────────────────────────────────────────────┐
 │  Consumer Thread                                     │
 │  Scans tx_flags[] ──► completion_handler ──► clear   │
 └─────────────────────────────────────────────────────┘
              tx_flags[slot] ──► Producer can reuse slot
```

### Key Design Decisions

1. **Host-side dispatcher with dynamic worker pool** -- The dispatcher runs as a dedicated CPU thread, polling `rx_flags` and dynamically allocating GPU workers via an atomic `idle_mask` bitmask. This replaced a device-side persistent kernel that hit a CUDA 128-launch limit.

2. **CUDA Graphs for inference** -- Each predecoder instance has a pre-captured CUDA graph containing TRT inference, output DMA copy, and a signal kernel. Input data is injected via a `pre_launch_fn` DMA callback before graph launch (since the source address is dynamic).

3. **Mapped pinned memory for GPU→CPU handoff** -- `cudaHostAllocMapped` provides a single address space visible to both CPU and GPU without explicit copies. GPU writes are made visible via libcu++ system-scope atomics with release semantics; CPU reads use acquire semantics.

4. **Queue depth 1 per predecoder** -- Each `AIPreDecoderService` has a single in-flight inference slot. Deeper queues were found to add complexity without measurable throughput benefit, since 8 parallel streams already exceed the GPU's throughput capacity.

5. **Decoupled predecoder and PyMatching workers** -- GPU polling threads release the predecoder stream immediately after copying output (~10 µs), then hand off to a separate PyMatching thread pool via `PyMatchQueue`. This prevents slow CPU decodes (~224 µs) from blocking GPU dispatch.

6. **ONNX model support with engine caching** -- The `AIDecoderService` accepts either a pre-built `.engine` file or an `.onnx` model. When given an ONNX file, it builds a TensorRT engine at runtime and optionally saves it to disk via the `engine_save_path` parameter.

7. **Per-worker PyMatching decoder pool** -- Each PyMatching thread gets its own pre-allocated decoder instance via `thread_local` assignment. This eliminates mutex contention on the decode path.

8. **Type-agnostic I/O buffers** -- All TRT I/O buffers use `void*` rather than `float*`, supporting uint8 and INT32 models natively without type casting.

9. **Stim-derived parity check matrix** -- The PyMatching decoders are initialized from a full parity check matrix (`H`) and observable matrix (`O`) exported from Stim, rather than the `cudaq-qec` surface code's per-slice `H_z`. This enables full-H decoding with proper edge weighting via priors.

---

## 4. Component Deep-Dive

### 4.1 Ring Buffer & RPC Protocol

**Files**: `dispatch_kernel_launch.h` (protocol), `cudaq_realtime.h` (C API), `realtime_pipeline.cu` (RingBufferManager)

The ring buffer is the communication channel between the producer (FPGA or test harness) and the GPU. It consists of:

| Buffer | Type | Size | Purpose |
|--------|------|------|---------|
| `rx_flags[N]` | `cuda::atomic<uint64_t, system>` | N slots | Non-zero = data ready; value is pointer to slot data |
| `tx_flags[N]` | `cuda::atomic<uint64_t, system>` | N slots | Non-zero = response ready; acknowledges to consumer |
| `rx_data` | `uint8_t*` | N x SLOT_SIZE | Slot payload area (mapped pinned) |

Each slot carries an **RPC message** in a packed wire format:

```
Request:  [RPCHeader: magic(4) | function_id(4) | arg_len(4) | request_id(4) | ptp_timestamp(8)]
          [payload: arg_len bytes]
          Total header: 24 bytes (CUDAQ_RPC_HEADER_SIZE)

Response: [RPCResponse: magic(4) | status(4) | result_len(4)]
          [payload: result_len bytes]
```

The `function_id` is an FNV-1a hash of the target function name, enabling the dispatcher to route requests to different predecoder instances.

The response payload for the PyMatching pipeline is a packed `DecodeResponse`:

```c
struct __attribute__((packed)) DecodeResponse {
    int32_t total_corrections;
    int32_t converged;
};
```

### 4.2 Host-Side Dispatcher

**File**: `realtime/lib/daemon/dispatcher/host_dispatcher.cu`

The dispatcher is a **spin-polling host thread** running on a dedicated CPU core. It monitors the ring buffer's `rx_flags` and dispatches work to GPU streams.

#### Worker Pool

The dispatcher manages a pool of `num_workers` GPU streams. Each worker is described by a `cudaq_host_dispatch_worker_t`:

```c
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

#### Dispatch Loop

```
while (!shutdown):
    rx_value = rx_flags[current_slot].load(acquire)
    if rx_value == 0: QEC_CPU_RELAX(); continue

    // Find free worker via idle_mask bitmask
    worker_id = ffsll(idle_mask.load(acquire)) - 1
    if worker_id < 0: QEC_CPU_RELAX(); continue

    // Claim worker, tag origin slot
    idle_mask.fetch_and(~(1ULL << worker_id), release)
    inflight_slot_tags[worker_id] = current_slot

    // Pre-launch: DMA input to TRT buffer
    if pre_launch_fn: pre_launch_fn(data, dev_ptr, stream)

    // Launch CUDA graph
    cudaGraphLaunch(graph_exec, stream)

    // Mark in-flight, consume slot
    tx_flags[current_slot].store(0xEEEE..., release)
    rx_flags[current_slot].store(0, release)

    // Post-launch callback (GPU-only mode)
    if post_launch_fn: post_launch_fn(...)

    current_slot = (current_slot + 1) % num_slots
```

### 4.3 AIDecoderService (Base Class)

**Files**: `ai_decoder_service.h`, `ai_decoder_service.cu`

The base class manages the TensorRT lifecycle.

#### Constructor

```cpp
AIDecoderService(const std::string& model_path, void** device_mailbox_slot,
                 const std::string& engine_save_path = "");
```

The constructor accepts either a `.engine` file (fast deserialization) or an `.onnx` file (builds TRT engine via autotuner). When `engine_save_path` is non-empty and the model is ONNX, the built engine is serialized to disk for caching.

#### Responsibilities

- **Engine loading**: Deserializes a TensorRT `.engine` file or builds from `.onnx` via `NvOnnxParser`.
- **Engine caching**: Saves built engines to disk via `engine_save_path` for fast reload.
- **Dynamic tensor binding**: Enumerates all I/O tensors from the engine, storing metadata in `TensorBinding` structs. Supports models with multiple outputs.
- **Buffer allocation**: Allocates persistent device buffers sized to the engine's static tensor shapes. Uses `void*` for type-agnostic I/O.
- **Dynamic batch handling**: Automatically pins dynamic dimensions to 1 via optimization profiles.

#### Dynamic Tensor Binding

```cpp
struct TensorBinding {
    std::string name;
    void* d_buffer = nullptr;
    size_t size_bytes = 0;
    bool is_input = false;
};
std::vector<TensorBinding> all_bindings_;
```

### 4.4 AIPreDecoderService (Predecoder + CPU Handoff)

**Files**: `ai_predecoder_service.h`, `ai_predecoder_service.cu`

This derived class replaces the base class's autonomous graph with one that hands inference results off to the CPU.

#### Constructor

```cpp
AIPreDecoderService(const std::string& engine_path, void** device_mailbox_slot,
                    int queue_depth = 1, const std::string& engine_save_path = "");
```

#### CUDA Graph Structure

```
[Pre-launch DMA: ring buffer → d_trt_input (host-side callback)]
     ↓
TRT enqueueV3 (AI predecoder inference)
     ↓
cudaMemcpyAsync D2D (d_trt_output_ → h_predecoder_outputs_)
     ↓
predecoder_signal_ready_kernel (ready_flags.store(1, release))
```

The input DMA copy is NOT in the graph — it's issued by the `pre_launch_fn` callback on the worker stream before `cudaGraphLaunch`, because the source address (ring buffer slot) changes each invocation.

#### Per-Predecoder Buffers (queue_depth=1)

| Buffer | Host Pointer | Device Pointer | Purpose |
|--------|-------------|---------------|---------|
| `h_ready_flags_` | CPU reads/writes | `d_ready_flags_` GPU writes | 1 = job ready, 0 = slot free |
| `h_ring_ptrs_` | CPU reads | `d_ring_ptrs_` GPU writes | Original ring buffer address per job |
| `h_predecoder_outputs_` | CPU reads | `d_predecoder_outputs_` GPU writes | TRT inference output (`void*`, uint8) |

All buffers are allocated with `cudaHostAllocMapped` and mapped to device pointers via `cudaHostGetDevicePointer`.

#### CPU Interface

```cpp
bool poll_next_job(PreDecoderJob& out_job);
void release_job(int slot_idx);
```

`poll_next_job` performs CAS(expected=1, desired=2) on `ready_flags[0]`. If successful, it populates the `PreDecoderJob` struct with the slot index, ring buffer pointer, and inference output pointer.

`release_job` stores 0 to the ready flag with release semantics, allowing the GPU to reuse the slot.

### 4.5 Decoupled CPU Worker Architecture

**File**: `test_realtime_predecoder_w_pymatching.cpp`

The CPU-side processing uses a **two-tier decoupled architecture**:

#### Tier 1: Predecoder Workers (GPU Polling)

Pipeline worker threads (1:1 with GPU streams) run in the `RealtimePipeline::worker_loop`. Each iteration:

1. Polls `poll_next_job()` (CAS on ready_flags).
2. Copies inference output to `deferred_outputs[origin_slot]` (per-slot buffer).
3. Computes syndrome density metrics.
4. Releases predecoder via `release_job(0)`.
5. Enqueues `PyMatchJob{origin_slot, request_id, ring_buffer_ptr}` to `PyMatchQueue`.
6. Returns `DEFERRED_COMPLETION` → pipeline releases `idle_mask`, skips `tx_flags`.

**Hold time**: ~10 µs (copy + release + enqueue).

#### Tier 2: PyMatching Workers (CPU Decode)

A separate thread pool (16 workers for d13_r104) processes `PyMatchJob`s:

1. Pops job from `PyMatchQueue` (blocks if empty).
2. Acquires per-thread PyMatching decoder via `thread_local` lock-free assignment.
3. Runs PyMatching MWPM decode over the full parity check matrix.
4. Writes `RPCResponse + DecodeResponse` into the ring buffer slot.
5. Calls `pipeline.complete_deferred(origin_slot)` → stores host address into `tx_flags`.

**Decode time**: ~224 µs average.

#### PyMatching Decoder Pool

```cpp
struct DecoderContext {
    std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders;
    std::atomic<int> next_decoder_idx{0};

    cudaq::qec::decoder* acquire_decoder() {
        thread_local int my_idx = next_decoder_idx.fetch_add(1);
        return decoders[my_idx % decoders.size()].get();
    }
};
```

Decoders are constructed at startup from immutable Stim-derived model inputs.
`O` remains available for the manual projection below, while the explicit
`errors` result type keeps PyMatching's output in the error-frame basis:

```cpp
auto H_full = stim_data.H.to_dense();
auto O = cudaq::qec::sparse_binary_matrix(stim_data.O.to_dense());
auto inputs = cudaq::qec::decoder_init(
    cudaq::qec::sparse_binary_matrix(H_full), std::move(O), stim_data.priors);
for (int i = 0; i < num_decode_workers; ++i)
    decoders.push_back(cudaq::qec::decoder::get(
        "pymatching", inputs, cudaq::qec::decode_result_type::errors,
        pm_params));
```

#### Observable Projection

When the observable matrix (`O`) is available, corrections are projected onto the logical observable:

```cpp
int obs_parity = 0;
for (size_t e = 0; e < result.result.size(); ++e)
    if (result.result[e] > 0.5 && obs_row[e])
        obs_parity ^= 1;
total_corrections += obs_parity;
```

The total corrections include both the predecoder's logical prediction (`output[0]`) and PyMatching's correction parity.

---

## 5. Data Flow

The following traces a single syndrome packet through the entire pipeline:

```
Step  Location          Action
────  ────────          ──────────────────────────────────────────────────
 1.   Producer          Writes RPCHeader (24 bytes) + uint8 detectors into rx_data[slot]
 2.   Injector          Sets rx_flags[slot] = host_ptr (release)
                        ── release fence ──
 3.   Dispatcher        Reads rx_flags[slot] (acquire), sees data
 4.   Dispatcher        Parses RPCHeader, extracts function_id
 5.   Dispatcher        Scans idle_mask via ffsll → finds free worker W
 6.   Dispatcher        Marks bit W busy, saves inflight_slot_tags[W] = slot
 7.   Dispatcher        Writes dev_ptr to h_mailbox_bank[W], __sync_synchronize()
 8.   Dispatcher        pre_launch_fn: h_ring_ptrs[0] = dev_ptr,
                        cudaMemcpyAsync(d_trt_input, dev_ptr+24, input_size, D2D, stream[W])
 9.   Dispatcher        cudaGraphLaunch(graph_exec[W], stream[W])
10.   Dispatcher        tx_flags[slot].store(0xEEEE..., release)  [IN_FLIGHT]
11.   Dispatcher        rx_flags[slot].store(0, release), advance slot
                        ── slot consumed ──

                        ── Inside CUDA Graph ──
12.   GPU               TRT enqueueV3: AI predecoder inference (uint8 → uint8)
13.   GPU               cudaMemcpyAsync D2D: d_trt_output_ → h_predecoder_outputs_
14.   GPU               predecoder_signal_ready_kernel: ready_flags.store(1, release)
                        ── Graph complete ──

15.   PreDec Worker     CAS(1, 2) on ready_flags[0] (acquire), wins
16.   PreDec Worker     Copies h_predecoder_outputs_ → deferred_outputs[slot]
17.   PreDec Worker     Computes syndrome density (input vs output nonzero counts)
18.   PreDec Worker     release_job(0): ready_flags.store(0, release)
19.   PreDec Worker     Extracts request_id from RPCHeader
20.   PreDec Worker     Enqueues PyMatchJob{slot, request_id, ring_buffer_ptr}
21.   PreDec Worker     Returns DEFERRED_COMPLETION
22.   Pipeline          idle_mask.fetch_or(1<<W, release)  [worker free]
                        ── tx_flags NOT written ──

23.   PyMatch Worker    Pops PyMatchJob from queue
24.   PyMatch Worker    Acquires per-thread decoder (thread_local)
25.   PyMatch Worker    Reads deferred_outputs[slot]: logical_pred + residual detectors
26.   PyMatch Worker    Runs PyMatching MWPM decode over full H matrix
27.   PyMatch Worker    Projects corrections onto observable O → obs_parity
28.   PyMatch Worker    Writes RPCResponse + DecodeResponse into ring_buffer_ptr
29.   PyMatch Worker    pipeline.complete_deferred(slot):
                        tx_flags[slot].store(host_addr, release)
                        ── FPGA/Consumer sees response ──

30.   Consumer          tx_flags[slot] != 0 and != 0xEEEE → harvest
31.   Consumer          completion_handler(request_id, slot, success)
32.   Consumer          slot_occupied[slot] = 0
33.   Consumer          __sync_synchronize()
34.   Consumer          clear_slot: tx_flags[slot] = 0
                        ── Slot available for reuse ──
```

---

## 6. Memory Architecture

### Allocation Map

```
┌─────────────────────────────────────────────────────────────┐
│                    PINNED MAPPED MEMORY                      │
│               (cudaHostAllocMapped + cudaHostGetDevicePointer)│
│                                                             │
│  Ring Buffer (RingBufferManager):                           │
│    rx_flags[16]           ← Producer writes, Dispatcher reads│
│    tx_flags[16]           ← Worker/PyMatch writes, Consumer  │
│    rx_data[16 x SLOT_SIZE]← Producer writes, GPU reads (DMA)│
│                              CPU reads (response area)       │
│                                                             │
│  Per-PreDecoder (x8):                                       │
│    h_ready_flags_[1]    ← GPU writes 1, CPU CAS 1→2, CPU 0 │
│    h_ring_ptrs_[1]      ← pre_launch_fn writes, CPU reads   │
│    h_predecoder_outputs_[1xN] ← GPU DMA writes, CPU reads   │
│                                                             │
│  Dispatcher:                                                │
│    h_mailbox_bank[8]    ← Dispatcher writes dev_ptr         │
│    shutdown_flag        ← Main writes, Dispatcher reads     │
│                                                             │
│  Lifecycle:                                                 │
│    idle_mask            ← Dispatcher clears, Workers set    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     DEVICE MEMORY                            │
│                                                             │
│  Per-PreDecoder (x8):                                       │
│    d_trt_input_   (void*)  ← DMA copy writes, TRT reads    │
│    d_trt_output_  (void*)  ← TRT writes, DMA copy reads    │
│    d_aux_buffers_ (void*)  ← Additional TRT outputs         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     HOST MEMORY                              │
│                                                             │
│  Per-Slot (x16):                                            │
│    deferred_outputs[slot]  ← Predecoder worker copies here  │
│                               PyMatch worker reads from here │
│                                                             │
│  Application:                                               │
│    slot_request[16]       ← Injector writes request_id      │
│    slot_occupied[16]      ← Injector sets 1, Consumer clears│
│    submit_ts[], complete_ts[], completed[] ← Timing data    │
│    decode_corrections[], decode_logical_pred[] ← Results    │
└─────────────────────────────────────────────────────────────┘
```

### Why Mapped Pinned Memory?

Traditional `cudaMemcpyAsync` requires explicit DMA transfers with stream synchronization overhead. Mapped pinned memory (`cudaHostAllocMapped`) gives the GPU a device-accessible pointer to host memory. GPU writes travel over NVLink-C2C (Grace Blackwell) and become visible to the CPU after appropriate libcu++ atomic fencing. This eliminates all explicit copy calls from the GPU→CPU output path.

Input data is still copied via explicit `cudaMemcpyAsync` D2D (ring buffer slot → TRT input buffer) because the ring buffer is reused while the GPU may still be processing.

---

## 7. Backpressure Protocol

Backpressure prevents the producer from overwriting ring buffer slots that are still in use. It operates through **slot availability**:

### Ring Buffer Level (Primary)

The `RingBufferInjector::try_submit()` checks if both `rx_flags[slot] == 0` AND `tx_flags[slot] == 0` before writing. If either is non-zero, the slot is busy:
- `rx_flags != 0`: Dispatcher hasn't consumed the slot yet.
- `tx_flags != 0`: Either IN_FLIGHT (`0xEEEE`) or completed (response addr) but not yet harvested by consumer.

The blocking `submit()` spins with `QEC_CPU_RELAX()` and increments a `backpressure_stalls` counter.

### Worker Level (Implicit)

If all `idle_mask` bits are 0 (all workers busy), the dispatcher spins on the current slot without advancing. This provides natural backpressure since `rx_flags[slot]` remains non-zero, preventing the producer from overwriting that slot.

---

## 8. Memory Ordering & Synchronization

The pipeline involves three independent agents (Producer, GPU, CPU workers/consumer) communicating through shared memory. All synchronization uses **libcu++ system-scope atomics** — no `volatile`, no `__threadfence_system()`.

### GPU → CPU (Signal Kernel → Worker Poll)

| Agent | Operation | Ordering Primitive |
|-------|-----------|-------------------|
| GPU | Write `h_predecoder_outputs_` (DMA copy in graph) | (ordered by graph node dependencies) |
| GPU | `ready_flags[0].store(1, release)` | system-scope atomic release |
| CPU Worker | `ready_flags[0].compare_exchange_strong(1, 2, acquire, relaxed)` | acquire on success, relaxed on failure |
| CPU Worker | Read `h_predecoder_outputs_` | (safe: ordered after acquire) |

### CPU → GPU (Job Release → Stream Reuse)

| Agent | Operation | Ordering Primitive |
|-------|-----------|-------------------|
| CPU Worker | Copy output to deferred buffer | (normal stores) |
| CPU Worker | `ready_flags[0].store(0, release)` | release ensures copy visible |
| GPU | `ready_flags[0].load(...)` sees 0 | GPU can write new results |

### Worker → Consumer (tx_flags)

| Agent | Operation | Ordering Primitive |
|-------|-----------|-------------------|
| PyMatch Worker | Write RPC response to ring buffer | (normal stores) |
| PyMatch Worker | `tx_flags[slot].store(addr, release)` | release ensures response visible |
| Consumer | `tx_flags[slot].load(acquire)` | acquire sees response data |

### Consumer → Producer (Slot Recycling)

| Agent | Operation | Ordering Primitive |
|-------|-----------|-------------------|
| Consumer | `slot_occupied[slot] = 0` | (normal store) |
| Consumer | `__sync_synchronize()` | full barrier |
| Consumer | `tx_flags[slot].store(0)`, `rx_flags[slot].store(0)` | slot free |
| Producer | `slot_available()` checks both flags == 0 | can reuse |

---

## 9. CUDA Graph Structure

Each predecoder has a pre-captured, host-launched CUDA graph:

```
 ┌──────────────────────────────────────────────────────┐
 │               Pre-Launch (host-side callback)         │
 │  pre_launch_fn:                                      │
 │    h_ring_ptrs[0] = slot_dev_ptr                     │
 │    cudaMemcpyAsync(d_trt_input,                      │
 │                    slot_dev + 24,   ← CUDAQ_RPC_HEADER_SIZE
 │                    input_size, D2D, stream)           │
 └──────────────────────┬───────────────────────────────┘
                        │
 ┌──────────────────────▼───────────────────────────────┐
 │               CUDA Graph (captured once)              │
 │                                                      │
 │  Node 1: TRT enqueueV3                               │
 │          (or passthrough_copy_kernel in SKIP_TRT)    │
 │                        │                             │
 │  Node 2: cudaMemcpyAsync D2D                         │
 │          d_trt_output_ → h_predecoder_outputs_ (mapped) │
 │                        │                             │
 │  Node 3: predecoder_signal_ready_kernel<<<1,1>>>     │
 │          ready_flags.store(1, release)                │
 └──────────────────────────────────────────────────────┘
```

The graph is instantiated with `cudaGraphInstantiate(&graph_exec_, graph, 0)` for host-launch mode. No device-side graph launch is used.

---

## 10. Pipeline Configurations

The test supports multiple surface code distances via the `PipelineConfig` struct. Model dimensions are derived automatically from TRT engine bindings:

| Config | Distance | Rounds | ONNX Model | Input (uint8) | Output (uint8) | Predecoders | PyMatch Workers | Slot Size |
|--------|----------|--------|------------|--------------|----------------|-------------|-----------------|-----------|
| `d7_r7` | 7 | 7 | `model1_d7_r7_unified_Z_batch1.onnx` | 504 | 505 | 16 | 32 | 1,024 |
| `d13_r13` | 13 | 13 | `predecoder_memory_d13_T13_X.onnx` | 3,276 | 3,277 | 16 | 32 | 4,096 |
| `d13_r104` | 13 | 104 | `predecoder_memory_d13_T104_X.onnx` | 17,472 | 17,473 | 8 | 16 | 32,768 |
| `d21_r21` | 21 | 21 | `model1_d21_r21_unified_X_batch1.onnx` | 13,860 | 13,861 | 16 | 32 | 16,384 |
| `d31_r31` | 31 | 31 | `model1_d31_r31_unified_Z_batch1.onnx` | 44,640 | 44,641 | 16 | 32 | 65,536 |

All models use **uint8** tensors for both input (detectors) and output (logical prediction + residual detectors).

The `slot_size` is computed as `round_up_pow2(CUDAQ_RPC_HEADER_SIZE + model_input_bytes)`.

Usage:

```bash
./test_realtime_predecoder_w_pymatching d7          # default
./test_realtime_predecoder_w_pymatching d13
./test_realtime_predecoder_w_pymatching d13_r104 104 20  # 104 µs rate, 20 sec
./test_realtime_predecoder_w_pymatching d21
./test_realtime_predecoder_w_pymatching d31
```

Optional flags:
- `--data-dir /path/to/stim/data`: Load real test data for correctness verification.

### Engine Caching

On first run with a given configuration, the ONNX model is compiled to a TensorRT engine and saved alongside the ONNX file (e.g., `predecoder_memory_d13_T104_X.engine`). Subsequent runs detect the cached engine and skip the build phase.

---

## 11. File Inventory

| File | Layer | Purpose |
|------|-------|---------|
| `realtime/include/.../cudaq_realtime.h` | API | C API header: structs, enums, ring buffer helpers, `CUDAQ_RPC_HEADER_SIZE` |
| `realtime/include/.../dispatch_kernel_launch.h` | API | RPC protocol structs (RPCHeader, RPCResponse), FNV-1a hash |
| `realtime/include/.../host_dispatcher.h` | API | Host dispatcher C API: `cudaq_host_dispatcher_config_t`, `cudaq_host_dispatch_worker_t` |
| `realtime/lib/.../host_dispatcher.cu` | Runtime | Host-side dispatcher loop implementation |
| `realtime/lib/.../cudaq_realtime_api.cpp` | Runtime | Ring buffer C API implementation |
| `libs/qec/include/.../pipeline.h` | Pipeline | `RealtimePipeline`, `RingBufferInjector`, callbacks, `DEFERRED_COMPLETION` |
| `libs/qec/lib/.../realtime_pipeline.cu` | Pipeline | Pipeline implementation: `RingBufferManager`, worker/consumer loops, injector |
| `libs/qec/include/.../ai_decoder_service.h` | QEC | Base class header: TRT lifecycle, dynamic tensor bindings, engine caching |
| `libs/qec/lib/.../ai_decoder_service.cu` | QEC | Base class impl: ONNX build, engine save/load, graph capture |
| `libs/qec/include/.../ai_predecoder_service.h` | QEC | Derived class header: CPU handoff, `poll_next_job`/`release_job` |
| `libs/qec/lib/.../ai_predecoder_service.cu` | QEC | Derived class impl: signal kernel, output DMA, graph capture |
| `libs/qec/include/.../nvtx_helpers.h` | Util | NVTX profiling macros (`NVTX_PUSH`, `NVTX_POP`) |
| `libs/qec/lib/.../test_realtime_predecoder_w_pymatching.cpp` | Test | End-to-end benchmark with real ONNX + PyMatching + correctness verification |
| `libs/qec/unittests/test_realtime_pipeline.cu` | Test | GTest unit/integration tests (21 tests, SKIP_TRT mode) |

---

## 12. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_SLOTS` | 16 | Ring buffer slot count |
| `slot_size` | Per-config (1024 - 65536) | Max payload per slot (derived from model input size) |
| `num_predecoders` | 8 (d13_r104) | Parallel predecoder instances = pipeline worker threads |
| `queue_depth` | 1 | Single in-flight inference per predecoder |
| `num_decode_workers` | 16 (d13_r104) | PyMatching thread pool size |
| `rate_us` | 104 | Inter-arrival time in microseconds |
| `duration_s` | 20 | Test duration in seconds |
| `warmup_count` | 20 | Requests excluded from latency stats |
| `max_requests` | 500,000 | Maximum requests per run |

### Capacity Analysis

- **Ring buffer**: 16 slots, each up to 32 KB for d13_r104.
- **GPU throughput**: 8 parallel streams × ~88 µs compute = ~90k req/s theoretical (far exceeds demand).
- **CPU throughput**: 16 PyMatching workers × ~224 µs decode = ~71k req/s theoretical.
- **Bottleneck**: PyMatching at 224 µs average, but 16 workers provide sufficient aggregate throughput for the 9.6k req/s demand at 104 µs inter-arrival.
- **Backpressure**: ~6.2M stalls over 20 s (noise floor of sub-microsecond spins when next round-robin slot is briefly busy).

---

## 13. Performance Benchmarking

### Pipeline Results (d=13, T=104, 104 µs rate, 20s)

Configuration: 16 slots, 8 predecoders, 16 PyMatching workers, Stim test data.

| Metric | Value |
|--------|-------|
| Throughput | 9,610 req/s |
| Submitted / Completed | 192,309 / 192,309 |
| Backpressure stalls | 6,193,418 |
| p50 latency | 352.3 µs |
| Mean latency | 393.4 µs |
| p90 latency | 514.7 µs |
| p95 latency | 606.3 µs |
| p99 latency | 1,248.8 µs |
| Max latency | 3,930.0 µs |
| stddev | 179.0 µs |

### Worker Timing Breakdown

| Component | Avg Time |
|-----------|----------|
| PyMatching decode | 224.4 µs |
| Total worker (PyMatch thread) | 224.5 µs |
| Worker overhead | 0.1 µs |

### Syndrome Density

| | Avg nonzero / total | Density |
|-|---------------------|---------|
| Input detectors | 932.7 / 17,472 | 5.34% |
| Output residual detectors | 16.1 / 17,472 | 0.09% |
| **Reduction** | | **98.3%** |

### Logical Error Rate

| Mode | LER | Mismatches |
|------|-----|------------|
| Full pipeline (predecoder + PyMatching) | **0.0020** | 384 / 192,309 |
| Predecoder only (output[0]) | 0.3980 | 76,537 / 192,309 |

### Raw TRT Baseline (trtexec)

| Mode | GPU Compute | Total Host Latency |
|------|-------------|-------------------|
| Default (single stream) | 107 µs | 119 µs |
| CUDA Graph + SpinWait | 90 µs | 99 µs |
| CUDA Graph + SpinWait + No Transfers | 88 µs | 88 µs |

### NVTX Profiling (per-stage timing)

| Stage | Avg (µs) | Median (µs) |
|-------|----------|-------------|
| PyMatchDecode | 277 | 223 |
| PreLaunchCopy | 8.8 | 8.3 |
| ConsumerComplete | 3.3 | 3.2 |
| Submit | 2.8 | 2.7 |
| PollJob | 2.3 | 1.9 |
| ReleaseJob | 2.0 | 1.9 |

Infrastructure overhead (ring buffer + dispatch + poll + consumer): **~18 µs per request**.

---

## 14. Portability

### Architecture Support

| Feature | x86_64 | aarch64 (Grace Blackwell) |
|---------|--------|----------------------|
| `QEC_CPU_RELAX()` | `_mm_pause()` | `asm volatile("yield")` |
| Cross-device atomics | libcu++ system-scope | libcu++ system-scope |
| Memory model | TSO (strong) | Weakly ordered (requires fences) |
| Interconnect | PCIe | NVLink-C2C |

The `QEC_CPU_RELAX()` macro is defined in both `ai_predecoder_service.h` and `host_dispatcher.h` and should be used by all polling code.

### CUDA Compute Capability

| Feature | Minimum |
|---------|---------|
| `cudaHostAllocMapped` | All CUDA devices |
| CUDA Graphs (host launch) | sm_50+ |
| libcu++ system-scope atomics | sm_70+ |

---

## 15. Limitations & Future Work

1. **PyMatching is the bottleneck**: At 224 µs average, PyMatching consumes 93% of CPU-stage time. A faster MWPM decoder (e.g., Fusion Blossom, GPU-accelerated matching) would directly reduce pipeline latency.

2. **Round-robin slot injection**: The `RingBufferInjector` uses strict round-robin slot assignment. If slot N is busy, the producer stalls even if slot N+1 is free. Out-of-order slot allocation would reduce backpressure but sacrifice FIFO ordering.

3. **Single data type**: The current test assumes uint8 detectors matching the predecoder model. Support for INT32 models would require element-size-aware input packing.

4. **Static TRT shapes only**: The current implementation assumes static input/output tensor shapes. Dynamic shapes would require per-invocation shape metadata in the RPC payload and runtime TRT profile switching.

5. **No queue drain on shutdown**: The PyMatching queue is shut down immediately; jobs that were enqueued but not yet decoded are silently dropped. A production system should drain the queue before stopping.

6. **Core pinning is advisory**: The pipeline pins threads to cores via `sched_setaffinity`, but does not isolate cores from the OS scheduler. A production deployment should use `isolcpus` or cgroups.

7. **INT8 quantization**: The predecoder model runs in FP16. INT8 quantization could reduce GPU compute from 88 µs to ~50 µs, though the GPU is not currently the bottleneck.

8. **Sparse PyMatching input**: The predecoder reduces syndrome density to 0.09%. Representing the sparse residual as a list of nonzero indices (rather than a dense vector) could speed up PyMatching's graph traversal.
