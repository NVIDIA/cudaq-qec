# Autonomous Decoder Developer Guide

This guide explains how to implement a custom quantum error correction (QEC) decoder that integrates with the CUDA-Q realtime dispatch system using the `autonomous_decoder` CRTP (Curiously Recurring Template Pattern) interface.

## Table of Contents

1. [Overview](#overview)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Creating a Custom Decoder](#creating-a-custom-decoder)
4. [Integrating with RPC Dispatch](#integrating-with-rpc-dispatch)
5. [Testing Your Decoder](#testing-your-decoder)
6. [Example: Mock Decoder](#example-mock-decoder)

---

## Overview

The `autonomous_decoder` framework enables zero-CPU-overhead QEC decoding by running decoders entirely on GPU, integrated with CUDA-Q's realtime dispatch kernel. Key features:

- **Zero-CPU data path**: Measurements arrive from FPGA/quantum hardware directly to GPU memory
- **Event-driven**: Decoders respond to incoming RPC messages via the dispatch kernel
- **CRTP-based polymorphism**: Compile-time polymorphism for zero-overhead dispatch
- **CUDA Graphs support**: Compatible with device-launched graphs for ultra-low latency

---

## Understanding the Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Quantum Hardware / FPGA                  │
└────────────────────┬────────────────────────────────────────┘
                     │ (measurements)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               GPU Ring Buffer (Host-Mapped)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Dispatch Kernel (Persistent, GPU-side)             │
│  • Polls RX flags                                            │
│  • Routes to decoder via function_id                         │
│  • Calls RPC handler                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│      Your Decoder RPC Handler (Device Function)              │
│  • Calls autonomous_decoder::decode()                        │
│  • Returns corrections via RPC response                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│    Your Decoder Implementation (autonomous_decoder<T>)       │
│  • decode_impl() - your custom decoding logic                │
│  • Device-side only, no CPU involvement                      │
└─────────────────────────────────────────────────────────────┘
```

### The CRTP Pattern

```cpp
template <typename Derived>
class autonomous_decoder {
public:
  __device__ void decode(const uint8_t* measurements,
                         uint8_t* corrections,
                         std::size_t num_measurements,
                         std::size_t num_observables) {
    // CRTP dispatch to derived class
    static_cast<Derived*>(this)->decode_impl(
        measurements, corrections, num_measurements, num_observables);
  }
};
```

Your decoder inherits from `autonomous_decoder<YourDecoder>` and implements `decode_impl()`.

---

## Creating a Custom Decoder

### Step 1: Define Your Decoder Context

Create a context structure to hold your decoder's state (matrices, parameters, lookup tables, etc.):

```cpp
// In: include/cudaq/qec/realtime/my_decoder_context.h

#pragma once
#include "cudaq/qec/realtime/decoder_context.h"

namespace cudaq::qec::realtime {

/// @brief Context for MyDecoder
struct my_decoder_context : public decoder_context_base {
  // Add decoder-specific data here
  const float *parity_check_matrix = nullptr;  // Example: H matrix
  const float *edge_weights = nullptr;         // Example: MWPM weights
  float temperature = 0.0f;                    // Example: belief prop parameter
  
  // Device-side scratch buffers (if needed)
  float *syndrome_buffer = nullptr;
  float *belief_buffer = nullptr;
};

} // namespace cudaq::qec::realtime
```

**Key points:**
- Inherit from `decoder_context_base` to get standard fields (D/O matrices, dimensions)
- Add pointers to device memory for decoder-specific data
- Keep this structure POD (Plain Old Data) for easy host↔device transfer

### Step 2: Implement Your Decoder Class

```cpp
// In: include/cudaq/qec/realtime/my_decoder.cuh

#pragma once
#include "cudaq/qec/realtime/autonomous_decoder.cuh"
#include "cudaq/qec/realtime/my_decoder_context.h"

namespace cudaq::qec::realtime {

/// @brief Custom QEC decoder using [algorithm name]
class my_decoder : public autonomous_decoder<my_decoder> {
public:
  /// @brief Constructor taking a reference to context
  __device__ __host__ explicit my_decoder(my_decoder_context& ctx)
      : ctx_(ctx) {}

  /// @brief Core decode implementation (required by CRTP)
  __device__ void decode_impl(const uint8_t* __restrict__ measurements,
                              uint8_t* __restrict__ corrections,
                              std::size_t num_measurements,
                              std::size_t num_observables);

  /// @brief Access to context
  __device__ __host__ my_decoder_context& context() { return ctx_; }
  __device__ __host__ const my_decoder_context& context() const { return ctx_; }

private:
  my_decoder_context& ctx_;
  
  // Optional: Add helper methods
  __device__ void compute_syndrome(const uint8_t* measurements, 
                                   float* syndrome);
  __device__ void decode_syndrome(const float* syndrome, 
                                  uint8_t* corrections);
};

} // namespace cudaq::qec::realtime
```

### Step 3: Implement the Decoder Logic

```cpp
// In: lib/realtime/my_decoder.cu

#include "cudaq/qec/realtime/my_decoder.cuh"

namespace cudaq::qec::realtime {

__device__ void my_decoder::decode_impl(
    const uint8_t* __restrict__ measurements,
    uint8_t* __restrict__ corrections,
    std::size_t num_measurements,
    std::size_t num_observables) {
  
  // Example: Simple single-threaded implementation
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    
    // Step 1: Compute syndrome from measurements
    compute_syndrome(measurements, ctx_.syndrome_buffer);
    
    // Step 2: Run your decoding algorithm
    // (e.g., MWPM, belief propagation, neural network, etc.)
    decode_syndrome(ctx_.syndrome_buffer, corrections);
    
    // Step 3: Corrections are written to output buffer
  }
  
  // For multi-threaded implementations, add __syncthreads() as needed
}

__device__ void my_decoder::compute_syndrome(
    const uint8_t* measurements, 
    float* syndrome) {
  // Example: Multiply measurements by parity check matrix
  for (std::size_t i = 0; i < ctx_.num_detectors; ++i) {
    float sum = 0.0f;
    // Use CSR format from ctx_.D_row_ptr, ctx_.D_col_idx
    for (std::uint32_t j = ctx_.D_row_ptr[i]; 
         j < ctx_.D_row_ptr[i + 1]; ++j) {
      std::uint32_t col = ctx_.D_col_idx[j];
      sum += measurements[col];
    }
    syndrome[i] = fmodf(sum, 2.0f); // Mod 2 for binary
  }
}

__device__ void my_decoder::decode_syndrome(
    const float* syndrome, 
    uint8_t* corrections) {
  // Your algorithm here!
  // Examples:
  // - MWPM: Minimum-weight perfect matching
  // - BP: Belief propagation
  // - UF: Union-find
  // - NN: Neural network inference
  
  for (std::size_t i = 0; i < ctx_.num_observables; ++i) {
    corrections[i] = 0; // Placeholder - implement your logic
  }
}

} // namespace cudaq::qec::realtime
```

**Performance tips:**
- For large problems, parallelize across threads/blocks
- Use shared memory for intermediate results
- Consider cooperative groups for complex synchronization
- Profile with `nsight-compute` to optimize memory access patterns

---

## Integrating with RPC Dispatch

### Step 4: Create RPC Handler Functions

```cpp
// In: lib/realtime/my_decoder.cu (continued)

namespace cudaq::qec::realtime {

//==============================================================================
// Global Device Decoder Instance
//==============================================================================

__device__ my_decoder* g_my_decoder = nullptr;

/// @brief Set the decoder instance from host
inline void set_my_decoder(my_decoder* decoder) {
  cudaMemcpyToSymbol(g_my_decoder, &decoder, sizeof(my_decoder*));
}

//==============================================================================
// RPC Handler for Regular Dispatch
//==============================================================================

/// @brief RPC-compatible wrapper for dispatch kernel
__device__ int my_decode_rpc(void* buffer, 
                             std::uint32_t arg_len,
                             std::uint32_t max_result_len,
                             std::uint32_t* result_len) {
  
  if (g_my_decoder == nullptr) {
    *result_len = 0;
    return -1; // Error: decoder not initialized
  }

  // Input measurements and output corrections share the buffer
  uint8_t* measurements = static_cast<uint8_t*>(buffer);
  uint8_t* corrections = static_cast<uint8_t*>(buffer);

  const auto& ctx = g_my_decoder->context();
  
  // Call the CRTP decoder interface
  g_my_decoder->decode(measurements, corrections, 
                       ctx.num_measurements, ctx.num_observables);

  *result_len = static_cast<std::uint32_t>(ctx.num_observables);
  return 0; // Success
}

/// @brief Get device function pointer for RPC registration
__device__ auto get_my_decode_rpc_ptr() { 
  return &my_decode_rpc; 
}

//==============================================================================
// RPC Handler for Graph-Based Dispatch (Optional, for sm_90+)
//==============================================================================

__global__ void my_decode_graph_kernel(void** buffer_ptr) {
  void* data_buffer = (buffer_ptr != nullptr) ? *buffer_ptr : nullptr;
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (data_buffer == nullptr || g_my_decoder == nullptr) {
      if (data_buffer != nullptr) {
        // Write error response
        auto* response = static_cast<cudaq::nvqlink::RPCResponse*>(data_buffer);
        response->magic = cudaq::nvqlink::RPC_MAGIC_RESPONSE;
        response->status = -1;
        response->result_len = 0;
      }
      return;
    }

    // Parse RPC header
    auto* header = static_cast<cudaq::nvqlink::RPCHeader*>(data_buffer);
    void* arg_buffer = static_cast<void*>(header + 1);

    // Decode
    uint8_t* measurements = static_cast<uint8_t*>(arg_buffer);
    uint8_t* corrections = static_cast<uint8_t*>(arg_buffer);
    
    const auto& ctx = g_my_decoder->context();
    g_my_decoder->decode(measurements, corrections, 
                         ctx.num_measurements, ctx.num_observables);

    // Write response
    auto* response = static_cast<cudaq::nvqlink::RPCResponse*>(data_buffer);
    response->magic = cudaq::nvqlink::RPC_MAGIC_RESPONSE;
    response->status = 0;
    response->result_len = static_cast<std::uint32_t>(ctx.num_observables);
  }
}

} // namespace cudaq::qec::realtime
```

### Step 5: Register Your Decoder with the Function Table

The function table is initialized in a device kernel to properly capture device function pointers:

```cpp
// In your source file (e.g., my_decoder.cu)

#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

// Define function ID (use FNV-1a hash of a unique name)
constexpr std::uint32_t MY_DECODE_FUNCTION_ID = 
    cudaq::nvqlink::fnv1a_hash("my_decoder");

/// @brief Initialize the device function table with your decoder
__global__ void init_my_decoder_function_table(cudaq_function_entry_t* entries) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Set handler pointer and dispatch mode
    entries[0].handler.device_fn_ptr = 
        reinterpret_cast<void*>(&cudaq::qec::realtime::my_decode_rpc);
    entries[0].function_id = MY_DECODE_FUNCTION_ID;
    entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    entries[0].reserved[0] = 0;
    entries[0].reserved[1] = 0;
    entries[0].reserved[2] = 0;

    // Define schema (example: 128-bit bit-packed input, 1-byte output)
    entries[0].schema.num_args = 1;
    entries[0].schema.num_results = 1;
    entries[0].schema.reserved = 0;
    
    // Argument 0: bit-packed detection events (128 bits = 16 bytes)
    entries[0].schema.args[0].type_id = CUDAQ_TYPE_BIT_PACKED;
    entries[0].schema.args[0].reserved[0] = 0;
    entries[0].schema.args[0].reserved[1] = 0;
    entries[0].schema.args[0].reserved[2] = 0;
    entries[0].schema.args[0].size_bytes = 16;      // 128 bits
    entries[0].schema.args[0].num_elements = 128;   // 128 bits
    
    // Result 0: correction byte
    entries[0].schema.results[0].type_id = CUDAQ_TYPE_UINT8;
    entries[0].schema.results[0].reserved[0] = 0;
    entries[0].schema.results[0].reserved[1] = 0;
    entries[0].schema.results[0].reserved[2] = 0;
    entries[0].schema.results[0].size_bytes = 1;
    entries[0].schema.results[0].num_elements = 1;
  }
}

// In your host-side setup code:

// Allocate device memory for function table
cudaq_function_entry_t* d_function_entries;
cudaMalloc(&d_function_entries, sizeof(cudaq_function_entry_t));

// Initialize function table on device
init_my_decoder_function_table<<<1, 1>>>(d_function_entries);
cudaDeviceSynchronize();

// Register with dispatcher
cudaq_function_table_t table;
table.entries = d_function_entries;
table.count = 1;
cudaq_dispatcher_set_function_table(dispatcher, &table);
```

**Key points:**
- Function table initialization **must** happen on the device (in a kernel) to properly capture device function pointers
- Initialize all `reserved` fields to 0
- Set `dispatch_mode` for each entry (`CUDAQ_DISPATCH_DEVICE_CALL` or `CUDAQ_DISPATCH_GRAPH_LAUNCH`)
- Schema describes the exact layout of arguments and results

#### Graph-Based Dispatch Setup (sm_90+)

For graph-based dispatch, the function table setup is different because `cudaGraphExec_t` is a host-side handle:

```cpp
// 1. Create CUDA graph with your decoder kernel
cudaStream_t capture_stream;
cudaStreamCreate(&capture_stream);

// Allocate buffer pointer for pointer indirection pattern
void** d_buffer_ptr;
cudaMalloc(&d_buffer_ptr, sizeof(void*));
cudaMemset(d_buffer_ptr, 0, sizeof(void*));

// Capture the graph
cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
my_decode_graph_kernel<<<1, 1, 0, capture_stream>>>(d_buffer_ptr);
cudaStreamEndCapture(capture_stream, &graph);

// Instantiate with device launch flag (required for device-side graph launch)
cudaGraphExec_t graph_exec;
cudaGraphInstantiateWithFlags(&graph_exec, graph, 
                              cudaGraphInstantiateFlagDeviceLaunch);

// Upload graph to device
cudaGraphUpload(graph_exec, capture_stream);
cudaStreamSynchronize(capture_stream);
cudaStreamDestroy(capture_stream);

// 2. Set up function table entry on HOST (graph_exec is a host handle)
cudaq_function_entry_t* d_function_entries;
cudaMalloc(&d_function_entries, sizeof(cudaq_function_entry_t));

cudaq_function_entry_t host_entry{};
host_entry.handler.graph_exec = graph_exec;  // Host-side graph handle
host_entry.function_id = MY_DECODE_FUNCTION_ID;
host_entry.dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
host_entry.reserved[0] = 0;
host_entry.reserved[1] = 0;
host_entry.reserved[2] = 0;

// Schema: same as device call mode
host_entry.schema.num_args = 1;
host_entry.schema.num_results = 1;
host_entry.schema.reserved = 0;
host_entry.schema.args[0].type_id = CUDAQ_TYPE_BIT_PACKED;
host_entry.schema.args[0].reserved[0] = 0;
host_entry.schema.args[0].reserved[1] = 0;
host_entry.schema.args[0].reserved[2] = 0;
host_entry.schema.args[0].size_bytes = 16;
host_entry.schema.args[0].num_elements = 128;
host_entry.schema.results[0].type_id = CUDAQ_TYPE_UINT8;
host_entry.schema.results[0].reserved[0] = 0;
host_entry.schema.results[0].reserved[1] = 0;
host_entry.schema.results[0].reserved[2] = 0;
host_entry.schema.results[0].size_bytes = 1;
host_entry.schema.results[0].num_elements = 1;

// Copy to device
cudaMemcpy(d_function_entries, &host_entry, sizeof(cudaq_function_entry_t),
           cudaMemcpyHostToDevice);

// 3. Register with dispatcher using graph-based dispatch API
cudaq_function_table_t table;
table.entries = d_function_entries;
table.count = 1;
cudaq_dispatcher_set_function_table(dispatcher, &table);
```

**Note:** Graph-based dispatch requires the dispatch kernel itself to run in a graph context. Use `cudaq_create_dispatch_graph_regular()` instead of standard `cudaq_dispatcher_start()`. See the CUDA-Q Realtime Host API documentation for details.

---

## Testing Your Decoder

### Step 6: Write Unit Tests

```cpp
// In: unittests/decoders/realtime/test_my_decoder.cu

#include <gtest/gtest.h>
#include "cudaq/qec/realtime/my_decoder.cuh"

class MyDecoderTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Allocate device context
    cudaMalloc(&d_ctx_, sizeof(cudaq::qec::realtime::my_decoder_context));
    
    // Initialize context with test data
    cudaq::qec::realtime::my_decoder_context ctx;
    ctx.num_measurements = 128;
    ctx.num_observables = 1;
    // ... set up matrices, buffers, etc.
    
    cudaMemcpy(d_ctx_, &ctx, sizeof(ctx), cudaMemcpyHostToDevice);
    
    // Allocate and initialize decoder
    cudaMalloc(&d_decoder_, sizeof(cudaq::qec::realtime::my_decoder));
    init_decoder_kernel<<<1, 1>>>(d_decoder_, d_ctx_);
    cudaDeviceSynchronize();
    
    // Set global decoder
    cudaq::qec::realtime::set_my_decoder(d_decoder_);
  }
  
  void TearDown() override {
    if (d_decoder_) cudaFree(d_decoder_);
    if (d_ctx_) cudaFree(d_ctx_);
  }
  
  cudaq::qec::realtime::my_decoder_context* d_ctx_ = nullptr;
  cudaq::qec::realtime::my_decoder* d_decoder_ = nullptr;
};

TEST_F(MyDecoderTest, BasicDecoding) {
  // Prepare test measurements
  std::vector<uint8_t> measurements(128);
  // ... fill with test data
  
  uint8_t* d_measurements;
  uint8_t* d_corrections;
  cudaMalloc(&d_measurements, 128);
  cudaMalloc(&d_corrections, 1);
  cudaMemcpy(d_measurements, measurements.data(), 128, cudaMemcpyHostToDevice);
  
  // Call decoder directly
  auto test_kernel = [=] __global__ () {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      cudaq::qec::realtime::g_my_decoder->decode(
          d_measurements, d_corrections, 128, 1);
    }
  };
  test_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  
  // Verify results
  uint8_t result;
  cudaMemcpy(&result, d_corrections, 1, cudaMemcpyDeviceToHost);
  EXPECT_EQ(result, expected_correction);
  
  cudaFree(d_measurements);
  cudaFree(d_corrections);
}
```

### Integration Testing

Follow the pattern in `test_realtime_decoding.cu`:
1. Set up ring buffers and dispatch kernel
2. Register your decoder's RPC handler
3. Send test measurements through the RPC protocol
4. Verify corrections returned via RPC responses

---

## Example: Mock Decoder

For a complete reference implementation, see:
- **Header**: `libs/qec/unittests/realtime/mock_decode_handler.cuh`
- **Implementation**: `libs/qec/unittests/realtime/mock_decode_handler.cu`
- **Context**: `libs/qec/include/cudaq/qec/realtime/decoder_context.h`
- **Tests**: `libs/qec/unittests/decoders/realtime/test_realtime_decoding.cu`

The mock decoder demonstrates:
- ✅ CRTP inheritance from `autonomous_decoder<mock_decoder>`
- ✅ Simple lookup-table based decoding
- ✅ RPC handler integration (both regular and graph modes)
- ✅ Complete test coverage with ring buffers

---

## Best Practices

### Memory Management
- Use device pointers in context structures, allocated once at setup
- Avoid dynamic allocation during decoding (pre-allocate scratch buffers)
- Consider using unified memory for easier debugging (switch to device memory for production)

### Performance
- Profile early and often with `nsight-compute`
- Optimize memory access patterns (coalesced reads/writes)
- Use shared memory for frequently accessed data
- Consider warp-level primitives for reductions

### Correctness
- Start with single-threaded implementation, parallelize later
- Write comprehensive unit tests before integration testing
- Test edge cases (all-zero syndromes, maximum-weight syndromes, etc.)
- Validate against CPU reference implementation

### Compatibility
- Support both regular and graph-based dispatch modes
- Maintain backward compatibility if modifying existing decoders
- Document any GPU architecture requirements (e.g., sm_90 for graphs)

---

## Additional Resources

- **UML Design Document**: `docs/autonomous_decoder_uml.puml`
- **CUDA-Q Realtime API**: `cuda-quantum/realtime/include/cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h`
- **Dispatch Kernel Source**: `cuda-quantum/realtime/lib/daemon/dispatcher/dispatch_kernel.cu`

For questions or issues, consult the CUDA-Q QEC development team.
