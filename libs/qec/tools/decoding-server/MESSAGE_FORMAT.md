# Decoding Server Message Format

Byte-level reference for the three data-plane RPCs the decoding server
accepts: `enqueue_syndromes`, `get_corrections`, and `reset_decoder`.

Source of truth in code:

| File | Contents |
|---|---|
| `libs/qec/include/cudaq/qec/realtime/decoder_rpc_wire_format.h` | Function IDs, payload structs, `RpcStatus`, size `static_assert`s |
| `libs/qec/lib/realtime/decoding-server-cqr/RpcSlot.h` | Request parsers and response writers |
| `libs/qec/lib/realtime/decoding-server-cqr/decoding_server_cqr.cpp` | Routing and handler-schema registration |
| `libs/qec/lib/realtime/decoding-server-cqr/DecodingSession.cpp` | Per-message semantics and status codes |
| `libs/qec/lib/realtime/simulation-cqr/simulation_cqr_device.cpp` | Client side: kernel API → wire |

## Conventions

- All scalars are little-endian.
- Payloads are tightly packed: no padding between fields, no trailing padding.
  `arg_len` / `result_len` are exact byte counts.
- 64-bit integers are registered as `CUDAQ_TYPE_INT64` (8 bytes) and
  interpreted as unsigned by the server. A negative `decoder_id` resolves to
  `INVALID_DECODER`.
- `bool` is `CUDAQ_TYPE_UINT8`: `0x00` = false, non-zero = true; send `0x01`.
- `CUDAQ_TYPE_BIT_PACKED` (`std::vector<bool>`): bit `i` is bit `i mod 8` of
  byte `i / 8` (LSB-first); unused high bits of the last byte are zero;
  `ceil(N/8)` data bytes. As a **request argument** it is preceded by a
  `uint64` element count `N` (this is what the `cc.device_call` lowering
  emits). As a **response result** there is no prefix; `result_len` is the
  byte count.
- `function_id = fnv1a_32(<callee name>)`.

## Framing

Request = 24-byte `RPCHeader` + payload. Response = 24-byte `RPCResponse` +
result payload (if any). Both from cudaq-realtime's `rpc_wire_format.h`.

```text
Offset   | Size | Request        | Response
---------+------+----------------+---------------------------------------
 0 ..  3 | 4    | magic          | magic       0x43555152 req / 0x43555153 resp
 4 ..  7 | 4    | function_id    | status      int32, 0 = OK
 8 .. 11 | 4    | arg_len        | result_len  0 on any error
12 .. 15 | 4    | request_id     | request_id  echoed
16 .. 23 | 8    | ptp_timestamp  | ptp_timestamp  echoed (0 if unused)
```

The server fills the response fields first and release-stores `magic` last.
Every request, including the fire-and-forget ones, gets a response; callers
that do not need it simply do not wait.

| RPC | `function_id` | `arg_len` | Result payload |
|---|---|---|---|
| `enqueue_syndromes` | `0x7ED8BE82` | `32 + ceil(N/8)` | none |
| `get_corrections` | `0x882D5BA1` | `17` | `ceil(R/8)` bytes |
| `reset_decoder` | `0x977A59CF` | `8` | none |

## `enqueue_syndromes`

Kernel API: `void enqueue_syndromes(uint64_t decoder_id, const std::vector<measure_result>& syndromes, uint64_t tag)`.
The wrapper emits the on-wire order `(decoder_id, counter = tag, syndrome_mapping_id = 0, syndrome_bits)`.

Schema: `args = { INT64, INT64, INT64, BIT_PACKED }`, no results.

`N` = syndrome bits, `B = ceil(N/8)`:

```text
Offset       | Size | Field
-------------+------+---------------------------------------------------
24 .. 31     | 8    | decoder_id
32 .. 39     | 8    | counter             application breadcrumb (`tag`)
40 .. 47     | 8    | syndrome_mapping_id must be 0
48 .. 55     | 8    | element count == N  (the vector's own prefix)
56 .. 56+B-1 | B    | syndrome bits       LSB-first
```

Status:

| Condition | `status` |
|---|---|
| Accepted (not necessarily decoded) | `OK` |
| `arg_len < 32 + B`, `N == 0`, `N > 2^20`, or `syndrome_mapping_id != 0` | `BAD_REQUEST`, session latched `failed` |

A decoder-side failure after acceptance cannot be reported here. The session
latches `failed` and the next `get_corrections` returns `INTERNAL_ERROR`; only
`reset_decoder` clears it.

## `get_corrections`

Kernel API: `std::vector<bool> get_corrections(uint64_t decoder_id, uint64_t return_size, bool reset)`.
The OUT vector's length is what the lowering sends as `return_size`.

Schema: `args = { INT64, INT64, UINT8 }`, `results = { BIT_PACKED }`.

Request (41 bytes total):

```text
Offset   | Size | Field
---------+------+-----------------------------------------
24 .. 31 | 8    | decoder_id
32 .. 39 | 8    | return_size == R
40       | 1    | reset        0x00 / 0x01, not padded
```

Response, `C = ceil(R/8)`:

```text
Offset       | Size | Field
-------------+------+-----------------------------------------
24 .. 24+C-1 | C    | corrections  LSB-first, no count prefix
```

| Condition | `status` | `result_len` |
|---|---|---|
| Decode complete, `R` == decoder observable count | `OK` | `C` |
| `arg_len < 17`, `R <= 0`, or `R != observable count` | `BAD_REQUEST` | 0 |
| No decode completed yet in this epoch | `NOT_READY` | 0 |
| Session latched `failed`, or decoder threw | `INTERNAL_ERROR` | 0 |

`reset = 1` clears the correction bits after copying them out (decoder
`clear_corrections()`); queued syndromes and decoder state survive.

Example: `decoder_id = 1`, `return_size = 3`, `reset = 1`, `request_id = 7`;
corrections `{1, 0, 1}`:

```text
req:  52 51 55 43  A1 5B 2D 88  11 00 00 00  07 00 00 00  00×8
      01 00 00 00 00 00 00 00   03 00 00 00 00 00 00 00   01
resp: 53 51 55 43  00 00 00 00  01 00 00 00  07 00 00 00  00×8
      05
```

## `reset_decoder`

Kernel API: `void reset_decoder(uint64_t decoder_id)`. Full reset: clears
queued syndromes, corrections, and the `failed` latch.

Schema: `args = { INT64 }`, no results.

```text
Offset   | Size | Field
---------+------+------------
24 .. 31 | 8    | decoder_id
```

| Condition | `status` |
|---|---|
| Reset succeeded | `OK` |
| `arg_len < 8` | `BAD_REQUEST` |
| Decoder threw | `INTERNAL_ERROR`, session latched `failed` |

## Routing and status codes

Before parsing, the dispatcher checks `magic`, reads payload bytes 0..7 as
`decoder_id` to find the session, then dispatches on `function_id`. Any new
message must keep `decoder_id` at offset 24.

| `status` | Value | Meaning |
|---|---|---|
| `OK` | 0 | Success |
| `INVALID_DECODER` | 1 | `decoder_id` not registered |
| `BAD_REQUEST` | 2 | Bad magic, unknown `function_id`, or the per-RPC checks above |
| `INTERNAL_ERROR` | 3 | Decoder exception, or latched enqueue failure |
| `NOT_READY` | 4 | `get_corrections` before a decode completed |
| `BUSY` | 5 | Reserved; not emitted by the current server |
| `SYNDROMES_DROPPED` | 6 | Reserved; not emitted by the current server |

`arg_len` is validated as a lower bound; trailing bytes beyond the payload are
ignored. Senders should still emit exact lengths.
