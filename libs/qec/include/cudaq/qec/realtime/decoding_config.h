/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "cudaq/qec/extended_dem.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec::decoding::config {

/// Dispatch shape for a decoder session -- HOW its RPCs are executed, not
/// which wire the bytes arrive on (the wire is a server-level transport
/// provider, selected independently).
/// host:         requests are dispatched on the CPU (HOST_CALL); works with
///               any transport provider (dev, CI, no GPU required).
/// device_graph: requests are dispatched on the GPU by the self-relaunching
///               device-graph scheduler (DeviceGraphTransceiver); requires a
///               decoder with a captured decode graph and a provider whose
///               rings are GPU-visible (e.g. GpuRoceTransceiver/DOCA).
/// YAML key: `dispatch: host|device_graph`.
enum class DecoderDispatch { host, device_graph };

/// @brief Decoder-specific constructor arguments, stored as a
/// `cudaqx::heterogeneous_map` -- the form every decoder's constructor
/// consumes. YAML conversion and key validation are driven by the parameter
/// schema the decoder registered (see cudaq/qec/decoder_config_schema.h), so
/// out-of-tree decoders participate without any framework changes. The
/// framework-owned `error_rate_vec` model input is also accepted here as a
/// legacy YAML alias, but is removed before plugin construction.
class decoder_custom_args_t {
public:
  decoder_custom_args_t() = default;
  decoder_custom_args_t(const cudaqx::heterogeneous_map &m) : map_(m) {}

  decoder_custom_args_t &operator=(const cudaqx::heterogeneous_map &m) {
    map_ = m;
    return *this;
  }

  cudaqx::heterogeneous_map &map() { return map_; }
  const cudaqx::heterogeneous_map &map() const { return map_; }
  bool empty() const { return map_.empty(); }

  /// Deep equality over the canonical custom-args value kinds.
  __attribute__((visibility("default"))) bool
  operator==(const decoder_custom_args_t &other) const;

private:
  cudaqx::heterogeneous_map map_;
};

/// @brief Configuration structure for decoder options.
struct decoder_config {
  int64_t id = 0;
  std::string type;
  /// Dispatch shape for this decoder's RPCs.  Defaults to host.  Set to
  /// device_graph for decoders where syndrome bits are DMA'd directly to GPU
  /// VRAM and decoded by a captured CUDA graph (e.g. nv_qldpc_decoder with
  /// RelayBP).
  DecoderDispatch dispatch = DecoderDispatch::host;
  /// CUDA device this decoder is pinned to at construction (see the
  /// "cuda_device_id" decoder parameter). Placement knob common to any
  /// GPU-accelerated decoder, hence at this level rather than inside the
  /// per-decoder custom args. Unset = unpinned.
  std::optional<int> cuda_device_id;
  /// The fields below describe the DEM three alternative ways, and exactly
  /// one of them applies:
  ///
  ///   - Flat form: H_sparse plus block_size, syndrome_size, O_sparse,
  ///     D_sparse and optional error_rate_vec, all sized for the whole
  ///     experiment.
  ///   - Chunk form: dem_chunks (which carries phases, connections, seam, and
  ///     num_rounds internally). The flat fields are derived by expanding the
  ///     phases, and must be omitted.
  ///   - DEM form: stim_dem_path, the Stim model text itself. The decoder
  ///     derives its dimensions, observable mapping and error rates from it.
  ///     H_sparse, O_sparse and error_rate_vec must be omitted; D_sparse is
  ///     still required, while block_size and syndrome_size are optional
  ///     checked assertions.
  ///
  /// See expand_dem_chunks() for the chunk-form derivation, which runs at
  /// decoder construction so the rest of the pipeline only ever sees the flat
  /// form. DEM form stays authoritative: the model text is what the decoder is
  /// built from.
  uint64_t block_size = 0;
  uint64_t syndrome_size = 0;
  /// Path to an authoritative Stim detector error model. Relative paths are
  /// resolved against the configuration file's directory, or the process
  /// working directory for a programmatic or raw-string configuration.
  std::string stim_dem_path;
  std::vector<std::int64_t> H_sparse;
  std::vector<std::int64_t> O_sparse;
  /// Maps raw measurements to detectors. Orthogonal to the model source and
  /// required by both.
  std::vector<std::int64_t> D_sparse;
  /// Error probability per H column. Input YAML also accepts the established
  /// `decoder_custom_args.error_rate_vec` spelling, and emitted YAML uses that
  /// spelling to preserve the researcher-facing wire format.
  std::vector<double> error_rate_vec;
  /// Optional per-phase DEM for a streaming, repeated-round decomposition.
  /// H_sparse above describes the whole experiment as one flat matrix; these
  /// phases describe one round each so the round count can be chosen at run
  /// time. num_rounds lives inside dem_chunks_spec. See
  /// cudaq::qec::dem_chunks_from_spec() for expansion to a chunk sequence.
  ///
  /// A configuration that also has a nonempty H_sparse is flat, and that
  /// matrix is the one decoders are built from. Form selection keys off
  /// H_sparse.empty(). Nonempty H_sparse is exactly the state
  /// expand_dem_chunks() leaves behind, allowing round-trip through YAML.
  std::optional<cudaq::qec::dem_chunks_spec> dem_chunks;
  decoder_custom_args_t decoder_custom_args;

  bool operator==(const decoder_config &) const = default;

  /// Return the parameter map a decoder's constructor should receive: the
  /// stored custom args with the legacy error-rate alias removed and
  /// schema-declared defaults materialized (see materialize_default_args in
  /// cudaq/qec/decoder_config_schema.h) when a schema is registered for
  /// `type`, so programmatically built configs get the same normalization the
  /// YAML parse path applies.
  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  decoder_custom_args_to_heterogeneous_map() const;

  /// Return the model error rates from either the canonical C++ field or the
  /// legacy `decoder_custom_args.error_rate_vec` YAML spelling. The legacy
  /// key is configuration compatibility data and is never included in the
  /// parameter map handed to a decoder plugin.
  ///
  /// @throws std::runtime_error if rates are supplied in both locations, or
  ///         more than once in nested decoder custom arguments.
  __attribute__((visibility("default"))) std::vector<double>
  effective_error_rate_vec() const;

  /// Validate `decoder_custom_args` against the parameter schema registered
  /// for `type`: unknown keys, missing required keys, and the schema's own
  /// validate hook (if any). Throws std::runtime_error on the first
  /// violation. YAML parsing applies the same checks automatically; call this
  /// to vet a configuration built programmatically before using it.
  __attribute__((visibility("default"))) void validate_custom_args() const;

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static decoder_config
  from_yaml_str(const std::string &yaml_str);
};

/// Transport override applied to the rings of one dispatch shape (see the
/// `device_graph` member of `transport_config`).
struct transport_shape_override {
  /// Provider name (e.g. udp, cpu_roce, gpu_roce) or /path/to/lib.so.
  /// A bare name resolves to the CUDA-Q realtime provider library whose
  /// soname is "libcudaq-realtime-bridge-" + name + ".so", with '_' in the
  /// name mapping to '-' to match the shipped hyphenated sonames (so
  /// gpu_roce loads libcudaq-realtime-bridge-gpu-roce.so).
  /// Empty = inherit the section default.
  std::string provider;
  /// Extra provider arguments appended for this shape's rings.
  std::vector<std::string> args;

  bool operator==(const transport_shape_override &) const = default;
};

/// Server-level transport section: the WIRE is deployment configuration and
/// lives OUTSIDE the decoders list.  Transports differ between rings only by
/// dispatch shape (a device_graph ring must be GPU-pollable), so the only
/// override is shape-keyed -- decoder entries carry no transport
/// information.
///
///   transport:
///     provider: udp
///     args: [--slot-size=256]
///     device_graph:
///       provider: udp          # "gpu_roce" on an HSB rig
///       args: [--pinned-rings]
///
/// Resolution per ring: shape override (device_graph rings) > this section's
/// provider/args. The standalone server permits a CLI fallback only for
/// non-GPU-RoCE, host-only configurations whose YAML names no provider. A
/// configuration containing device_graph dispatch or using GPU RoCE must
/// define its provider and arguments in YAML (the deployment file is the
/// source of truth for GPU-visible rings).
struct transport_config {
  std::string provider;
  std::vector<std::string> args;
  transport_shape_override device_graph;

  /// Resolve the provider and ordered arguments for a device-graph ring.
  /// The shape-specific provider overrides the section provider, while its
  /// arguments are appended to the section arguments.
  transport_shape_override resolve_device_graph() const {
    transport_shape_override resolved;
    resolved.provider =
        device_graph.provider.empty() ? provider : device_graph.provider;
    resolved.args = args;
    resolved.args.insert(resolved.args.end(), device_graph.args.begin(),
                         device_graph.args.end());
    return resolved;
  }

  bool operator==(const transport_config &) const = default;
};

class multi_decoder_config {
public:
  std::vector<decoder_config> decoders;
  /// Optional server-level transport section. The standalone server's CLI
  /// defaults may fill an empty section only for non-GPU-RoCE, host-only
  /// configurations.
  transport_config transport;

  bool operator==(const multi_decoder_config &) const = default;

  /// Validate every decoder's custom args (see
  /// decoder_config::validate_custom_args).
  __attribute__((visibility("default"))) void validate_custom_args() const;

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static multi_decoder_config
  from_yaml_str(const std::string_view yaml_str);
};

/// @brief Rewrite a chunk-form configuration into the equivalent flat form,
/// filling block_size, syndrome_size, H_sparse, O_sparse, D_sparse and
/// error_rate_vec from `dem_chunks` expanded `num_rounds` times. Everything
/// downstream of this therefore only has to understand the flat form.
///
/// Does nothing to a configuration that is already flat (one whose `H_sparse`
/// is nonempty, or which carries no `dem_chunks` at all), so it is safe to
/// call unconditionally. An empty H_sparse with dem_chunks present is still
/// treated as chunk form.
///
/// @return The closed DEM the flat fields were derived from, so a caller that
///         also wants its per-fault priors does not have to expand a second
///         time. Empty when the configuration was already flat.
/// @throws std::runtime_error if `num_rounds` is missing, or if the phases
///         cannot be expanded to that many rounds.
__attribute__((visibility("default")))
std::optional<cudaq::qec::detector_error_model>
expand_dem_chunks(decoder_config &config);

/// @brief Generate a JSON Schema (draft 2020-12) document describing valid
/// `multi_decoder_config` YAML files, so third-party tools (check-jsonschema,
/// python jsonschema, yaml-language-server, ...) can validate user-provided
/// configurations offline. The per-decoder `decoder_custom_args` sections are
/// generated from the parameter schemas registered at call time, so decoder
/// plugins (in-tree and out-of-tree alike) appear automatically once their
/// library is loaded. Schema `validate` hooks are arbitrary C++ and are not
/// representable; a document that passes the JSON Schema may still be
/// rejected by those hooks when parsed.
__attribute__((visibility("default"))) std::string decoder_config_json_schema();

/// @brief Configure the decoders (`multi_decoder_config` variant). This
/// function configures both local decoders, and if running on remote target
/// hardware, will submit the configuration to the remote target for further
/// processing.
/// @param config The configuration to use.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders(multi_decoder_config &config);

/// @brief Configure the decoders, resolving relative model paths (such as
/// `stim_dem_path`) against @p base_dir. The overload above uses the process
/// working directory as it stands when resolution starts.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders(multi_decoder_config &config,
                   const std::filesystem::path &base_dir);

/// @brief Configure the decoders from a file. This function configures both
/// local decoders, and if running on remote target hardware, will submit the
/// configuration to the remote target for further processing.
/// @param config_file The file to read the configuration from.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders_from_file(const char *config_file);

/// @brief Configure the decoders from a string. This function configures both
/// local decoders, and if running on remote target hardware, will submit the
/// configuration to the remote target for further processing.
/// @param config_str The string to read the configuration from.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders_from_str(const char *config_str);

/// @brief Finalize the decoders. This function finalizes local decoders.
__attribute__((visibility("default"))) void finalize_decoders();

/// @brief Return the most recently passed multi_decoder_config, or an empty
/// pointer if configure_decoders() has not been called in this process.
/// Used by the decoding-server DeviceCallService plugin to build
/// DecodingSessions on the in-process host_dispatch path without requiring
/// CUDAQ_QEC_DECODER_CONFIG.  Returns shared ownership: a concurrent
/// configure_decoders() replaces the stored config but cannot free it out
/// from under the caller.
__attribute__((visibility("default")))
std::shared_ptr<const multi_decoder_config>
last_configured_multi_decoder_config();

} // namespace cudaq::qec::decoding::config
