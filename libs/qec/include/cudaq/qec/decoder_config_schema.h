/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include <functional>
#include <string>
#include <vector>

namespace cudaq::qec::decoding::config {

/// Decoder parameter schemas.
///
/// A decoder that wants its constructor parameters to be configurable through
/// the realtime decoding YAML (the `decoder_custom_args` section of a
/// `decoder_config`) registers a `decoder_schema` describing those parameters,
/// keyed by the same name it registers its `cudaq::qec::decoder` extension
/// under. The YAML layer uses the schema to convert the section to and from a
/// `cudaqx::heterogeneous_map` generically -- no decoder-specific parsing code
/// is required in the framework, so out-of-tree decoder plugins can make
/// themselves "realtime ready" from their own shared library.
///
/// The value kinds are a deliberately small, closed set of canonical storage
/// types. Values parsed from YAML are stored in the map with exactly the type
/// listed for their kind; `heterogeneous_map::get` handles related-type
/// retrieval (e.g. an `int` stored for an `int32` parameter can be read back
/// as `std::size_t`).
enum class param_kind {
  boolean,    ///< stored as bool
  int32,      ///< stored as int
  uint64,     ///< stored as std::size_t
  f64,        ///< stored as double
  string,     ///< stored as std::string
  f64_vec,    ///< stored as std::vector<double>
  f64_matrix, ///< stored as std::vector<std::vector<double>>
  /// Nested mapping parsed with the fixed schema named by
  /// `param_spec::subschema`; stored as a nested heterogeneous_map.
  subschema,
  /// Nested mapping whose schema is selected at parse time by the value of
  /// the sibling string key named by `param_spec::discriminator` (e.g.
  /// `global_decoder_params` is parsed with the schema registered under the
  /// value of `global_decoder`); stored as a nested heterogeneous_map.
  discriminated,
};

/// One parameter of a decoder's custom-args schema.
struct param_spec {
  std::string key;
  param_kind kind = param_kind::string;
  /// Parsing fails when a required key is absent.
  bool required = false;
  /// For kind::subschema: the registered schema name to parse the nested
  /// mapping with.
  std::string subschema;
  /// For kind::discriminated: the sibling key whose (string) value names the
  /// schema to parse the nested mapping with.
  std::string discriminator;
  /// For kind::discriminated: when the discriminator is present and names a
  /// registered schema but this key is absent, insert an empty nested map so
  /// downstream consumers see the section (matches the trt_decoder
  /// global_decoder_params defaulting behavior).
  bool materialize_empty = false;
};

/// Declarative description of a decoder's `decoder_custom_args` section.
struct decoder_schema {
  /// Registry key. For decoders this must match the name the decoder is
  /// registered under (the YAML `type` value). Schemas that only serve as
  /// nested sections (e.g. "srelay_bp") may use any unique name.
  std::string name;
  std::vector<param_spec> params;
  /// Optional cross-field validation hook, invoked after a section has been
  /// parsed and its required keys checked. Throw std::runtime_error to
  /// reject the configuration.
  std::function<void(const cudaqx::heterogeneous_map &)> validate;
};

/// Register (or replace) a schema. Decoder plugins call this from a static
/// initializer in the same shared library that registers the decoder itself;
/// the plugin loader runs before any configuration is parsed.
///
/// Schemas are never unregistered or unloaded, by design: registered schemas
/// (including their validate hooks) stay in the registry for the remainder of
/// the process, even after the registering plugin is dlclose'd at exit. The
/// registry is intentionally leaked so no schema destructor can run after its
/// plugin's code has been unloaded.
__attribute__((visibility("default"))) void
register_decoder_schema(decoder_schema schema);

/// Look up a schema by name. Returns nullptr when no schema is registered
/// under `name`. The returned pointer remains valid for the process lifetime
/// (schemas are never unregistered; see register_decoder_schema).
__attribute__((visibility("default"))) const decoder_schema *
find_decoder_schema(const std::string &name);

/// Names of all registered schemas (for diagnostics and introspection).
__attribute__((visibility("default"))) std::vector<std::string>
registered_decoder_schema_names();

/// Validate a custom-args map against the schema registered under
/// `schema_name`: every key must appear in the schema (nested sections are
/// checked recursively), every required key must be present, and the schema's
/// `validate` hook (if any) runs last. Throws std::runtime_error describing
/// the first violation. Maps that did not come from the YAML parser (e.g.
/// built programmatically or from a Python dict) get the same checks the
/// parser applies, so configurations can be validated before use. A map for
/// a name with no registered schema is rejected unless it is empty.
__attribute__((visibility("default"))) void
validate_custom_args(const std::string &schema_name,
                     const cudaqx::heterogeneous_map &args);

/// Overload taking a resolved schema and an error-message context (e.g.
/// "decoder_custom_args (trt_decoder)"). This is the single canonical walk;
/// the YAML parser applies it after parsing a section.
__attribute__((visibility("default"))) void
validate_custom_args(const decoder_schema &schema,
                     const cudaqx::heterogeneous_map &args,
                     const std::string &context);

/// Apply schema-declared defaults to `args` (recursively): a discriminated
/// section with `materialize_empty` set is inserted as an empty nested map
/// when its discriminator names a registered schema and the section key is
/// absent. Idempotent. The YAML parser applies this to parsed sections, and
/// decoder_config::decoder_custom_args_to_heterogeneous_map() applies it to
/// programmatically built configs so both paths hand decoders the same map.
__attribute__((visibility("default"))) void
materialize_default_args(const decoder_schema &schema,
                         cudaqx::heterogeneous_map &args);

/// Remove (with a warning per key) every key `args` holds that is not in
/// `schema`, recursing into nested sections whose schemas resolve. Applied by
/// decoder_config::decoder_custom_args_to_heterogeneous_map() so a non-schema
/// key can never take effect locally but silently vanish remotely. The one
/// exception in emitted YAML is the framework-owned `error_rate_vec`
/// compatibility alias, which is not a decoder plugin parameter.
__attribute__((visibility("default"))) void
drop_non_schema_keys(const decoder_schema &schema,
                     cudaqx::heterogeneous_map &args);

/// Deep equality over the canonical value kinds stored in custom-args maps
/// (scalars, double vectors/matrices, and nested maps). Values of other
/// types compare unequal.
__attribute__((visibility("default"))) bool
custom_args_maps_equal(const cudaqx::heterogeneous_map &a,
                       const cudaqx::heterogeneous_map &b);

} // namespace cudaq::qec::decoding::config
