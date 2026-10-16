# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Deprecated typed decoder-config classes.

These classes reproduce the pre-schema typed configuration API
(``nv_qldpc_decoder_config`` and friends) on top of the schema-driven dict
interface that replaced it. Each instance is a thin wrapper around a dict of
explicitly-set parameters; assigning one to
``decoder_config.decoder_custom_args`` (or passing it to
``decoder_config.set_decoder_custom_args``) converts that dict through the
parameter schema the decoder registered, exactly as if the dict had been
assigned directly.

New code should assign plain dicts instead::

    dc.decoder_custom_args = {"max_iterations": 50}

and can discover any decoder's parameters with
``cudaq_qec.decoder_param_schema(name)``. These classes emit a
``DeprecationWarning`` on construction and will be removed in a future
release.
"""

import warnings

__all__ = [
    "srelay_bp_config",
    "nv_qldpc_decoder_config",
    "single_error_lut_config",
    "multi_error_lut_config",
    "pymatching_config",
    "chromobius_config",
    "trt_decoder_config",
    "sliding_window_config",
]


class _deprecated_typed_config:
    """Base for the deprecated typed decoder-config shims.

    ``_fields`` is the exact attribute surface the pre-schema class exposed;
    every field behaves like the ``std::optional`` it used to be: unset reads
    as ``None``, assigning ``None`` clears it, and ``to_heterogeneous_map``
    only emits fields that were explicitly set.
    """

    # Overridden by subclasses. ``_schema_name`` names the registered decoder
    # parameter schema used to convert the dict when the shim is handed to a
    # decoder_config (it wins over decoder_config.type, so old code works
    # regardless of assignment order).
    _schema_name = None
    _fields = ()

    def __init__(self, map=None):
        warnings.warn(
            f"{type(self).__name__} is deprecated; assign a plain dict to "
            "decoder_config.decoder_custom_args instead (see "
            "cudaq_qec.decoder_param_schema for the accepted keys).",
            DeprecationWarning,
            stacklevel=2)
        object.__setattr__(self, "_args", {})
        if map is not None:
            self._load_map(dict(map))

    def _load_map(self, map):
        # Mirrors the old from_heterogeneous_map: known keys are read, unknown
        # keys are ignored.
        for key, value in map.items():
            if key in self._fields:
                self._args[key] = value

    @classmethod
    def from_heterogeneous_map(cls, map):
        return cls(map)

    def to_heterogeneous_map(self):
        """Return the explicitly-set parameters as a plain dict."""
        out = {}
        for key in self._fields:
            if key in self._args:
                out[key] = _as_plain_value(self._args[key])
        return out

    @classmethod
    def _param_kinds(cls):
        # Field name -> schema kind, from the decoder's registered parameter
        # schema. Lazy: the registry is populated at plugin load, and a miss
        # is not cached so a schema registered later is still picked up.
        if not cls.__dict__.get("_kind_cache"):
            kinds = {}
            try:
                from . import decoder_param_schema
                schema = decoder_param_schema(cls._schema_name)
                if schema:
                    kinds = {entry["key"]: entry["kind"] for entry in schema}
            except Exception:
                pass
            cls._kind_cache = kinds
        return cls._kind_cache

    def __setattr__(self, name, value):
        if name not in self._fields:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        if value is None:
            self._args.pop(name, None)
            return
        kind = self._param_kinds().get(name)
        if kind is not None:
            _check_param_kind(type(self).__name__, name, kind, value)
        self._args[name] = value

    def __getattr__(self, name):
        # Only reached when normal lookup fails, i.e. for unset fields and
        # genuinely unknown attributes.
        if name in type(self)._fields:
            return object.__getattribute__(self, "_args").get(name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def __eq__(self, other):
        return type(other) is type(self) and self._args == other._args

    def __repr__(self):
        settings = ", ".join(
            f"{k}={self._args[k]!r}" for k in self._fields if k in self._args)
        return f"{type(self).__name__}({settings})"


def _as_plain_value(value):
    if isinstance(value, _deprecated_typed_config):
        return value.to_heterogeneous_map()
    return value


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_param_kind(cls_name, name, kind, value):
    # Reproduce the TypeErrors the old nanobind setters raised for
    # clearly-wrong values. Sequence kinds only inspect list/tuple contents,
    # so array-likes (e.g. numpy) pass through to the conversion layer.
    def fail(expected):
        raise TypeError(f"{cls_name}.{name} expects {expected}, got "
                        f"{type(value).__name__}")

    if kind == "bool":
        if not isinstance(value, bool):
            fail("a bool")
    elif kind == "int32":
        if not isinstance(value, int) or isinstance(value, bool):
            fail("an int")
    elif kind == "uint64":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail("a non-negative int")
    elif kind == "float64":
        if not _is_number(value):
            fail("a float")
    elif kind == "string":
        if not isinstance(value, str):
            fail("a str")
    elif kind == "float64_vec":
        if _is_number(value) or isinstance(value, (str, bytes, dict)):
            fail("a list of floats")
        if isinstance(value,
                      (list, tuple)) and not all(_is_number(x) for x in value):
            fail("a list of floats")
    elif kind == "float64_matrix":
        if _is_number(value) or isinstance(value, (str, bytes, dict)):
            fail("a list of lists of floats")
        if isinstance(value, (list, tuple)) and not all(
                isinstance(row,
                           (list, tuple)) and all(_is_number(x)
                                                  for x in row)
                for row in value):
            fail("a list of lists of floats")
    elif kind in ("subschema", "discriminated"):
        if not isinstance(value, (_deprecated_typed_config, dict)):
            fail("a config object or dict")


class srelay_bp_config(_deprecated_typed_config):
    """Deprecated: Relay-BP decoder configuration."""
    _schema_name = "srelay_bp"
    _fields = ("pre_iter", "num_sets", "stopping_criterion", "stop_nconv")


class nv_qldpc_decoder_config(_deprecated_typed_config):
    """Deprecated: nv-qldpc-decoder custom args."""
    _schema_name = "nv-qldpc-decoder"
    _fields = ("use_sparsity", "error_rate", "error_rate_vec", "max_iterations",
               "n_threads", "use_osd", "osd_method", "osd_order",
               "bp_batch_size", "osd_batch_size", "iter_per_check",
               "clip_value", "bp_method", "scale_factor", "proc_float",
               "gamma0", "gamma_dist", "explicit_gammas", "srelay_config",
               "bp_seed", "composition")

    def _load_map(self, map):
        srelay = map.pop("srelay_config", None)
        super()._load_map(map)
        if srelay is not None:
            self._args["srelay_config"] = srelay_bp_config(srelay)


class single_error_lut_config(_deprecated_typed_config):
    """Deprecated: single_error_lut decoder configuration."""
    _schema_name = "single_error_lut"
    _fields = ()


class multi_error_lut_config(_deprecated_typed_config):
    """Deprecated: multi_error_lut decoder configuration."""
    _schema_name = "multi_error_lut"
    _fields = ("lut_error_depth",)


class pymatching_config(_deprecated_typed_config):
    """Deprecated: pymatching decoder configuration."""
    _schema_name = "pymatching"
    _fields = ("error_rate_vec", "merge_strategy")


class chromobius_config(_deprecated_typed_config):
    """Deprecated: chromobius decoder configuration."""
    _schema_name = "chromobius"
    _fields = ("drop_mobius_errors_involving_remnant_errors",
               "ignore_decomposition_failures", "include_coords_in_mobius_dem",
               "return_weight", "write_mobius_match_to_stderr")


class trt_decoder_config(_deprecated_typed_config):
    """Deprecated: trt_decoder custom args."""
    _schema_name = "trt_decoder"
    _fields = ("onnx_load_path", "engine_load_path", "engine_save_path",
               "precision", "memory_workspace", "batch_size", "use_cuda_graph",
               "engine_output_format", "global_decoder",
               "global_decoder_params")

    _global_decoder_classes = {
        "pymatching": pymatching_config,
        "chromobius": chromobius_config,
    }

    def _load_map(self, map):
        params = map.pop("global_decoder_params", None)
        super()._load_map(map)
        cls = self._global_decoder_classes.get(self._args.get("global_decoder"))
        if params is not None:
            if cls is None:
                # Matches the old from_heterogeneous_map: this typed API only
                # ever recognized pymatching/chromobius global decoders. Use
                # plain dicts to configure any other registered decoder.
                raise RuntimeError(
                    "global_decoder_params does not support global_decoder "
                    f"'{self._args.get('global_decoder')}'")
            self._args["global_decoder_params"] = cls(params)
        elif cls is not None:
            # The old from_heterogeneous_map materialized a default typed
            # config for a recognized global_decoder.
            self._args["global_decoder_params"] = cls()

    def to_heterogeneous_map(self):
        out = super().to_heterogeneous_map()
        # The old struct always emitted a (possibly empty) params section for
        # a recognized global_decoder. The framework's materialize_empty
        # defaulting produces the same final map either way, but old code may
        # inspect this map directly.
        if ("global_decoder_params" not in out and
                self._args.get("global_decoder")
                in self._global_decoder_classes):
            out["global_decoder_params"] = {}
        return out


class sliding_window_config(_deprecated_typed_config):
    """Deprecated: sliding_window decoder custom args."""
    _schema_name = "sliding_window"
    _fields = ("window_size", "step_size", "num_syndromes_per_round",
               "straddle_start_round", "straddle_end_round", "error_rate_vec",
               "inner_decoder_name", "single_error_lut_params",
               "multi_error_lut_params", "nv_qldpc_decoder_params")

    # The old typed struct had one field per supported inner decoder but
    # always emitted a single "inner_decoder_params" section, taking the first
    # set field in this order.
    _inner_param_fields = (
        ("single_error_lut_params", "single_error_lut",
         single_error_lut_config),
        ("multi_error_lut_params", "multi_error_lut", multi_error_lut_config),
        ("nv_qldpc_decoder_params", "nv-qldpc-decoder",
         nv_qldpc_decoder_config),
    )

    def _load_map(self, map):
        inner = map.pop("inner_decoder_params", None)
        super()._load_map(map)
        if inner is not None:
            for field, decoder_name, cls in self._inner_param_fields:
                if self._args.get("inner_decoder_name") == decoder_name:
                    self._args[field] = cls(inner)
                    break

    def to_heterogeneous_map(self):
        out = {}
        inner_fields = tuple(f for f, _, _ in self._inner_param_fields)
        for key in self._fields:
            if key in self._args and key not in inner_fields:
                out[key] = _as_plain_value(self._args[key])
        for field, _, _ in self._inner_param_fields:
            if field in self._args:
                inner = _as_plain_value(self._args[field])
                # The old struct only emitted a non-empty section.
                if inner:
                    out["inner_decoder_params"] = inner
                break
        return out
