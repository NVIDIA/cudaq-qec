# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


def _ensure_cuda_runtime_loaded():
    """Ensure CUDA runtime libraries are in the process before loading native extensions."""
    try:
        import cudaq
    except ImportError:
        pass


_ensure_cuda_runtime_loaded()
del _ensure_cuda_runtime_loaded

import functools

from .patch import patch
try:
    from ._pycudaqx_qec_the_suffix_matters_cudaq_qec import *
except ImportError as exc:
    err = str(exc)
    if "libcustabilizer" in err or "cuStabilizer" in err:
        raise ImportError(
            "Failed to load cudaq_qec native extension because cuStabilizer "
            "runtime libraries are missing. Install the matching cuQuantum "
            "package for your CUDA wheel (for example, "
            "'cuquantum-python-cu12>=26.03.0' or "
            "'cuquantum-python-cu13>=26.03.0').") from exc
    if "libcudart" in err:
        raise ImportError(
            f"{err}. Ensure 'nvidia-cuda-runtime-cuXX' is installed "
            "alongside 'cuda-quantum-cuXX'.") from exc
    raise

__version__ = qecrt.__version__
code = qecrt.code
Code = qecrt.Code
_native_decoder = qecrt.decoder
Decoder = qecrt.Decoder


def decoder(name):
    """Register a Python class as a decoder plugin under `name`.

    Wraps the native registration decorator so that any user-defined
    `decode_batch` override is checked at runtime to return a
    BatchDecoderResult. Returning a list[DecoderResult] (the pre-batch API)
    is no longer supported.
    """
    native = _native_decoder(name)

    def wrap(cls):
        if "decode_batch" in cls.__dict__:
            original = cls.decode_batch
            cls_name = cls.__name__

            @functools.wraps(original)
            def checked_decode_batch(self, *args, **kwargs):
                result = original(self, *args, **kwargs)
                if not isinstance(result, qecrt.BatchDecoderResult):
                    raise TypeError(
                        f"{cls_name}.decode_batch must return a "
                        f"BatchDecoderResult; got "
                        f"{type(result).__name__}. See BatchDecoderResult "
                        f"in the cudaq_qec docs for the supported "
                        f"construction surface.")
                return result

            cls.decode_batch = checked_decode_batch
        return native(cls)

    return wrap


TwoQubitDepolarization = qecrt.TwoQubitDepolarization
TwoQubitBitFlip = qecrt.TwoQubitBitFlip
operation = qecrt.operation
get_code = qecrt.get_code
get_available_codes = qecrt.get_available_codes
get_decoder = qecrt.get_decoder
dem_from_stim_text = qecrt.dem_from_stim_text
DecoderResult = qecrt.DecoderResult
BatchDecoderResult = qecrt.BatchDecoderResult
AsyncDecoderResult = qecrt.AsyncDecoderResult
DetectorErrorModel = qecrt.DetectorErrorModel
generate_random_bit_flips = qecrt.generate_random_bit_flips
sample_memory_circuit = qecrt.sample_memory_circuit
x_sample_memory_circuit = qecrt.x_sample_memory_circuit
z_sample_memory_circuit = qecrt.z_sample_memory_circuit
sample_code_capacity = qecrt.sample_code_capacity
dem_from_memory_circuit = qecrt.dem_from_memory_circuit
x_dem_from_memory_circuit = qecrt.x_dem_from_memory_circuit
z_dem_from_memory_circuit = qecrt.z_dem_from_memory_circuit
decoder_context_from_memory_circuit = qecrt.decoder_context_from_memory_circuit
DecoderContext = qecrt.DecoderContext
d_sparse = qecrt.d_sparse

dump_pcm = qecrt.dump_pcm
generate_random_pcm = qecrt.generate_random_pcm
get_pcm_for_rounds = qecrt.get_pcm_for_rounds
get_sorted_pcm_column_indices = qecrt.get_sorted_pcm_column_indices
pcm_is_sorted = qecrt.pcm_is_sorted
reorder_pcm_columns = qecrt.reorder_pcm_columns
shuffle_pcm_columns = qecrt.shuffle_pcm_columns
simplify_pcm = qecrt.simplify_pcm
sort_pcm_columns = qecrt.sort_pcm_columns
pcm_extend_to_n_rounds = qecrt.pcm_extend_to_n_rounds
generate_timelike_sparse_detector_matrix = qecrt.generate_timelike_sparse_detector_matrix
pcm_to_sparse_vec = qecrt.pcm_to_sparse_vec

multi_decoder_config = qecrt.config.multi_decoder_config
decoder_config = qecrt.config.decoder_config
decoder_dispatch = qecrt.config.decoder_dispatch
transport_config = qecrt.config.transport_config
transport_shape_override = qecrt.config.transport_shape_override
configure_decoders_from_file = qecrt.config.configure_decoders_from_file
configure_decoders_from_str = qecrt.config.configure_decoders_from_str
finalize_decoders = qecrt.config.finalize_decoders
configure_decoders = qecrt.config.configure_decoders
expand_dem_chunks = qecrt.config.expand_dem_chunks
decoder_param_schema = qecrt.config.decoder_param_schema
registered_decoder_schemas = qecrt.config.registered_decoder_schemas
decoder_config_json_schema = qecrt.config.decoder_config_json_schema

# Deprecated typed decoder-config classes, kept for backward compatibility
# with the pre-schema API. They warn on construction; new code should assign
# plain dicts to decoder_config.decoder_custom_args instead.
from . import _compat
from ._compat import (nv_qldpc_decoder_config, multi_error_lut_config,
                      trt_decoder_config, pymatching_config, chromobius_config)

for _compat_cls_name in _compat.__all__:
    # The classes used to live in the compiled config submodule; keep that
    # spelling working too (e.g. qec.qecrt.config.sliding_window_config).
    setattr(qecrt.config, _compat_cls_name, getattr(_compat, _compat_cls_name))
del _compat_cls_name

stabilizer_grid = qecrt.stabilizer_grid
role_to_str = qecrt.role_to_str
sc_orientation = qecrt.sc_orientation

from .dem_sampling import dem_sampling
from . import relay_solutions

from .plugins import decoders, codes
import pkgutil, importlib


def iter_namespace(ns_pkg):
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


for finder, name, ispkg in iter_namespace(decoders):
    try:
        importlib.import_module(name)
    except (ModuleNotFoundError, ImportError) as e:
        pass

for finder, name, ispkg in iter_namespace(codes):
    try:
        importlib.import_module(name)
    except (ModuleNotFoundError, ImportError) as e:
        pass

try:
    import cudaq
    from .loader import qec_set_target_callback
    cudaq.register_set_target_callback(qec_set_target_callback, "cudaq_qec")
except ImportError:
    pass
