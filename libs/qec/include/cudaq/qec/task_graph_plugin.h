/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <stddef.h>
#include <stdint.h>

// This C gateway is checked before calling any C++ registration entry point.
// Bump the framework version and the versioned registration symbols together
// whenever the task-graph factory, metadata or node C++ ABI changes.
#define CUDAQ_QEC_TASK_GRAPH_PLUGIN_PROTOCOL_VERSION 1u
#define CUDAQ_QEC_TASK_GRAPH_FRAMEWORK_ABI_VERSION 2u
#define CUDAQ_QEC_TASK_GRAPH_PLUGIN_DESCRIBE_SYMBOL                            \
  "cudaq_qec_task_graph_plugin_describe"
#define CUDAQ_QEC_TASK_GRAPH_PLUGIN_REGISTER_SYMBOL                            \
  "cudaq_qec_task_graph_plugin_register"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cudaq_qec_task_graph_plugin_descriptor {
  uint32_t struct_size;
  uint32_t protocol_version;
  uint32_t framework_abi_version;
  uint32_t glibcxx_cxx11_abi;
  uint32_t cxx_abi_version;
  uint32_t reserved;
  uint64_t node_build_context_size;
  uint64_t node_kind_spec_size;
  uint64_t task_graph_node_size;
  const char *framework_commit;
  const char *build_id;
} cudaq_qec_task_graph_plugin_descriptor;

typedef const cudaq_qec_task_graph_plugin_descriptor *(
    *cudaq_qec_task_graph_plugin_describe_function)(void);
typedef int (*cudaq_qec_task_graph_plugin_register_function)(char *error,
                                                             size_t capacity);

// Plugins export the two symbols named above with default visibility. Describe
// returns immutable module-owned metadata. Register returns zero on success,
// otherwise a nonzero status and a NUL-terminated diagnostic when capacity > 0.
// Neither entry point may throw. Registration is synchronous, must not replace
// an existing kind/alias, and may only occur when the loader calls register.
// The host rolls back failed registration before unmapping the module; after
// success the module stays mapped for the process lifetime.

#ifdef __cplusplus
}
#endif
