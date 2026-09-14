#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Build a test-only copy of logger_forwarder.cpp with race-pause hooks.

The installed library is not modified. Unique product anchors must exist or
this script exits non-zero so the race target cannot silently miss a site.
"""

import sys
from pathlib import Path

HOOK_DECL = """\
extern "C" {
void qec_cc_logger_hook_after_producer_guard();
void qec_cc_logger_hook_after_load_enqueue_pos();
void qec_cc_logger_hook_after_enqueue_cas_ready();
void qec_cc_logger_hook_after_load_dequeue_pos();
}

"""

REPLACEMENTS = [
    (
        "    producer_guard guard(active_producers_);\n",
        "    producer_guard guard(active_producers_);\n"
        "    qec_cc_logger_hook_after_producer_guard();\n",
    ),
    (
        "  bool try_enqueue_one(queued_log_record &record) {\n"
        "    std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);\n",
        "  bool try_enqueue_one(queued_log_record &record) {\n"
        "    std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);\n"
        "    qec_cc_logger_hook_after_load_enqueue_pos();\n",
    ),
    (
        "      if (dif == 0) {\n"
        "        if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,\n",
        "      if (dif == 0) {\n"
        "        qec_cc_logger_hook_after_enqueue_cas_ready();\n"
        "        if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,\n",
    ),
    (
        "  bool try_dequeue_one(queued_log_record &record) {\n"
        "    std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);\n",
        "  bool try_dequeue_one(queued_log_record &record) {\n"
        "    std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);\n"
        "    qec_cc_logger_hook_after_load_dequeue_pos();\n",
    ),
]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: gen_logger_forwarder_races_tu.py SRC DST",
              file=sys.stderr)
        return 2
    src = Path(sys.argv[1]).read_text()
    for old, new in REPLACEMENTS:
        if old not in src:
            print("missing logger_forwarder hook anchor:\n" + old,
                  file=sys.stderr)
            return 1
        src = src.replace(old, new, 1)
    Path(sys.argv[2]).write_text(HOOK_DECL + src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
