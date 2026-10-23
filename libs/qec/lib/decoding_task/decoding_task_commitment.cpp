/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The window schedule of a shot and commitment validation. The schedule
// places every window instance of every layer and maps each incoming edge
// to its producer instance. Validation checks every probe shot in three
// passes: sends against receives, then row finalization, then each layer
// variant's parallel columns. Commit zones are whole rounds and a fault
// touches only its own round and the next, so row finalization reads which
// rows each round's faults touch from the template's chunks and never
// stitches a shot. Runs at configuration time; nothing here is on the hot
// path.

#include "decoding_task_commitment.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace cudaq::qec {
namespace {

// Throws std::invalid_argument from schedule_windows().
[[noreturn]] void reject_schedule(const std::string &what) {
  throw std::invalid_argument("schedule_windows: " + what);
}

// Throws std::invalid_argument from validate_commitment().
[[noreturn]] void reject(const std::string &what) {
  throw std::invalid_argument("validate_commitment: " + what);
}

// Whether round q lies in r.
bool contains(round_range r, std::int64_t q) {
  return q >= r.begin && q < r.end;
}

// "[b, e)".
std::string describe(round_range r) {
  return "[" + std::to_string(r.begin) + ", " + std::to_string(r.end) + ")";
}

// "[b, e), [b, e)".
std::string describe(const std::vector<round_range> &ranges) {
  std::string out;
  for (const auto &r : ranges)
    out += (out.empty() ? "" : ", ") + describe(r);
  return out;
}

// "bulk", or the variant's prefix position and terminal clip.
std::string describe(const variant_key &key) {
  std::string out;
  if (key.bulk())
    return "bulk";
  if (key.position)
    out = "position " + std::to_string(*key.position);
  if (key.terminal)
    out += (out.empty() ? "" : ", ") + std::string("terminal ") +
           std::to_string(*key.terminal);
  return out;
}

// "edge A.forward -> B.receive_0 (index_offset -1)".
std::string describe(const task_graph_edge &e) {
  auto out = "edge " + e.from_node + "." + e.from_output + " -> " + e.to_node +
             "." + e.to_input;
  if (e.index_offset != 0)
    out += " (index_offset " + std::to_string(e.index_offset) + ")";
  return out;
}

// Rejects, for @p what, an edge reaching further than window_round_limit
// instances, which no shot holds, before any arithmetic on its offset.
void check_index_offsets(const std::vector<task_graph_edge> &edges,
                         const std::string &what) {
  for (const auto &e : edges)
    if (e.index_offset < -window_round_limit ||
        e.index_offset > window_round_limit)
      throw std::invalid_argument(
          what + ": " + describe(e) + " reaches " +
          (e.index_offset < 0 ? "back" : "forward") + " beyond " +
          std::to_string(window_round_limit) + " instances, past any shot");
}

// " of the 6-round shot".
std::string of_shot(std::size_t num_rounds) {
  return " of the " + std::to_string(num_rounds) + "-round shot";
}

// "A[1]".
std::string name(const scheduled_window &w) {
  return w.node + "[" + std::to_string(w.k) + "]";
}

// "A[0], B[1]", for windows given by schedule index.
std::string names(const window_schedule &s,
                  const std::vector<std::size_t> &windows) {
  std::string out;
  for (const auto i : windows)
    out += (out.empty() ? "" : ", ") + name(s.windows[i]);
  return out;
}

// The faults of one chunk on each row they touch: the first fault column on
// each row of the chunk's own round, and on each row of the next round
// through its outgoing seam; -1 where none does.
struct chunk_touch {
  std::vector<std::int64_t> own;
  std::vector<std::int64_t> next;
  std::int64_t first_detectable = -1; ///< First column touching any row
};

// A round's rows are its chunk's to_seam band; the from_seam band holds the
// next round's rows, in the same order.
chunk_touch touch_of(const extended_dem &chunk, const dem_chunks_spec &spec) {
  chunk_touch out;
  const auto rows = chunk.H.to_nested_csr();
  const auto band = [&](seam_id id, std::vector<std::int64_t> &first) {
    if (!chunk.has_seam(id))
      return;
    const auto &seam = chunk.get_seam(id);
    for (auto r = seam.row_begin; r < seam.row_end; ++r) {
      std::int64_t column = -1;
      if (!rows[r].empty())
        column = *std::min_element(rows[r].begin(), rows[r].end());
      first.push_back(column);
      if (column >= 0 &&
          (out.first_detectable < 0 || column < out.first_detectable))
        out.first_detectable = column;
    }
  };
  band(spec.seam.to_seam, out.own);
  band(spec.seam.from_seam, out.next);
  return out;
}

// The chunk_touch of each round of a shot, read once per distinct chunk.
// The template keeps its chunks, so their addresses key the cache.
std::vector<const chunk_touch *>
touch_rounds(const experiment_template &t, std::size_t num_rounds,
             std::map<const extended_dem *, chunk_touch> &cache) {
  std::vector<const chunk_touch *> out;
  for (std::size_t q = 0; q < num_rounds; ++q) {
    const auto &chunk = t.chunk(q, num_rounds);
    auto it = cache.find(&chunk);
    if (it == cache.end())
      it = cache.emplace(&chunk, touch_of(chunk, t.spec())).first;
    out.push_back(&it->second);
  }
  return out;
}

// The windows holding each round, as own or imported rounds, and the
// windows committing it, by schedule index.
struct round_windows {
  std::vector<std::vector<std::size_t>> holders;
  std::vector<std::vector<std::size_t>> committers;
};

// Lists each window under the rounds of its zones and imports.
round_windows index_rounds(const window_schedule &s) {
  round_windows out;
  out.holders.resize(s.num_rounds);
  out.committers.resize(s.num_rounds);
  for (std::size_t i = 0; i < s.windows.size(); ++i) {
    const auto &w = s.windows[i];
    for (auto q = w.rounds.begin; q < w.rounds.end; ++q)
      out.holders[q].push_back(i);
    for (auto q = w.output.imports.begin; q < w.output.imports.end; ++q)
      out.holders[q].push_back(i);
    for (auto q = w.commit.begin; q < w.commit.end; ++q)
      out.committers[q].push_back(i);
  }
  return out;
}

// The schedule index of each window, by node and k.
std::map<std::pair<std::string, std::int64_t>, std::size_t>
positions(const window_schedule &s) {
  std::map<std::pair<std::string, std::int64_t>, std::size_t> out;
  for (std::size_t i = 0; i < s.windows.size(); ++i)
    out[{s.windows[i].node, s.windows[i].k}] = i;
  return out;
}

// The send of producer p that input d reads; schedule_windows() checked
// that it exists.
const window_send &send_of(const scheduled_window &p,
                           const window_dependency &d) {
  return *std::find_if(
      p.sends.begin(), p.sends.end(),
      [&](const window_send &send) { return send.id == d.from_output; });
}

// "edge A.forward -> B.receive_0 (index_offset -1)": the edge behind input
// d of consumer c.
std::string describe(const window_dependency &d, const scheduled_window &c) {
  task_graph_edge e;
  e.from_node = d.from_node;
  e.from_output = d.from_output;
  e.to_node = c.node;
  e.to_input = d.to_input;
  e.index_offset = d.k - c.k;
  return describe(e);
}

// Every edge's delivery against its consumer: a delta lands on the
// consumer's own rounds, and a residual covers exactly its imports.
void check_deliveries(const window_schedule &s) {
  const auto index = positions(s);
  for (const auto &c : s.windows) {
    // The inputs already read, by producer and send: a send arriving twice
    // would cancel itself.
    std::map<std::tuple<std::string, std::int64_t, std::string>, std::string>
        received;
    for (const auto &input : c.inputs) {
      std::vector<std::int64_t> sent;
      std::vector<std::int64_t> imported;
      std::vector<std::int64_t> differ;
      if (!input.exists)
        continue;
      const auto &p = s.windows[index.at({input.from_node, input.k})];
      const auto &send = send_of(p, input);
      const bool delta = send.mode == send_mode::delta;
      const auto [first, fresh] = received.emplace(
          std::tuple{input.from_node, input.k, input.from_output},
          input.to_input);
      if (!fresh)
        reject(name(c) + of_shot(s.num_rounds) + " receives " + name(p) +
               "'s " + (delta ? "delta" : "residual") + " '" + send.id +
               "' twice, on " + first->second + " and " + input.to_input);
      const auto what = describe(input, c) + " in the " +
                        std::to_string(s.num_rounds) +
                        "-round shot: " + name(p) + "'s " +
                        (delta ? "delta" : "residual") + " '" + send.id + "'";
      if (delta) {
        for (const auto &r : send.rounds) {
          if (r.begin >= c.rounds.begin && r.end <= c.rounds.end)
            continue;
          const auto outside = r.begin < c.rounds.begin
                                   ? r.begin
                                   : std::max(r.begin, c.rounds.end);
          reject(what + " reaches round " + std::to_string(outside) +
                 ", outside " + name(c) + "'s rounds " + describe(c.rounds));
        }
        continue;
      }
      for (const auto &r : send.rounds)
        for (auto q = r.begin; q < r.end; ++q)
          sent.push_back(q);
      for (auto q = c.output.imports.begin; q < c.output.imports.end; ++q)
        imported.push_back(q);
      std::set_symmetric_difference(sent.begin(), sent.end(), imported.begin(),
                                    imported.end(), std::back_inserter(differ));
      if (!differ.empty())
        reject(
            what + " covers " +
            (sent.empty() ? "no rounds" : "rounds " + describe(send.rounds)) +
            ", but " + name(c) + " imports " +
            (imported.empty() ? "no rounds" : describe(c.output.imports)) +
            "; round " + std::to_string(differ.front()) + " differs");
    } // end - for(input)
  } // end - for(c)
} // end - check_deliveries()

// Whether window w finalizes a row of round q: it has a column on the row
// and every one is a commit column. own and incoming say whether faults of
// round q and of round q - 1 touch the row. An open input puts a boundary
// column on every row of the window's first round, or a truncated crossing
// column on each row the previous round's faults touch.
bool finalizes(const scheduled_window &w, std::int64_t q, bool own,
               bool incoming) {
  const bool at_open_input = q == w.rounds.begin && w.input.exists &&
                             w.input.disposition == port_disposition::open;
  bool any = false;
  if (at_open_input &&
      (w.input.boundary == input_boundary::columns || incoming))
    return false;
  if (own && contains(w.rounds, q)) {
    if (!contains(w.commit, q))
      return false;
    any = true;
  }
  if (incoming && contains(w.rounds, q - 1)) {
    if (!contains(w.commit, q - 1))
      return false;
    any = true;
  }
  return any;
}

// Whether window p sends window c, through one of c's inputs, a send whose
// rounds hold round q.
bool sends_round(const scheduled_window &p, const scheduled_window &c,
                 std::int64_t q) {
  for (const auto &input : c.inputs) {
    if (input.from_node != p.node || input.k != p.k)
      continue;
    const auto &rounds = send_of(p, input).rounds;
    if (std::any_of(rounds.begin(), rounds.end(),
                    [&](round_range r) { return contains(r, q); }))
      return true;
  }
  return false;
}

// F1 and F3 on every row that some fault touches, in row order. The faults
// on a row are round q's and round q - 1's, each named by its first column.
// F2 needs no check: check_commits() has already found every detectable
// fault committed once, so each fault on the row is the finalizer's column
// or committed by another window.
void check_rows(const experiment_template &t, const window_schedule &s,
                const round_windows &index,
                const std::vector<const chunk_touch *> &touch) {
  const auto R = s.num_rounds;
  const std::vector<std::int64_t> none;
  for (std::size_t q = 0; q < R; ++q) {
    const auto round = static_cast<std::int64_t>(q);
    const auto &own = touch[q]->own;
    const auto &incoming = q > 0 ? touch[q - 1]->next : none;
    for (std::size_t i = 0; i < own.size(); ++i) {
      const std::array<std::pair<std::int64_t, std::int64_t>, 2> faults{
          {{round, own[i]},
           {round - 1, i < incoming.size() ? incoming[i] : -1}}};
      std::vector<std::size_t> finalizers;
      const auto row = [&] {
        return "row " + std::to_string(t.row_offset(q, R) + i) + " (round " +
               std::to_string(q) + ")" + of_shot(R);
      };
      if (faults[0].second < 0 && faults[1].second < 0)
        continue;
      for (const auto w : index.holders[q])
        if (finalizes(s.windows[w], round, faults[0].second >= 0,
                      faults[1].second >= 0))
          finalizers.push_back(w);
      if (finalizers.empty())
        reject("F1: " + row() + " has no finalizer; held by " +
               (index.holders[q].empty() ? "no window"
                                         : names(s, index.holders[q])));
      if (finalizers.size() > 1)
        reject("F1: " + row() + " has " + std::to_string(finalizers.size()) +
               " finalizers: " + names(s, finalizers));
      const auto &f = s.windows[finalizers.front()];
      for (const auto &[x, column] : faults) {
        if (column < 0)
          continue;
        const auto fault = "fault " +
                           std::to_string(t.column_offset(x, R) + column) +
                           " (round " + std::to_string(x) + ")";
        for (const auto w : index.committers[x]) {
          const auto &p = s.windows[w];
          if (w != finalizers.front() && !sends_round(p, f, round))
            reject("F3: " + row() + ", finalized by " + name(f) + ": " +
                   name(p) + " commits " + fault + " on it, but no send of " +
                   name(p) + " to " + name(f) + " covers round " +
                   std::to_string(q));
        }
      } // end - for(faults)
    } // end - for(i)
  } // end - for(q)
} // end - check_rows()

// F4: every round with a detectable fault is committed by exactly one
// window, so each of its faults is.
void check_commits(const experiment_template &t, const window_schedule &s,
                   const round_windows &index,
                   const std::vector<const chunk_touch *> &touch) {
  const auto R = s.num_rounds;
  for (std::size_t q = 0; q < R; ++q) {
    const auto &committers = index.committers[q];
    if (touch[q]->first_detectable < 0 || committers.size() == 1)
      continue;
    reject("F4: fault " +
           std::to_string(t.column_offset(q, R) + touch[q]->first_detectable) +
           " (round " + std::to_string(q) + ")" + of_shot(R) +
           " is committed by " +
           (committers.empty() ? "no window"
                               : std::to_string(committers.size()) +
                                     " windows: " + names(s, committers)));
  }
}

// Each window with imports takes their values from exactly one residual.
void check_imports(const window_schedule &s) {
  const auto index = positions(s);
  for (const auto &c : s.windows) {
    std::size_t residuals = 0;
    if (c.output.imports.empty())
      continue;
    for (const auto &input : c.inputs)
      if (input.exists &&
          send_of(s.windows[index.at({input.from_node, input.k})], input)
                  .mode == send_mode::residual)
        ++residuals;
    if (residuals != 1)
      reject(name(c) + of_shot(s.num_rounds) + " imports rounds " +
             describe(c.output.imports) + " but receives " +
             (residuals == 0 ? "no residual"
                             : std::to_string(residuals) + " residuals"));
  }
}

// Builds window s's variant and rejects its first commit column with a
// conflicting parallel column, naming both by round.
void check_parallel_columns(std::shared_ptr<const experiment_template> context,
                            const window_request &request,
                            const scheduled_window &s, std::size_t num_rounds) {
  const auto &t = *context;
  const auto w = window_from_dem_chunks(context, request, s.k, num_rounds);
  const auto column = [&](std::size_t c) {
    auto r = w.rounds.begin;
    const auto first = t.column_offset(w.rounds.begin, num_rounds);
    if (c >= w.num_fault_columns)
      return "column " + std::to_string(c);
    while (t.column_offset(r + 1, num_rounds) - first <=
           w.fault_column_sources[c].front())
      ++r;
    return "column " + std::to_string(c) + " (round " + std::to_string(r) + ")";
  };
  if (w.parallel_conflicts.empty())
    return;
  const auto &p = w.parallel_conflicts.front();
  std::string reason;
  switch (p.conflict) {
  case parallel_conflict::other_zone:
    reason = ", which is in another zone";
    break;
  case parallel_conflict::boundary:
    reason = ", a boundary or truncated column";
    break;
  case parallel_conflict::outside_rows:
    reason = ", whose rows outside the window differ";
    break;
  case parallel_conflict::observable:
    reason = ", which has a different observable effect; "
             "conflicting_parallel: most_likely resolves it";
    break;
  }
  reject(name(s) + " (" + describe(s.variant) + ")" + of_shot(num_rounds) +
         ": commit " + column(p.commit_column) + " is parallel to " +
         column(p.other_column) + reason);
} // end - check_parallel_columns()

} // namespace

// Places each layer's instances in order, then maps each incoming edge to
// its producer instance k + index_offset.
window_schedule
schedule_windows(std::shared_ptr<const experiment_template> context,
                 const std::vector<window_layer> &layers,
                 const std::vector<task_graph_edge> &edges,
                 std::size_t num_rounds) {
  window_schedule schedule;
  std::map<std::string, const window_layer *> by_node;
  if (!context)
    reject_schedule("no template");
  check_index_offsets(edges, "schedule_windows");
  for (const auto &layer : layers) {
    experimental::check_window_request(layer.request);
    if (!by_node.emplace(layer.node, &layer).second)
      reject_schedule("node '" + layer.node + "' is repeated");
  }
  for (const auto &e : edges) {
    const auto from = by_node.find(e.from_node);
    if (from == by_node.end() || by_node.count(e.to_node) == 0)
      reject_schedule(describe(e) + " names a node that is not a window layer");
    const auto &sends = from->second->request.sends;
    if (std::none_of(sends.begin(), sends.end(), [&](const auto &send) {
          return send.id == e.from_output;
        }))
      reject_schedule(describe(e) + ": " + e.from_node + " has no send '" +
                      e.from_output + "'");
  }
  schedule.num_rounds = num_rounds;
  for (const auto &layer : layers) {
    const auto instances = window_instances(layer.request, num_rounds);
    for (auto k = instances.begin; k < instances.end; ++k) {
      auto w = place_window(context, layer.request, k, num_rounds);
      scheduled_window s;
      s.node = layer.node;
      s.k = k;
      s.variant = window_variant(*context, layer.request, k, num_rounds);
      s.rounds = w.rounds;
      s.commit = w.commit;
      s.buffers = std::move(w.buffers);
      s.input = w.input;
      s.output = w.output;
      s.sends = std::move(w.sends);
      for (const auto &e : edges) {
        if (e.to_node != layer.node)
          continue;
        const auto j = k + e.index_offset;
        const auto producer =
            window_instances(by_node.at(e.from_node)->request, num_rounds);
        s.inputs.push_back({e.from_node, e.from_output, j, e.to_input,
                            j >= producer.begin && j < producer.end});
      }
      schedule.windows.push_back(std::move(s));
    } // end - for(k)
  } // end - for(layer)
  return schedule;
} // end - schedule_windows()

// The longest layer's variant probe, one period longer, so the probe shots
// also hold the bulk around two consecutive bulk windows of every layer;
// longer still by the periods the farthest edge reaches back, so that every
// consumer variant meets its producer. Offsets and periods lie within
// window_round_limit, so the sum stays within range.
std::size_t commitment_probe_rounds(const experiment_template &context,
                                    const std::vector<window_layer> &layers,
                                    const std::vector<task_graph_edge> &edges) {
  std::size_t end = context.min_rounds() + 1;
  std::int64_t reach = 1;
  check_index_offsets(edges, "commitment_probe_rounds");
  if (!context.repeats() || layers.empty())
    return end;
  const auto &first = layers.front();
  for (const auto &e : edges)
    reach = std::max(reach, std::abs(e.index_offset));
  for (const auto &layer : layers) {
    if (layer.request.period != first.request.period)
      throw std::invalid_argument(
          "commitment_probe_rounds: layers " + first.node + " and " +
          layer.node + " have periods " + std::to_string(first.request.period) +
          " and " + std::to_string(layer.request.period) +
          "; every layer needs the same period");
    end = std::max(end, variant_probe_rounds(context, layer.request));
  }
  return end + static_cast<std::size_t>(reach * first.request.period);
} // end - commitment_probe_rounds()

// Sends are checked for every probe shot before anything else, so an edge
// with the wrong index map reports its mismatch rather than the row it
// leaves uncovered; then F4 for every shot, so a fault committed twice or
// not at all is reported as such rather than as a row it unsettles; then
// the rows; parallel columns last, once per layer variant.
void validate_commitment(std::shared_ptr<const experiment_template> context,
                         const std::vector<window_layer> &layers,
                         const std::vector<task_graph_edge> &edges) {
  std::vector<window_schedule> schedules;
  std::map<const extended_dem *, chunk_touch> touches;
  std::set<std::pair<std::string, variant_key>> variants;
  std::map<std::string, const window_request *> requests;
  if (!context)
    reject("no template");
  for (const auto &layer : layers)
    requests[layer.node] = &layer.request;
  const auto end = commitment_probe_rounds(*context, layers, edges);
  for (auto R = context->min_rounds(); R < end; ++R) {
    schedules.push_back(schedule_windows(context, layers, edges, R));
    check_deliveries(schedules.back());
  }
  for (const auto &schedule : schedules)
    check_commits(*context, schedule, index_rounds(schedule),
                  touch_rounds(*context, schedule.num_rounds, touches));
  for (const auto &schedule : schedules) {
    check_rows(*context, schedule, index_rounds(schedule),
               touch_rounds(*context, schedule.num_rounds, touches));
    check_imports(schedule);
  }
  for (const auto &schedule : schedules)
    for (const auto &w : schedule.windows)
      if (variants.insert({w.node, w.variant}).second)
        check_parallel_columns(context, *requests.at(w.node), w,
                               schedule.num_rounds);
} // end - validate_commitment()

namespace {

// Throws std::invalid_argument from validate_window_topology(), which the
// graph builder runs.
[[noreturn]] void reject_topology(const std::string &what) {
  throw std::invalid_argument("from_topology: " + what);
}

// The request of `n`'s window port `port`, or null.
const window_request *layer_request(const task_graph_node_config &n,
                                    const std::string &port) {
  for (const auto &w : n.windows)
    if (w.layer == port)
      return &w.request;
  return nullptr;
}

// The nodes reachable from `id` along edges, in the order first reached.
// The graph's output sink is not a node: each edge into it is a result of
// its own, so two nodes that both feed it share nothing there.
std::vector<std::string> downstream(const std::string &id,
                                    const std::vector<task_graph_edge> &edges) {
  std::vector<std::string> out;
  std::vector<std::string> todo{id};
  while (!todo.empty()) {
    const auto at = todo.front();
    todo.erase(todo.begin());
    for (const auto &e : edges)
      if (e.from_node == at && e.to_node != "output" &&
          std::find(out.begin(), out.end(), e.to_node) == out.end()) {
        out.push_back(e.to_node);
        todo.push_back(e.to_node);
      }
  }
  return out;
}

// The receive port index of "receive_<k>", or nothing for another name: k
// is written in decimal without leading zeros, as the ports are named, and
// at most 9 digits long, which no num_receives approaches and every size_t
// holds.
std::optional<std::size_t> receive_index(const std::string &port) {
  const std::string prefix = "receive_";
  std::size_t k = 0;
  if (port.size() <= prefix.size() || port.compare(0, prefix.size(), prefix))
    return std::nullopt;
  const auto digits = port.substr(prefix.size());
  if (digits.size() > 9 || (digits.size() > 1 && digits.front() == '0') ||
      !std::all_of(digits.begin(), digits.end(),
                   [](char c) { return c >= '0' && c <= '9'; }))
    return std::nullopt;
  for (const auto c : digits)
    k = 10 * k + static_cast<std::size_t>(c - '0');
  return k;
}

// A cycle of receives at index_offset 0 among `windows`, as the nodes along
// it with the first repeated at the end, or empty if there is none. Each
// such receive makes a window wait on another of its own index. A depth-first
// search: `path` holds the nodes being visited, and a receive back onto it
// closes a cycle.
std::vector<std::string> same_index_cycle(
    const std::map<std::string, const task_graph_node_config *> &windows,
    const std::vector<task_graph_edge> &edges) {
  std::vector<std::string> path;
  std::set<std::string> done;
  std::function<bool(const std::string &)> visit = [&](const std::string &id) {
    path.push_back(id);
    for (const auto &e : edges) {
      if (e.from_node != id || !windows.count(e.to_node) ||
          e.to_input == "input" || e.index_offset != 0 || done.count(e.to_node))
        continue;
      const auto on_path = std::find(path.begin(), path.end(), e.to_node);
      if (on_path != path.end()) {
        path.erase(path.begin(), on_path);
        path.push_back(e.to_node);
        return true;
      }
      if (visit(e.to_node))
        return true;
    }
    path.pop_back();
    done.insert(id);
    return false;
  };
  for (const auto &[id, n] : windows)
    if (!done.count(id) && visit(id))
      return path;
  return {};
} // end - same_index_cycle()

} // namespace

// Shape first, so a mistake is named for what it is rather than as the
// commitment failure it causes; then validate_commitment() per ingest node.
void validate_window_topology(const std::vector<task_graph_node_config> &nodes,
                              const std::vector<task_graph_edge> &edges) {
  std::map<std::string, const task_graph_node_config *> ingests;
  std::map<std::string, const task_graph_node_config *> windows;
  std::map<std::string, std::shared_ptr<const experiment_template>> templates;
  std::map<std::string, std::pair<std::string, std::string>> fed_by;
  std::map<std::string, std::vector<window_layer>> layers;
  std::map<std::string, std::vector<task_graph_edge>> sends;
  for (const auto &n : nodes) {
    const auto *kind = find_task_graph_node_kind(n.kind);
    if (kind && kind->name == "ingest")
      ingests[n.id] = &n;
    else if (kind && kind->name == "decode_window")
      windows[n.id] = &n;
  }
  if (ingests.empty() && windows.empty())
    return;

  for (const auto &[id, n] : ingests) {
    std::size_t fed = 0;
    std::size_t other = 0;
    for (const auto &e : edges) {
      if (e.to_node != id)
        continue;
      if (e.from_node == "measurements" || e.from_node == "dets")
        ++fed;
      else
        ++other;
    }
    if (fed != 1 || other != 0)
      reject_topology("ingest node '" + id +
                      "' must be fed by measurements alone");
    if (!n->dem)
      reject_topology("ingest node '" + id + "' has no dem_chunks template");
    templates[id] = std::make_shared<const experiment_template>(*n->dem);
    if (n->node_params.contains("read_logical") &&
        n->node_params.get<bool>("read_logical") &&
        templates[id]->readout().empty())
      reject_topology("ingest node '" + id +
                      "' reads the logical value, but its template's last "
                      "phase has no m2o_sparse");
  } // end - for(ingests)

  for (const auto &[id, n] : windows) {
    const auto num_receives =
        n->node_params.contains("num_receives")
            ? n->node_params.get<std::size_t>("num_receives")
            : std::size_t{0};
    for (const auto &e : edges) {
      if (e.to_node != id)
        continue;
      const auto receive = receive_index(e.to_input);
      if (e.to_input == "input") {
        const auto ingest = ingests.find(e.from_node);
        if (ingest == ingests.end() ||
            !layer_request(*ingest->second, e.from_output))
          reject_topology("decode_window node '" + id + "' is fed by '" +
                          e.from_node + "." + e.from_output +
                          "', which is not a window port of an ingest node");
        if (fed_by.count(id))
          reject_topology("decode_window node '" + id +
                          "' is fed by more than one window port");
        fed_by[id] = {e.from_node, e.from_output};
      } else if (!receive) {
        reject_topology("decode_window node '" + id + "' has no input '" +
                        e.to_input + "'");
      } else if (*receive >= num_receives) {
        reject_topology("decode_window node '" + id + "' has num_receives " +
                        std::to_string(num_receives) +
                        ", so it has no input '" + e.to_input + "'");
      } else if (!windows.count(e.from_node)) {
        reject_topology("decode_window node '" + id + "' receives on '" +
                        e.to_input + "' from '" + e.from_node +
                        "', which is not a decode_window node");
      } else if (e.index_offset > 0) {
        reject_topology(describe(e) + " has a positive index_offset: a window "
                                      "receives only from its own or earlier "
                                      "instances");
      }
    } // end - for(e)
    if (!fed_by.count(id))
      reject_topology("decode_window node '" + id +
                      "' needs an input from a window port of an ingest node");
  } // end - for(windows)

  // Receives at index_offset 0 that close a cycle leave every window on it
  // waiting on its own index.
  if (const auto cycle = same_index_cycle(windows, edges); !cycle.empty()) {
    std::string nodes = "'" + cycle.front() + "'";
    for (std::size_t i = 1; i < cycle.size(); ++i)
      nodes += " -> '" + cycle[i] + "'";
    reject_topology("decode_window receives at index_offset 0 form a cycle, " +
                    nodes +
                    ", so no window on it can start; give one of these "
                    "receives a negative index_offset");
  }

  for (const auto &e : edges)
    if (windows.count(e.to_node) && e.to_input != "input") {
      const auto &from = fed_by.at(e.from_node).first;
      const auto &to = fed_by.at(e.to_node).first;
      if (from != to)
        reject_topology(describe(e) + " joins the windows of ingest nodes '" +
                        from + "' and '" + to + "'");
      sends[to].push_back(e);
    }
  for (auto a = ingests.begin(); a != ingests.end(); ++a)
    for (auto b = std::next(a); b != ingests.end(); ++b) {
      if (*a->second->dem == *b->second->dem)
        continue;
      const auto from_b = downstream(b->first, edges);
      for (const auto &node : downstream(a->first, edges))
        if (std::find(from_b.begin(), from_b.end(), node) != from_b.end())
          reject_topology("ingest nodes '" + a->first + "' and '" + b->first +
                          "' have different templates but both reach '" + node +
                          "'");
    }

  for (const auto &[id, fed] : fed_by)
    layers[fed.first].push_back(
        {id, fed.second, *layer_request(*ingests.at(fed.first), fed.second)});
  for (const auto &[id, ingest_layers] : layers)
    validate_commitment(templates.at(id), ingest_layers, sends[id]);
} // end - validate_window_topology()

} // namespace cudaq::qec
