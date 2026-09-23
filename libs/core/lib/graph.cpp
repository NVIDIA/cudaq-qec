/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda-qx/core/graph.h"

#include <algorithm>

namespace cudaqx {

void graph::add_edge(int u, int v, double weight) {
  // Check if the edge already exists
  auto it_u = std::find_if(adjacency_list[u].begin(), adjacency_list[u].end(),
                           [v](const auto &p) { return p.first == v; });

  if (it_u == adjacency_list[u].end()) {
    // Edge doesn't exist, so add it
    adjacency_list[u].push_back({v, weight});
    adjacency_list[v].push_back({u, weight});
  }
  // If the edge already exists, do nothing
}

void graph::clear() {
  adjacency_list.clear();
  node_weights.clear();
}

void graph::add_node(int node, double weight) {
  if (adjacency_list.find(node) == adjacency_list.end()) {
    adjacency_list[node] = std::vector<std::pair<int, double>>();
    node_weights[node] = weight;
  }
}

void graph::set_node_weight(int node, double weight) {
  node_weights[node] = weight;
}

double graph::get_node_weight(int node) const {
  auto it = node_weights.find(node);
  return (it != node_weights.end()) ? it->second : 0.0;
}

bool graph::edge_exists(int i, int j) const {
  auto it_i = adjacency_list.find(i);
  if (it_i != adjacency_list.end()) {
    // Search for node j in i's adjacency list
    return std::any_of(it_i->second.begin(), it_i->second.end(),
                       [j](const auto &pair) { return pair.first == j; });
  }
  return false;
}

std::vector<std::pair<int, int>> graph::get_disconnected_vertices() const {
  std::vector<std::pair<int, int>> disconnected_pairs;
  auto nodes = get_nodes();
  for (auto ni : nodes)
    for (auto nj : nodes)
      if (ni < nj)
        if (!edge_exists(ni, nj))
          disconnected_pairs.push_back({ni, nj});

  // Sort the pairs for consistent ordering
  std::sort(disconnected_pairs.begin(), disconnected_pairs.end());

  return disconnected_pairs;
}

std::vector<int> graph::get_neighbors(int node) const {
  auto it = adjacency_list.find(node);
  if (it != adjacency_list.end()) {
    std::vector<int> neighbors;
    neighbors.reserve(it->second.size());

    // Extract node IDs from pairs
    for (const auto &pair : it->second) {
      neighbors.push_back(pair.first);
    }

    // Sort to ensure consistent ordering
    std::sort(neighbors.begin(), neighbors.end());

    return neighbors;
  }
  return std::vector<int>();
}

std::vector<std::pair<int, double>>
graph::get_weighted_neighbors(int node) const {
  auto it = adjacency_list.find(node);
  if (it != adjacency_list.end()) {
    return it->second;
  }
  return std::vector<std::pair<int, double>>();
}
std::vector<int> graph::get_nodes() const {
  std::vector<int> nodes;
  for (const auto &pair : adjacency_list) {
    nodes.push_back(pair.first);
  }
  return nodes;
}

int graph::num_nodes() const { return adjacency_list.size(); }

int graph::num_edges() const {
  int total = 0;
  for (const auto &pair : adjacency_list) {
    total += pair.second.size();
  }
  return total / 2; // Each edge is counted twice
}

void graph::remove_edge(int u, int v) {
  auto it_u = adjacency_list.find(u);
  auto it_v = adjacency_list.find(v);

  if (it_u != adjacency_list.end()) {
    it_u->second.erase(
        std::remove_if(it_u->second.begin(), it_u->second.end(),
                       [v](const auto &p) { return p.first == v; }),
        it_u->second.end());
  }

  if (it_v != adjacency_list.end()) {
    it_v->second.erase(
        std::remove_if(it_v->second.begin(), it_v->second.end(),
                       [u](const auto &p) { return p.first == u; }),
        it_v->second.end());
  }
}
void graph::remove_node(int node) {
  adjacency_list.erase(node);
  node_weights.erase(node);
  for (auto &pair : adjacency_list) {
    pair.second.erase(
        std::remove_if(pair.second.begin(), pair.second.end(),
                       [node](const auto &p) { return p.first == node; }),
        pair.second.end());
  }
}
void graph::dfs(int node, std::unordered_set<int> &visited) const {
  visited.insert(node);
  for (int neighbor : get_neighbors(node)) {
    if (visited.find(neighbor) == visited.end()) {
      dfs(neighbor, visited);
    }
  }
}

bool graph::is_connected() const {
  if (adjacency_list.empty()) {
    return true; // Empty graph is considered connected
  }

  std::unordered_set<int> visited;
  int start_node = adjacency_list.begin()->first;
  dfs(start_node, visited);

  return visited.size() == adjacency_list.size();
}

int graph::get_degree(int node) const {
  auto it = adjacency_list.find(node);
  if (it != adjacency_list.end()) {
    return it->second.size();
  }
  return 0; // Node not found
}

double graph::get_edge_weight(int u, int v) const {
  auto it_u = adjacency_list.find(u);
  if (it_u != adjacency_list.end()) {
    auto edge_it = std::find_if(it_u->second.begin(), it_u->second.end(),
                                [v](const auto &p) { return p.first == v; });
    if (edge_it != it_u->second.end()) {
      return edge_it->second;
    }
  }
  return -1.0; // Edge not found
}

bool graph::update_edge_weight(int u, int v, double weight) {
  auto it_u = adjacency_list.find(u);
  auto it_v = adjacency_list.find(v);

  bool updated = false;

  if (it_u != adjacency_list.end()) {
    auto edge_it = std::find_if(it_u->second.begin(), it_u->second.end(),
                                [v](const auto &p) { return p.first == v; });
    if (edge_it != it_u->second.end()) {
      edge_it->second = weight;
      updated = true;
    }
  }

  if (it_v != adjacency_list.end()) {
    auto edge_it = std::find_if(it_v->second.begin(), it_v->second.end(),
                                [u](const auto &p) { return p.first == u; });
    if (edge_it != it_v->second.end()) {
      edge_it->second = weight;
      updated = true;
    }
  }

  return updated;
}

} // namespace cudaqx
