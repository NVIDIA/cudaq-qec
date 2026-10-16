/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cudaqx {

/// @brief A class representing an undirected weighted graph
class graph {
private:
  /// Adjacency list representation of the graph with weights
  /// Maps node ID to vector of (neighbor_id, weight) pairs
  std::unordered_map<int, std::vector<std::pair<int, double>>> adjacency_list;

  /// Node weights storage
  std::unordered_map<int, double> node_weights;

  /// @brief Depth-first search helper function
  /// @param node The starting node for DFS
  /// @param visited Set of visited nodes
  void dfs(int node, std::unordered_set<int> &visited) const;

public:
  /// @brief Add a weighted edge between two nodes
  /// @param u First node
  /// @param v Second node
  /// @param weight Edge weight
  void add_edge(int u, int v, double weight = 1.0);

  /// @brief Add a node to the graph
  /// @param node The node to add
  /// @param weight Node weight
  void add_node(int node, double weight = 1.0);

  /// @brief Check if an edge exists between two nodes
  /// @param i First node
  /// @param j Second node
  /// @return True if edge exists, false otherwise
  bool edge_exists(int i, int j) const;

  /// @brief Set the weight of a node
  /// @param node The node to set weight for
  /// @param weight The weight value
  void set_node_weight(int node, double weight);

  /// @brief Get the weight of a node
  /// @param node The node to get weight for
  /// @return Node weight, or 0.0 if node doesn't exist
  double get_node_weight(int node) const;

  /// @brief Remove an edge between two nodes
  /// @param u First node
  /// @param v Second node
  void remove_edge(int u, int v);

  /// @brief Remove a node and all its incident edges from the graph
  /// @param node The node to remove
  void remove_node(int node);

  /// @brief Get the neighbors of a node
  /// @param node The node to get neighbors for
  /// @return Vector of neighboring node IDs
  std::vector<int> get_neighbors(int node) const;

  /// @brief Get the neighbors of a node with their weights
  /// @param node The node to get neighbors for
  /// @return Vector of pairs containing (neighbor_id, weight)
  std::vector<std::pair<int, double>> get_weighted_neighbors(int node) const;

  /// @brief Get all pairs of vertices that are not connected
  /// @return Vector of pairs representing disconnected vertices
  std::vector<std::pair<int, int>> get_disconnected_vertices() const;

  /// @brief Get all nodes in the graph
  /// @return Vector of all nodes
  std::vector<int> get_nodes() const;

  /// @brief Get the number of nodes in the graph
  /// @return Number of nodes
  int num_nodes() const;

  /// @brief Get the number of edges in the graph
  /// @return Number of edges
  int num_edges() const;

  /// @brief Check if the graph is connected
  /// @return True if the graph is connected, false otherwise
  bool is_connected() const;

  /// @brief Get the degree of a node
  /// @param node The node to get the degree for
  /// @return Degree of the node
  int get_degree(int node) const;

  /// @brief Get the weight of an edge between two nodes
  /// @param u First node
  /// @param v Second node
  /// @return Edge weight, or -1 if edge doesn't exist
  double get_edge_weight(int u, int v) const;

  /// @brief Update the weight of an existing edge
  /// @param u First node
  /// @param v Second node
  /// @param weight New edge weight
  /// @return True if edge exists and weight was updated, false otherwise
  bool update_edge_weight(int u, int v, double weight);

  /// @brief Clear all nodes and edges from the graph
  void clear();
};

} // namespace cudaqx
