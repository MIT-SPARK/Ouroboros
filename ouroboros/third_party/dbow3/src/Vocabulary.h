/**
 * File: Vocabulary.h
 * Date: February 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated vocabulary
 * License: see the LICENSE.txt file
 *
 */

#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "BowVector.h"

namespace DBoW3 {

template <typename T>
using Descriptors = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using Descriptor = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
class Vocabulary {
 public:
  Vocabulary(const std::string& filename);

  ~Vocabulary() = default;

  void transform(const Descriptors<T>& features, BowVector& v) const;

  void save(const std::string& filename, bool compressed = true) const;

  bool load(const std::string& filename);

 protected:
  struct Node {
    NodeId id;
    WordValue weight;
    std::vector<NodeId> children;
    NodeId parent;
    Descriptor<T> descriptor;
    WordId word_id;

    Node() : id(0), weight(0), parent(0), word_id(0) {}
    Node(NodeId _id) : id(_id), weight(0), parent(0), word_id(0) {}

    bool isLeaf() const { return children.empty(); }
  };

  void lookup(const Descriptor<T>& feature, WordId& id, WordValue& weight) const;

  int m_k;
  int m_L;
  WeightingType m_weighting;
  ScoringType m_scoring;
  std::vector<Node> m_nodes;
  std::vector<Node*> m_words;
};

}  // namespace DBoW3
