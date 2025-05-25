/**
 * File: Vocabulary.h
 * Date: February 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated vocabulary
 * License: see the LICENSE.txt file
 *
 */

#pragma once

#include <string>
#include <vector>

#include "BowVector.h"
#include "FeatureVector.h"

namespace DBoW3 {

class Vocabulary {
 public:
  Vocabulary(const std::string& filename);

  ~Vocabulary() = default;

  void transform(const std::vector<cv::Mat>& features, BowVector& v) const;

  void save(const std::string& filename, bool binary_compressed = true) const;

  void load(const std::string& filename);

 protected:
  struct Node {
    NodeId id;
    WordValue weight;
    std::vector<NodeId> children;
    NodeId parent;
    cv::Mat descriptor;
    WordId word_id;

    Node() : id(0), weight(0), parent(0), word_id(0) {}
    Node(NodeId _id) : id(_id), weight(0), parent(0), word_id(0) {}

    bool isLeaf() const { return children.empty(); }
  };

  void transform(const cv::Mat& feature, WordId& id, WordValue& weight) const;

  int m_k;
  int m_L;
  WeightingType m_weighting;
  std::vector<Node> m_nodes;
  std::vector<Node*> m_words;
};

}  // namespace DBoW3
