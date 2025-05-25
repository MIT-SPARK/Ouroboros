/**
 * File: Vocabulary.h
 * Date: February 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated vocabulary
 * License: see the LICENSE.txt file
 *
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "BowVector.h"
#include "FeatureVector.h"

namespace DBoW3 {

class Vocabulary {
  friend class FastSearch;

 public:
  Vocabulary(const std::string& filename);

  ~Vocabulary() = default;

  /**
   * Returns the number of words in the vocabulary
   * @return number of words
   */
  unsigned int size() const { return (unsigned int)m_words.size(); }

  /**
   * Returns whether the vocabulary is empty (i.e. it has not been trained)
   * @return true iff the vocabulary is empty
   */
  bool empty() const { return m_words.empty(); }

  /** Clears the vocabulary object
   */
  void clear();
  /**
   * Transforms a set of descriptores into a bow vector
   * @param features
   * @param v (out) bow vector of weighted words
   */
  void transform(const std::vector<cv::Mat>& features, BowVector& v) const;
  /**
   * Transforms a set of descriptores into a bow vector
   * @param features, one per row
   * @param v (out) bow vector of weighted words
   */
  void transform(const cv::Mat& features, BowVector& v) const;
  /**
   * Transform a set of descriptors into a bow vector and a feature vector
   * @param features
   * @param v (out) bow vector
   * @param fv (out) feature vector of nodes and feature indexes
   * @param levelsup levels to go up the vocabulary tree to get the node index
   */
  void transform(const std::vector<cv::Mat>& features, BowVector& v, FeatureVector& fv, int levelsup) const;

  /**
   * Transforms a single feature into a word (without weight)
   * @param feature
   * @return word id
   */
  WordId transform(const cv::Mat& feature) const;

  /**
   * Returns the id of the node that is "levelsup" levels from the word given
   * @param wid word id
   * @param levelsup 0..L
   * @return node id. if levelsup is 0, returns the node id associated to the
   *   word id
   */
  NodeId getParentNode(WordId wid, int levelsup) const;

  /**
   * Returns the ids of all the words that are under the given node id,
   * by traversing any of the branches that goes down from the node
   * @param nid starting node id
   * @param words ids of words
   */
  void getWordsFromNode(NodeId nid, std::vector<WordId>& words) const;

  /**
   * Returns the branching factor of the tree (k)
   * @return k
   */
  int getBranchingFactor() const { return m_k; }

  /**
   * Returns the depth levels of the tree (L)
   * @return L
   */
  int getDepthLevels() const { return m_L; }

  /**
   * Returns the real depth levels of the tree on average
   * @return average of depth levels of leaves
   */
  float getEffectiveLevels() const;

  /**
   * Returns the descriptor of a word
   * @param wid word id
   * @return descriptor
   */
  cv::Mat getWord(WordId wid) const;

  /**
   * Returns the weight of a word
   * @param wid word id
   * @return weight
   */
  WordValue getWordWeight(WordId wid) const;

  /**
   * Returns the weighting method
   * @return weighting method
   */
  WeightingType getWeightingType() const { return m_weighting; }

  /**
   * Changes the weighting method
   * @param type new weighting type
   */
  void setWeightingType(WeightingType type);

  /**
   * Saves the vocabulary into a file. If filename extension contains .yml, opencv YALM format is used. Otherwise,
   * binary format is employed
   * @param filename
   */
  void save(const std::string& filename, bool binary_compressed = true) const;

  /**
   * Loads the vocabulary from a file created with save
   * @param filename.
   */
  void load(const std::string& filename);

  /**
   * Stops those words whose weight is below minWeight.
   * Words are stopped by setting their weight to 0. There are not returned
   * later when transforming image features into vectors.
   * Note that when using IDF or TF_IDF, the weight is the idf part, which
   * is equivalent to -log(f), where f is the frequency of the word
   * (f = Ni/N, Ni: number of training images where the word is present,
   * N: number of training images).
   * Note that the old weight is forgotten, and subsequent calls to this
   * function with a lower minWeight have no effect.
   * @return number of words stopped now
   */
  int stopWords(double minWeight);

  /** Returns the size of the descriptor employed. If the Vocabulary is empty, returns -1
   */
  int getDescritorSize() const;
  /** Returns the type of the descriptor employed normally(8U_C1, 32F_C1)
   */
  int getDescritorType() const;

 protected:
  ///  reference to descriptor
  typedef const cv::Mat pDescriptor;

  /// Tree node
  struct Node {
    /// Node id
    NodeId id;
    /// Weight if the node is a word
    WordValue weight;
    /// Children
    std::vector<NodeId> children;
    /// Parent node (undefined in case of root)
    NodeId parent;
    /// Node descriptor
    cv::Mat descriptor;

    /// Word id if the node is a word
    WordId word_id;

    /**
     * Empty constructor
     */
    Node() : id(0), weight(0), parent(0), word_id(0) {}

    /**
     * Constructor
     * @param _id node id
     */
    Node(NodeId _id) : id(_id), weight(0), parent(0), word_id(0) {}

    /**
     * Returns whether the node is a leaf node
     * @return true iff the node is a leaf
     */
    bool isLeaf() const { return children.empty(); }
  };

 protected:
  /**
   * Returns the word id associated to a feature
   * @param feature
   * @param id (out) word id
   * @param weight (out) word weight
   * @param nid (out) if given, id of the node "levelsup" levels up
   * @param levelsup
   */
  void transform(const cv::Mat& feature, WordId& id, WordValue& weight, NodeId* nid, int levelsup = 0) const;

  /**
   * Returns the word id associated to a feature
   * @param feature
   * @param id (out) word id
   * @param weight (out) word weight
   * @param nid (out) if given, id of the node "levelsup" levels up
   * @param levelsup
   */
  void transform(const cv::Mat& feature, WordId& id, WordValue& weight) const;

  /**
   * Returns the word id associated to a feature
   * @param feature
   * @param id (out) word id
   */
  void transform(const cv::Mat& feature, WordId& id) const;

  /**
   * Create the words of the vocabulary once the tree has been built
   */
  void createWords();

  /**
   * Sets the weights of the nodes of tree according to the given features.
   * Before calling this function, the nodes and the words must be already
   * created (by calling HKmeansStep and createWords)
   * @param features
   */
  void setNodeWeights(const std::vector<std::vector<cv::Mat> >& features);

  /**
   * Writes printable information of the vocabulary
   * @param os stream to write to
   * @param voc
   */
  friend std::ostream& operator<<(std::ostream& os, const Vocabulary& voc);

 protected:
  /// Branching factor
  int m_k;

  /// Depth levels
  int m_L;

  /// Weighting method
  WeightingType m_weighting;

  /// Tree nodes
  std::vector<Node> m_nodes;

  /// Words of the vocabulary (tree leaves)
  /// this condition holds: m_words[wid]->word_id == wid
  std::vector<Node*> m_words;

 public:
  // for debug (REMOVE)
  Node* getNodeWord(uint32_t idx) { return m_words[idx]; }
};

}  // namespace DBoW3
