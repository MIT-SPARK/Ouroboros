#include "Vocabulary.h"

#include <fstream>

namespace DBoW3 {

namespace {

template <typename T>
double distance(const Descriptor<T>& a, const Descriptor<T>& b) {
  return (a - b).squaredNorm();
}

double distance(const Descriptor<uint8_t>& a, const Descriptor<uint8_t>& b) {
  // binary descriptor

  // Bit count function got from:
  // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
  // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0

  const uint64_t *pa, *pb;
  pa = a.ptr<uint64_t>();  // a & b are actually CV_8U
  pb = b.ptr<uint64_t>();

  uint64_t v, ret = 0;
  int n = a.cols / sizeof(uint64_t);
  for (size_t i = 0; i < n; ++i, ++pa, ++pb) {
    v = *pa ^ *pb;
    v = v - ((v >> 1) & (uint64_t) ~(uint64_t)0 / 3);
    v = (v & (uint64_t) ~(uint64_t)0 / 15 * 3) + ((v >> 2) & (uint64_t) ~(uint64_t)0 / 15 * 3);
    v = (v + (v >> 4)) & (uint64_t) ~(uint64_t)0 / 255 * 15;
    ret += (uint64_t)(v * ((uint64_t) ~(uint64_t)0 / 255)) >> (sizeof(uint64_t) - 1) * CHAR_BIT;
  }

  return ret;
}

inline double distance(const Eigen::VectorXf& a, const Eigen::VectorXf& b) { return (a - b).squaredNorm(); }

}  // namespace

Vocabulary::Vocabulary(const std::string& filename) { load(filename); }

void Vocabulary::transform(const Eigen::MatrixXf& features, BowVector& v) const {
  v.clear();
  if (m_words.empty()) {
    return;
  }

  for (int c = 0; c < features.cols(); ++c) {
    WordId id;
    WordValue w;
    const auto& feat = features.col(c);
    // w is the idf value if TF_IDF, 1 if TF
    transform(feat, id, w);

    if (m_weighting == TF || m_weighting == TF_IDF) {
      if (w > 0) {  // not stopped
        v.addWeight(id, w);
      }

      if (!v.empty()) {
        // unnecessary when normalizing
        const double nd = v.size();
        for (auto vit = v.begin(); vit != v.end(); vit++) {
          vit->second /= nd;
        }
      }
    } else {
      // IDF || BINARY
      if (w > 0) {  // not stopped
        v.addIfNotExist(id, w);
      }
    }
  }
}

void Vocabulary::transform(const Eigen::VectorXf& feature, WordId& word_id, WordValue& weight) const {
  // propagate the feature down the tree
  // level at which the node must be stored in nid, if given
  NodeId final_id = 0;  // root

  do {
    auto const& nodes = m_nodes[final_id].children;
    uint64_t best_d = std::numeric_limits<uint64_t>::max();
    int idx = 0, bestidx = 0;
    for (const auto& id : nodes) {
      uint64_t dist = distance(feature, m_nodes[id].descriptor);
      if (dist < best_d) {
        best_d = dist;
        final_id = id;
        bestidx = idx;
      }

      idx++;
    }
  } while (!m_nodes[final_id].isLeaf());

  // turn node id into word id
  word_id = m_nodes[final_id].word_id;
  weight = m_nodes[final_id].weight;
}

void Vocabulary::save(const std::string& filename, bool compressed) const {
  std::ofstream file_out(filename, std::ios::binary);
  if (!file_out) {
    throw std::runtime_error("Vocabulary::saveBinary Could not open file :" + filename + " for writing");
  }

  uint64_t sig = 88877711233;  // magic number describing the file
  file_out.write((char*)&sig, sizeof(sig));
  file_out.write((char*)&compressed, sizeof(compressed));
  uint32_t nnodes = m_nodes.size();
  file_out.write((char*)&nnodes, sizeof(nnodes));
  if (nnodes == 0) {
    return;
  }

  std::stringstream aux_stream;
  aux_stream.write((char*)&m_k, sizeof(m_k));
  aux_stream.write((char*)&m_L, sizeof(m_L));
  aux_stream.write((char*)&m_scoring, sizeof(m_scoring));
  aux_stream.write((char*)&m_weighting, sizeof(m_weighting));

  std::vector<NodeId> parents = {0};
  while (!parents.empty()) {
    NodeId pid = parents.back();
    parents.pop_back();

    const auto& parent = m_nodes[pid];
    for (auto pit : parent.children) {
      const auto& child = m_nodes[pit];
      aux_stream.write((char*)&child.id, sizeof(child.id));
      aux_stream.write((char*)&pid, sizeof(pid));
      aux_stream.write((char*)&child.weight, sizeof(child.weight));
      DescManip::toStream(child.descriptor, aux_stream);
      if (!child.isLeaf()) {
        parents.push_back(pit);
      }
    }
  }

  // words
  uint32_t m_words_size = m_words.size();
  aux_stream.write((char*)&m_words_size, sizeof(m_words_size));
  for (auto wit = m_words.begin(); wit != m_words.end(); wit++) {
    WordId id = wit - m_words.begin();
    aux_stream.write((char*)&id, sizeof(id));
    aux_stream.write((char*)&(*wit)->id, sizeof((*wit)->id));
  }

  if (compressed) {
    throw std::runtime_error("TBD");
  } else {
    file_out << aux_stream.rdbuf();
  }
}

bool Vocabulary::load(const std::string& filename) {
  // check first if it is a binary file
  std::ifstream ifile(filename, std::ios::binary);
  if (!ifile) {
    throw std::runtime_error("Vocabulary::load Could not open file :" + filename + " for reading");
  }

  m_words.clear();
  m_nodes.clear();
  uint64_t sig = 0;  // magic number describing the file
  ifile.read((char*)&sig, sizeof(sig));
  if (sig != 88877711233) {
    throw std::runtime_error("Vocabulary::fromStream  is not of appropriate type");
  }

  bool compressed;
  ifile.read((char*)&compressed, sizeof(compressed));
  uint32_t nnodes;
  ifile.read((char*)&nnodes, sizeof(nnodes));
  if (nnodes == 0) {
    return true;
  }

  std::istream* _used_str = 0;
  if (compressed) {
    throw std::runtime_erorr("TBD");
  } else {
    _used_str = &ifile;
  }

  _used_str->read((char*)&m_k, sizeof(m_k));
  _used_str->read((char*)&m_L, sizeof(m_L));
  _used_str->read((char*)&m_scoring, sizeof(m_scoring));
  _used_str->read((char*)&m_weighting, sizeof(m_weighting));

  m_nodes.resize(nnodes);
  m_nodes[0].id = 0;

  for (size_t i = 1; i < m_nodes.size(); ++i) {
    NodeId nid;
    _used_str->read((char*)&nid, sizeof(NodeId));
    Node& child = m_nodes[nid];
    child.id = nid;
    _used_str->read((char*)&child.parent, sizeof(child.parent));
    _used_str->read((char*)&child.weight, sizeof(child.weight));
    DescManip::fromStream(child.descriptor, *_used_str);
    m_nodes[child.parent].children.push_back(child.id);
  }

  uint32_t m_words_size;
  _used_str->read((char*)&m_words_size, sizeof(m_words_size));
  m_words.resize(m_words_size);
  for (unsigned int i = 0; i < m_words.size(); ++i) {
    WordId wid;
    NodeId nid;
    _used_str->read((char*)&wid, sizeof(wid));
    _used_str->read((char*)&nid, sizeof(nid));
    m_nodes[nid].word_id = wid;
    m_words[wid] = &m_nodes[nid];
  }

  return true;
}

}  // namespace DBoW3
