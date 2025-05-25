#include "Vocabulary.h"

#include <fstream>

// #include "DescManip.h"

namespace DBoW3 {

Vocabulary::Vocabulary(const std::string& filename) { load(filename); }

void Vocabulary::transform(const std::vector<cv::Mat>& features, BowVector& v) const {
  v.clear();

  if (m_words.empty()) {
    return;
  }

  if (m_weighting == TF || m_weighting == TF_IDF) {
    for (auto fit = features.begin(); fit < features.end(); ++fit) {
      WordId id;
      WordValue w;
      // w is the idf value if TF_IDF, 1 if TF
      transform(*fit, id, w);
      if (w > 0) {  // not stopped
        v.addWeight(id, w);
      }
    }

    if (!v.empty() && !must) {
      // unnecessary when normalizing
      const double nd = v.size();
      for (BowVector::iterator vit = v.begin(); vit != v.end(); vit++) {
        vit->second /= nd;
      }
    }
  } else {
    // IDF || BINARY
    for (auto fit = features.begin(); fit < features.end(); ++fit) {
      WordId id;
      WordValue w; // w is idf if IDF, or 1 if BINARY
      transform(*fit, id, w);
      if (w > 0) { // not stopped
        v.addIfNotExist(id, w);
      }
    }  // if add_features
  }  // if m_weighting == ...
}

void Vocabulary::transform(const cv::Mat& feature, WordId& word_id, WordValue& weight) const {
  // propagate the feature down the tree
  // level at which the node must be stored in nid, if given
  NodeId final_id = 0;  // root

  // binary descriptor
  // int ntimes=0;
  if (feature.type() == CV_8U) {
    do {
      auto const& nodes = m_nodes[final_id].children;
      uint64_t best_d = std::numeric_limits<uint64_t>::max();
      int idx = 0, bestidx = 0;
      for (const auto& id : nodes) {
        // compute distance
        uint64_t dist = DescManip::distance_8uc1(feature, m_nodes[id].descriptor);
        if (dist < best_d) {
          best_d = dist;
          final_id = id;
          bestidx = idx;
        }
        idx++;
      }
    } while (!m_nodes[final_id].isLeaf());
  } else {
    do {
      auto const& nodes = m_nodes[final_id].children;
      uint64_t best_d = std::numeric_limits<uint64_t>::max();
      int idx = 0, bestidx = 0;
      for (const auto& id : nodes) {
        // compute distance
        uint64_t dist = DescManip::distance(feature, m_nodes[id].descriptor);
        if (dist < best_d) {
          best_d = dist;
          final_id = id;
          bestidx = idx;
        }
        idx++;
      }
    } while (!m_nodes[final_id].isLeaf());
  }

  // turn node id into word id
  word_id = m_nodes[final_id].word_id;
  weight = m_nodes[final_id].weight;
}

void Vocabulary::save(const std::string& filename, bool binary_compressed) const {
  std::ofstream file_out(filename, std::ios::binary);
  if (!file_out) {
    throw std::runtime_error("Vocabulary::saveBinary Could not open file :" + filename + " for writing");
  }

  toStream(file_out, binary_compressed);
}

void Vocabulary::load(const std::string& filename) {
  // check first if it is a binary file
  std::ifstream ifile(filename, std::ios::binary);
  if (!ifile) {
    throw std::runtime_error("Vocabulary::load Could not open file :" + filename + " for reading");
  }

  if (!load(ifile)) {
    if (filename.find(".txt") != std::string::npos) {
      load_fromtxt(filename);
    }
  }
}

bool Vocabulary::load(std::istream& ifile) {
  uint64_t sig;  // magic number describing the file
  ifile.read((char*)&sig, sizeof(sig));
  if (sig != 88877711233) {
    return false;
  }

  ifile.seekg(0, std::ios::beg);
  fromStream(ifile);
  return true;
}

void Vocabulary::toStream(std::ostream& out_str, bool compressed) const {
  uint64_t sig = 88877711233;  // magic number describing the file
  out_str.write((char*)&sig, sizeof(sig));
  out_str.write((char*)&compressed, sizeof(compressed));
  uint32_t nnodes = m_nodes.size();
  out_str.write((char*)&nnodes, sizeof(nnodes));
  if (nnodes == 0) return;
  // save everything to a stream
  std::stringstream aux_stream;
  aux_stream.write((char*)&m_k, sizeof(m_k));
  aux_stream.write((char*)&m_L, sizeof(m_L));
  aux_stream.write((char*)&m_scoring, sizeof(m_scoring));
  aux_stream.write((char*)&m_weighting, sizeof(m_weighting));
  // nodes
  std::vector<NodeId> parents = {0};  // root

  while (!parents.empty()) {
    NodeId pid = parents.back();
    parents.pop_back();

    const Node& parent = m_nodes[pid];

    for (auto pit : parent.children) {
      const Node& child = m_nodes[pit];
      aux_stream.write((char*)&child.id, sizeof(child.id));
      aux_stream.write((char*)&pid, sizeof(pid));
      aux_stream.write((char*)&child.weight, sizeof(child.weight));
      DescManip::toStream(child.descriptor, aux_stream);
      // add to parent list
      if (!child.isLeaf()) parents.push_back(pit);
    }
  }
  // words
  // save size
  uint32_t m_words_size = m_words.size();
  aux_stream.write((char*)&m_words_size, sizeof(m_words_size));
  for (auto wit = m_words.begin(); wit != m_words.end(); wit++) {
    WordId id = wit - m_words.begin();
    aux_stream.write((char*)&id, sizeof(id));
    aux_stream.write((char*)&(*wit)->id, sizeof((*wit)->id));
  }

  // now, decide if compress or not
  if (compressed) {
    qlz_state_compress state_compress;
    memset(&state_compress, 0, sizeof(qlz_state_compress));
    // Create output buffer
    int chunkSize = 10000;
    std::vector<char> compressed(chunkSize + size_t(400), 0);
    std::vector<char> input(chunkSize, 0);
    int64_t total_size = static_cast<int64_t>(aux_stream.tellp());
    uint64_t total_compress_size = 0;
    // calculate how many chunks will be written
    uint32_t nChunks = total_size / chunkSize;
    if (total_size % chunkSize != 0) nChunks++;
    out_str.write((char*)&nChunks, sizeof(nChunks));
    // start compressing the chunks
    while (total_size != 0) {
      int readSize = chunkSize;
      if (total_size < chunkSize) readSize = total_size;
      aux_stream.read(&input[0], readSize);
      uint64_t compressed_size = qlz_compress(&input[0], &compressed[0], readSize, &state_compress);
      total_size -= readSize;
      out_str.write(&compressed[0], compressed_size);
      total_compress_size += compressed_size;
    }
  } else {
    out_str << aux_stream.rdbuf();
  }
}

void Vocabulary::load_fromtxt(const std::string& filename) {
  std::ifstream ifile(filename);
  if (!ifile) throw std::runtime_error("Vocabulary:: load_fromtxt  Could not open file for reading:" + filename);
  int n1, n2;
  {
    std::string str;
    getline(ifile, str);
    std::stringstream ss(str);
    ss >> m_k >> m_L >> n1 >> n2;
  }
  if (m_k < 0 || m_k > 20 || m_L < 1 || m_L > 10 || n1 < 0 || n1 > 5 || n2 < 0 || n2 > 3)
    throw std::runtime_error("Vocabulary loading failure: This is not a correct text file!");

  m_scoring = (ScoringType)n1;
  m_weighting = (WeightingType)n2;
  // nodes
  int expected_nodes = (int)((pow((double)m_k, (double)m_L + 1) - 1) / (m_k - 1));
  m_nodes.reserve(expected_nodes);

  m_words.reserve(pow((double)m_k, (double)m_L + 1));

  m_nodes.resize(1);
  m_nodes[0].id = 0;

  int counter = 0;
  while (!ifile.eof()) {
    std::string snode;
    getline(ifile, snode);
    if (counter++ % 100 == 0) std::cerr << ".";
    // std::cout<<snode<<std::endl;
    if (snode.size() == 0) break;
    std::stringstream ssnode(snode);

    int nid = m_nodes.size();
    m_nodes.resize(m_nodes.size() + 1);
    m_nodes[nid].id = nid;

    int pid;
    ssnode >> pid;
    m_nodes[nid].parent = pid;
    m_nodes[pid].children.push_back(nid);

    int nIsLeaf;
    ssnode >> nIsLeaf;

    // read until the end and add to data
    std::vector<float> data;
    data.reserve(100);
    float d;
    while (ssnode >> d) data.push_back(d);
    // the weight is the last
    m_nodes[nid].weight = data.back();
    data.pop_back();  // remove
    // the rest, to the descriptor
    m_nodes[nid].descriptor.create(1, data.size(), CV_8UC1);
    auto ptr = m_nodes[nid].descriptor.ptr<uchar>(0);
    for (auto d : data) *ptr++ = d;

    if (nIsLeaf > 0) {
      int wid = m_words.size();
      m_words.resize(wid + 1);

      m_nodes[nid].word_id = wid;
      m_words[wid] = &m_nodes[nid];
    } else {
      m_nodes[nid].children.reserve(m_k);
    }
  }
}

void Vocabulary::fromStream(std::istream& str) {
  m_words.clear();
  m_nodes.clear();
  uint64_t sig = 0;  // magic number describing the file
  str.read((char*)&sig, sizeof(sig));
  if (sig != 88877711233) {
    throw std::runtime_error("Vocabulary::fromStream  is not of appropriate type");
  }

  bool compressed;
  str.read((char*)&compressed, sizeof(compressed));
  uint32_t nnodes;
  str.read((char*)&nnodes, sizeof(nnodes));
  if (nnodes == 0) return;
  std::stringstream decompressed_stream;
  std::istream* _used_str = 0;
  if (compressed) {
    qlz_state_decompress state_decompress;
    memset(&state_decompress, 0, sizeof(qlz_state_decompress));
    int chunkSize = 10000;
    std::vector<char> decompressed(chunkSize);
    std::vector<char> input(chunkSize + 400);
    // read how many chunks are there
    uint32_t nChunks;
    str.read((char*)&nChunks, sizeof(nChunks));
    for (int i = 0; i < nChunks; i++) {
      str.read(&input[0], 9);
      int c = qlz_size_compressed(&input[0]);
      str.read(&input[9], c - 9);
      size_t d = qlz_decompress(&input[0], &decompressed[0], &state_decompress);
      decompressed_stream.write(&decompressed[0], d);
    }
    _used_str = &decompressed_stream;
  } else {
    _used_str = &str;
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
}

}  // namespace DBoW3
