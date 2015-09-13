#include "singa.h"
namespace singa {

class RNNLayer : public NeuronLayer {
 public:
  inline int window() { return window_; }

 protected:
  //!< effect window size for bptt
  int window_;
};

/**
 * Word embedding layer that get one row from the embedding matrix for each
 * word based on the word index
 */
class EmbeddingLayer : public RNNLayer {
 public:
  ~EmbeddingLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{embed_};
    return params;
  }


 private:
  int word_dim_;
  int vocab_size_;
  //!< word embedding matrix of size vocab_size_ x word_dim_
  Param* embed_;
};


/**
 * hid[t]=sigmoid(hid[t-1]*W+src[t])
 */
class HiddenLayer : public RNNLayer {
 public:
  ~HiddenLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_};
    return params;
  }


 private:
  Param* weight_;
};

/**
 * p(word at t is from class c)=softmax(src[t]*Wc)[c]
 * p(w|c)=softmax(src[t]*Ww[Start(c):End(c)])
 * p(word at t is w)=p(from class c)*p(w|c)
 */
class OutputLayer : public RNNLayer {
 public:
  ~OutputLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{word_weight_, class_weight_};
    return params;
  }

 private:
  vector<Blob<float>> pword_;
  Blob<float> pclass_;
  Param* word_weight_, *class_weight_;
};
}
