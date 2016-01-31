/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#ifndef EXAMPLES_DPM_DPM_H_
#define EXAMPLES_DPM_DPM_H_

#include <string>
#include <vector>
#include "singa/singa.h"
#include "./dpm.pb.h"
#include "singa/neuralnet/layer.h"
#include "singa/proto/job.pb.h"
#include "singa/proto/common.pb.h"
#include "singa/utils/context.h"
#include "singa/utils/singleton.h"

namespace dpm {
using std::vector;
using singa::LayerProto;
using singa::Layer;
using singa::Param;
using singa::Blob;
using singa::Metric;

/**
 * Input layer that get read records from two data shards: Dynamic Data Shard and Label Data Shard
 */
class DataLayer : public singa::InputLayer {
 public:
  ~DataLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;

  int batchsize() const { return batchsize_; }
  int unroll_len() const { return unroll_len_; }
  int feature_len() const { return feature_len_; }

 private:
  int batchsize_ = 0;
  int unroll_len_ = 1;
  int feature_len_ = 1;
  singa::io::Store* store_ = nullptr;
  singa::io::Store* store2_ = nullptr;
};

/**
 * Unroll layer that has similar functionality with OneHotLayer in CharRNN example (Model V1)
 */
class UnrollLayer : public singa::InputLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers);

 private:
  int batchsize_, feature_len_;
};

/**
 * Time Unroll layer that is modified from Unroll layer to consider time information (Model V2)
 */
class UnrollV2Layer : public singa::InputLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers);
  Blob<float>* laptime_info() const {
   return &laptime_info_;
  }

 private:
  int batchsize_, feature_len_;
  Blob<float> laptime_info_; // a member for storing all lap time information
};

/**
 * DPMGru layer which is modified to consider time information (Model V2)
 */
class DPMGruLayer : public NeuronLayer {
 public:
  ~DPMGruLayer();
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  ConnectionType dst_layer_connection() const override {
    return kOneToMany;
  }
  Blob<float>* mutable_grad(const Layer* from) override {
    if (typeid(*from) == typeid(DPMGruLayer))
      return gradvec_[1];
    else
      return gradvec_[0];
  }
  const Blob<float>& grad(const Layer* from) override {
    if (typeid(*from) == typeid(DPMGruLayer))
      return *gradvec_[1];
    else
      return *gradvec_[0];
  }
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_z_hx_, weight_r_hx_, weight_c_hx_,
      weight_z_hh_, weight_r_hh_, weight_c_hh_, weight_theta_};

    if (bias_z_ != nullptr && bias_r_ != nullptr && bias_c_ != nullptr && bias_theta_ != nullptr) {
      params.push_back(bias_z_);
      params.push_back(bias_r_);
      params.push_back(bias_c_);
      params.push_back(bias_theta_);
    }
    return params;
  }

 private:
  int batchsize_;  // batch size
  int vdim_, hdim_;  // dimensions
  Blob<float> *update_gate_, *reset_gate_, *new_memory_, *new_update_gate_, *time_part_;
  Param *weight_z_hx_, *weight_z_hh_, *bias_z_;  // update gate
  Param *weight_r_hx_, *weight_r_hh_, *bias_r_;  // reset gate
  Param *weight_c_hx_, *weight_c_hh_, *bias_c_;  // new memory
  Param *weight_theta_, *bias_theta_;  // handling time span between each two input records
};


/**
 * Label layer for fetching label information from the src input layer (TimeSpanDataLayer) for DPM models.
 */
class DPMLabelLayer : public singa::InputLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
 private:
  int batchsize_, feature_len_, unroll_len_;
};

/**
 * Time layer for fetching Delta_T information from the src input layer (DataLayer) for DPM models.
 */
class DPMTimeLayer : public singa::InputLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
 private:
  int batchsize_, feature_len_, unroll_len_;
};

/**
 * CombinationLayer as an extension based on InnerproductLayer, which can handle 2 src inputs
 */
class CombinationLayer : public singa::NeuronLayer {
 public:
  ~CombinationLayer();
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  singa::ConnectionType src_neuron_connection(int k) const override {
    return singa::kOneToAll;
  }
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight1_, weight2_, bias_};
    return params;
  }

 private:
  int batchsize_;
  int vdim_, hdim_;
  bool transpose_;
  Param *weight1_, *weight2_, *bias_; // weight1_ is whc, weight2_ is alpha, and bias is b
};

}  // namespace dpm
#endif  // EXAMPLES_DPM_DPM_H_
