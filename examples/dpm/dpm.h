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

namespace dpm {
using std::vector;
using singa::LayerProto;
using singa::Layer;
using singa::Param;
using singa::Blob;
using singa::Metric;

/**
 * Combination layer as an extension based on InnerproductLayer, which can handle 2 src inputs
 */
class CombinationLayer : public NeuronLayer {
 public:
  ~CombinationLayer();
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  ConnectionType src_neuron_connection(int k) const override {
    return kOneToAll;
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
