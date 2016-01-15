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
#include "./dpm.h"

#include <string>
#include <algorithm>
#include "mshadow/tensor.h"
#include "mshadow/tensor_expr.h"
#include "mshadow/cxxnet_op.h"
#include "./dpm.pb.h"

namespace dpm {
using std::vector;
using std::string;

using namespace mshadow;
using mshadow::cpu;
using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Tensor;

/*******CombinationLayer**************/

CombinationLayer::~CombinationLayer() {
  delete weight1_;
  delete weight2_;
  delete bias_;
}

void CombinationLayer::Setup(const LayerProto &proto,
                             const vector<Layer*>& srclayers) {
  Layer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 2); // CombinationLayer has 2 src layers
  const auto& src_gru = srclayers[0]->data(this); // 1st src layer is gru layer
  const auto& src_timespan = srclayers[1]->data(this); // 2nd layer is time_span layer
  batchsize_ = src_gru.shape()[0];
  vdim_ = src_gru.count() / batchsize_;
  hdim_ = conf.GetExtension(combination_conf).num_output();
  transpose_ = conf.GetExtension(combination_conf).tanspose();
  //if (partition_dim() > 0)
  //  hdim_ /= srclayers.at(0)->num_partitions();
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  weight1_ = Param::Create(conf.param(0));
  weight2_ = Param::Create(conf.param(1));
  bias_ = Param::Create(conf.param(2));
  if(transpose_) {
    weight1_->Setup(vector<int>{vdim_,hdim_});
    weight2_->Setup(vector<int>{1,hdim_});
  }
  else { // this is our case
    weight1_->Setup(vector<int>{hdim_,vdim_});
    weight2_->Setup(vector<int>{hdim_,1});
  }
  bias_->Setup(vector<int>{hdim_});
}

void CombinationLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  Blob<float>* tmp = new Blob<float>(batchsize_, hdim_); // Use "tmp" to store the computation result from TimeSpanUnit
  if(transpose_) {
    MMDot(srclayers[0]->data(this), weight1_->data(), &data_); // First part of data_
    MMDot(srclayers[1]->data(this), weight2_->data(), tmp); // Second part of data_
  }
  else {
    MMDot(srclayers[0]->data(this), weight1_->data().T(), &data_); // First part of data_
    MMDot(srclayers[1]->data(this), weight2_->data().T(), tmp); // Second part of data_
  }
  AXPY(1.0f, *tmp, &data_); // Combine 2 parts
  MVAddRow(bias_->data(), &data_); // Add bias
  delete tmp;
}

void CombinationLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  float beta = 0.0f;
  if(flag & kAggGrad)
    beta = 1.0f;
  MVSumRow(1.0f, beta, grad_, bias_->mutable_grad());
  if(transpose_) {
    GEMM(1.0f, beta, srclayers[0]->data(this).T(), grad_,
         weight1_->mutable_grad());
    GEMM(1.0f, beta, srclayers[1]->data(this).T(), grad_,
         weight2_->mutable_grad());
  }
  else {
    GEMM(1.0f, beta, grad_.T(), srclayers[0]->data(this),
         weight1_->mutable_grad());
    GEMM(1.0f, beta, grad_.T(), srclayers[1]->data(this),
         weight2_->mutable_grad());
  }

  if (srclayers[0]->mutable_grad(this) != nullptr) { // Compute the gradient for src_layer: gru layer; Note that gradient for src_layer: Delta_T is DataLayer, should be no grad_
    if (transpose_)
      MMDot(grad_, weight_->data().T(), srclayers[0]->mutable_grad(this));
    else
      MMDot(grad_, weight_->data(), srclayers[0]->mutable_grad(this));
  }
}

}   // end of namespace dpm
