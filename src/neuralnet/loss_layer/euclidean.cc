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

#include <glog/logging.h>
#include "singa/neuralnet/loss_layer.h"
#include "mshadow/tensor.h"
#include <math.h>

namespace singa {

using namespace mshadow;
using mshadow::cpu;

using mshadow::Shape1;
using mshadow::Tensor;

using std::vector;

void EuclideanLossLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 2);
  Layer::Setup(conf, srclayers);
}

void EuclideanLossLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  int count = srclayers[0]->data(this).count(); // kaiping: Output of InnerProduct Layer Variant, as we set num_output = 1, so this count is actually batchsize
  CHECK_EQ(count, srclayers[1]->data(this).count());
  const float* reconstruct_dptr = srclayers[0]->data(this).cpu_data(); // kaiping: prediction results, pi
  const float* input_dptr = srclayers[1]->data(this).cpu_data(); // kaiping: Information of LabelLayer, ri
  float loss = 0;
  float aver_p = 0;
  float aver_r = 0;
  float aver_p_square = 0;
  float aver_r_square = 0;
  float aver_p_times_r = 0;
  for (int i = 0; i < count; i++) {
      loss += (input_dptr[i] - reconstruct_dptr[i]) *
        (input_dptr[i] - reconstruct_dptr[i]);
      aver_p += reconstruct_dptr[i];
      aver_r += input_dptr[i];
      aver_p_square += reconstruct_dptr[i] * reconstruct_dptr[i];
      aver_r_square += input_dptr[i] * input_dptr[i];
      aver_p_times_r += reconstruct_dptr[i] * input_dptr[i];
      //LOG(ERROR) << "******Loss Layer****** Ground truth value, Estimate value:  " << input_dptr[i] << "   " << reconstruct_dptr[i];
  }
  loss_ += loss / srclayers[0]->data(this).shape()[0]; // divided by batchsize_
  aver_p_ += aver_p / srclayers[0]->data(this).shape()[0];
  aver_r_ += aver_r / srclayers[0]->data(this).shape()[0];
  aver_p_square_ += aver_p_square / srclayers[0]->data(this).shape()[0];
  aver_r_square_ += aver_r_square / srclayers[0]->data(this).shape()[0];
  aver_p_times_r_ += aver_p_times_r / srclayers[0]->data(this).shape()[0];
  counter_++; // counter_ is the number of batches between each 2 printing
}

void EuclideanLossLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  int count = srclayers[0]->data(this).count();
  CHECK_EQ(count, srclayers[1]->data(this).count());
  const float* reconstruct_dptr = srclayers[0]->data(this).cpu_data();
  const float* input_dptr = srclayers[1]->data(this).cpu_data();
  Blob<float>* gsrcblob = srclayers[0]->mutable_grad(this); // kaiping: from here, we know srclayers[0] is output of InnerProduct Layer Variant
  float* gsrcptr = gsrcblob->mutable_cpu_data();
  for (int i = 0; i < count; i++) {
    gsrcptr[i] = reconstruct_dptr[i]-input_dptr[i]; // kaiping: from here we know loss function = (1/2) * (y-gt)^2
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc /= srclayers[0]->data(this).shape()[0];
}
const std::string EuclideanLossLayer::ToString(bool debug, int flag) {
  if (debug)
    return Layer::ToString(debug, flag);
  aver_p_ = aver_p_ / counter_;
  aver_r_ = aver_r_ / counter_;
  aver_p_square_ = aver_p_square_ / counter_;
  aver_r_square_ = aver_r_square_ / counter_;
  aver_p_times_r_ = aver_p_times_r_ / counter_;

  float nMSE = (aver_p_square_ + aver_r_square_ - 2 * aver_p_times_r_) / (aver_r_square_ - aver_r_ * aver_r_);
  float part1 = sqrt(aver_p_square_ - aver_p_ * aver_p_);
  float part2 = sqrt(aver_r_square_ - aver_r_ * aver_r_);
  float Rvalue = (aver_p_times_r_ - aver_r_ * aver_p_) / (part1 * part2);
  float test = aver_p_square_ + aver_r_square_ - 2 * aver_p_times_r_; // use this value (equal to Loss to check)
//  LOG(ERROR) << "counter:  " << counter_;
//  LOG(ERROR) << "R_top:  " << (aver_p_times_r_ - aver_r_ * aver_p_);
//  LOG(ERROR) << "R_down:  " << (part1 * part2);
//  LOG(ERROR) << "nMSE_top:  " << (aver_p_square_ + aver_r_square_ - 2 * aver_p_times_r_);
//  LOG(ERROR) << "nMSE_down:  " << (aver_r_square_ - aver_r_ * aver_r_);
  string disp = "Loss = " + std::to_string(loss_ / counter_)
    + ", nMSE = " + std::to_string(nMSE)
    + ", R = " + std::to_string(Rvalue)
    + ", testMSE = " + std::to_string(test);
  counter_ = 0;
  loss_ = 0;
  aver_p_ = 0;
  aver_r_ = 0;
  aver_p_square_ = 0;
  aver_r_square_ = 0;
  aver_p_times_r_ = 0;
  return disp;
}
}  // namespace singa
