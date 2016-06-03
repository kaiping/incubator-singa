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
// Originally this is from dpm_nuh.cc file
#include "./dpm.h"

#include <string>
#include <algorithm>
#include "mshadow/tensor.h"
#include "mshadow/tensor_expr.h"
#include "mshadow/cxxnet_op.h"
#include "singa/utils/singleton.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/singa_op.h"
#include "./dpm.pb.h"
#include "../../include/singa/utils/singa_op.h"
#include "dpm_for_printing_data.h"

namespace dpm {
using std::vector;
using std::string;

using namespace mshadow;
//using namespace singa;
using mshadow::cpu;
using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Tensor;


/*******DataLayer**************/
DataLayer::~DataLayer() {
  if (store_ != nullptr)
    delete store_;
  if (store2_ != nullptr)
    delete store2_;
}

void DataLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "Setup @ Data";
  InputLayer::Setup(conf, srclayers);
  batchsize_ = conf.GetExtension(data_conf).batchsize();
  unroll_len_ = conf.GetExtension(data_conf).unroll_len();
  // e.g., feature_len = 598 (age, edu, gen, lap + feature_values (e.g., 592) + deltaT + MMSCORE)
  feature_len_ = conf.GetExtension(data_conf).feature_len();
  //LOG(ERROR) << "Batch size: " << batchsize_;
  //LOG(ERROR) << "Unroll length: " << unroll_len_;
  //LOG(ERROR) << "Feature length: " << feature_len_;
  datavec_.clear();
  // each unroll layer has a input blob
  for (int i = 0; i <= unroll_len_; i++) {
    datavec_.push_back(new Blob<float>(batchsize_,feature_len_)); // including all features: 598-dim
  }
}

void DataLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "This is DataLayer's ComputeFeature";
  string key, value;
  string key2, value2;
  DynamicRecord dynamic;
  OutTimeRecord outtime;
  //LOG(ERROR) << "Comp @ Data ----------";
  if (store_ == nullptr) {
    store_ = singa::io::OpenStore(
        layer_conf_.GetExtension(data_conf).backend(),
        layer_conf_.GetExtension(data_conf).path(),
        singa::io::kRead);
  }
  if (store2_ == nullptr) {
    store2_ = singa::io::OpenStore(
        layer_conf_.GetExtension(data_conf).backend(),
        layer_conf_.GetExtension(data_conf).label_path(),
        singa::io::kRead);
  }

  // initialize data: this is to make sure that GRU units with no records are filled with 0
  for (int l=0; l<unroll_len_; l++) {
      float* ptr = datavec_[l]->mutable_cpu_data();
      memset(ptr, 0, sizeof(float) * datavec_[l]->count());
  }

  for (int b = 0; b < batchsize_; b++) {

     int l=0; // idx of unroll layers
     int scnt=0; // number of samples for one patient

     while(1) {
        if (!store_->Read(&key, &value)) {
           store_->SeekToFirst();
           CHECK(store_->Read(&key, &value));
           store2_->SeekToFirst();
        }
        dynamic.ParseFromString(value);

        if (dynamic.patient_id() == -1) { // dummy record to separate each 2 patients' information

           if (!store2_->Read(&key2, &value2)) {
              store2_->SeekToFirst();
              CHECK(store2_->Read(&key2, &value2));
           }
           outtime.ParseFromString(value2);

           float* ptr = datavec_[unroll_len_-1]->mutable_cpu_data(); // Only store label info in last unrolled unit
           ptr[b * feature_len_ + feature_len_ - 2 + 0] = static_cast<float>(outtime.delta_time()); // the second last feature
           ptr[b * feature_len_ + feature_len_ - 2 + 1] = static_cast<float>(outtime.mmscore()); // the last feature

           //LOG(ERROR) << "label: batch " << b << ", dt: " << static_cast<float>(outtime.delta_time()) << ", mm: " << static_cast<float>(outtime.mmscore());

           scnt = 0;
           break; // information for 1 patient is finished
        }

        if (scnt++ == 0) { // A new patient
           // Get index of unroll (gru unit). Fill it from the end
           l = unroll_len_ - static_cast<int>(dynamic.nb_sample());
        }
        float* ptr = datavec_[l]->mutable_cpu_data();
        ptr[b * feature_len_ + 0] = static_cast<float>(dynamic.age());  // feature_len_ is 596 (4 <3 demo features and lap_time> + 592)
        //ptr[b * feature_len_ + 1] = static_cast<float>(dynamic.education());
        ptr[b * feature_len_ + 1] = static_cast<float>(dynamic.gender());
        ptr[b * feature_len_ + 2] = static_cast<float>(dynamic.lap_time());
//         LOG(ERROR) << "Patient ID: " << dynamic.patient_id() << " age " << ptr[b * feature_len_ + 0];
//         LOG(ERROR) << "Patient ID: " << dynamic.patient_id() << " edu " << ptr[b * feature_len_ + 1];
//         LOG(ERROR) << "Patient ID: " << dynamic.patient_id() << " gender " << ptr[b * feature_len_ + 2];
//         LOG(ERROR) << "Patient ID: " << dynamic.patient_id() << " lt " << ptr[b * feature_len_ + 3];

         // all zero
        for (int i=3; i<feature_len_-2; i++)
           ptr[b * feature_len_ + i] = 0;
        // except the observed
        for (int k=0; k<dynamic.feature_value_size(); k++) {
           int idx = dynamic.observed_idx(k);
           ptr[b * feature_len_ + 3 + idx] = static_cast<float>(dynamic.feature_value(k));
           //LOG(ERROR) << "Patient ID: " << dynamic.patient_id() << " Detailed features: " << "(l,b)=(" << l << "," << b << ")" << static_cast<float>(dynamic.feature_value(k));
        }
//        for (int i=4; i<feature_len_-2; i++) {
//            ptr[b * feature_len_ + i] = static_cast<float>(dynamic.feature_value(i - 4));
//            LOG(ERROR) << "Patient ID: " << dynamic.patient_id() << " Detailed features: " << static_cast<float>(dynamic.feature_value(i - 4));
//        }

        //LOG(ERROR) << "(l,b)=(" << l << "," << b << "), pid " << static_cast<int>(dynamic.patient_id())
                                                 //<< ", lap: " << static_cast<int>(dynamic.lap_time());

        l++;
     }
  }
}

/*******UnrollLayer**************/
void UnrollLayer::Setup(const LayerProto& conf,
  const vector<Layer*>& srclayers) {
  InputLayer::Setup(conf, srclayers);
  batchsize_ = srclayers.at(0)->data(unroll_index()).shape(0);
  feature_len_ = dynamic_cast<DataLayer*>(srclayers[0])->feature_len();  // feature_len_ is 598 = 4 + 592 + 2
  data_.Reshape(batchsize_, feature_len_ - 2);  // reshape data for each unit
}

void UnrollLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  float* ptr = data_.mutable_cpu_data();
  memset(ptr, 0, sizeof(float) * data_.count());
  const float* idx = srclayers[0]->data(unroll_index()).cpu_data();
  for (int b = 0; b < batchsize_; b++) { // 4 demographical features
      ptr[b * (feature_len_ - 2) + 0] = static_cast<float>( idx[b * feature_len_ + 0] );  // age
      //ptr[b * (feature_len_ - 2) + 1] = static_cast<float>( idx[b * feature_len_ + 1] );  // edu
      ptr[b * (feature_len_ - 2) + 1] = static_cast<float>( idx[b * feature_len_ + 1] );  // gen
      ptr[b * (feature_len_ - 2) + 2] = static_cast<float>( idx[b * feature_len_ + 2] );  // lap_time
//      LOG(ERROR) << "data_ for UnrollLayer: age " <<ptr[b * (feature_len_ - 2) + 0]  ;
//      LOG(ERROR) << "data_ for UnrollLayer: edu " <<ptr[b * (feature_len_ - 2) + 1]  ;
//      LOG(ERROR) << "data_ for UnrollLayer: gen " <<ptr[b * (feature_len_ - 2) + 2]  ;
//      LOG(ERROR) << "data_ for UnrollLayer: lt " <<ptr[b * (feature_len_ - 2) + 3]  ;
      for (int i=3; i<feature_len_-2; i++) { // (feature_len_ - 6) = 592 features
          ptr[b * (feature_len_ - 2) + i] = static_cast<float>( idx[b * feature_len_ + i] );  // feature_value
//          LOG(ERROR) << "data_ for UnrollLayer: " <<ptr[b * (feature_len_ - 2) + i] ;
      }
  }
}

/*******UnrolV2LLayer for Model 2**************/
void UnrollV2Layer::Setup(const LayerProto& conf,
  const vector<Layer*>& srclayers) {
  InputLayer::Setup(conf, srclayers);
  batchsize_ = srclayers.at(0)->data(unroll_index()).shape(0);
  feature_len_ = dynamic_cast<DataLayer*>(srclayers[0])->feature_len();  // feature_len_ is 598 = 4 (3 demo + lap_time) + 592 + 2
  data_.Reshape(batchsize_, feature_len_ - 3);  // reshape data for each unit, do NOT include lap_time info in data_
  laptime_info_.Reshape(batchsize_, 1); // for 1 patient in 1 Unroll part/GRU part, only 1 dimension of feature
}

void UnrollV2Layer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "This is UnrollV2Layer's ComputeFeature";
  // fill information in data_
  float* ptr = data_.mutable_cpu_data();
  memset(ptr, 0, sizeof(float) * data_.count());
  // fill information in laptime_info_
  float* ptr_time = laptime_info_.mutable_cpu_data();
  memset(ptr_time, 0, sizeof(float) * laptime_info_.count());
  // data info from src: DataLayer
  const float* idx = srclayers[0]->data(unroll_index()).cpu_data();
  for (int b = 0; b < batchsize_; b++) { // 3 demographical features + lap_time information
      ptr[b * (feature_len_ - 3) + 0] = static_cast<float>( idx[b * feature_len_ + 0] );  // age
      //ptr[b * (feature_len_ - 3) + 2] = static_cast<float>( idx[b * feature_len_ + 1] );  // edu
      ptr[b * (feature_len_ - 3) + 1] = static_cast<float>( idx[b * feature_len_ + 1] );  // gen
      //ptr[b * (feature_len_ - 2) + 3] = static_cast<float>( idx[b * feature_len_ + 2] );  // lap_time not included in data_

      // fill information into laptime_info_
      ptr_time[b * 1 + 0] = static_cast<float>( idx[b * feature_len_ + 2] );  // lap_time
//      LOG(ERROR) << "data_ for UnrollLayer: age " <<ptr[b * (feature_len_ - 2) + 0]  ;
//      LOG(ERROR) << "data_ for UnrollLayer: edu " <<ptr[b * (feature_len_ - 2) + 1]  ;
//      LOG(ERROR) << "data_ for UnrollLayer: gen " <<ptr[b * (feature_len_ - 2) + 2]  ;
//      LOG(ERROR) << "data_ for UnrollLayer: lt " <<ptr[b * (feature_len_ - 2) + 3]  ;
      for (int i=3; i<feature_len_-2; i++) { // (feature_len_ - 6) = 592 features
          ptr[b * (feature_len_ - 3) + (i - 1)] = static_cast<float>( idx[b * feature_len_ + i] );  // feature_value
//          LOG(ERROR) << "data_ for UnrollLayer: " <<ptr[b * (feature_len_ - 2) + i] ;
      }
  }
}

/*******UnrolV3LLayer for Model 3**************/
void UnrollV3Layer::Setup(const LayerProto& conf,
  const vector<Layer*>& srclayers) {
  InputLayer::Setup(conf, srclayers);
  batchsize_ = srclayers.at(0)->data(unroll_index()).shape(0);
  feature_len_ = dynamic_cast<DataLayer*>(srclayers[0])->feature_len();  // feature_len_ is 598 = 4 (3 demo + lap_time) + 592 + 2
  data_.Reshape(batchsize_, feature_len_ - 5);  // reshape data for each unit, do NOT include lap_time, 3 demo features in data_
  laptime_info_.Reshape(batchsize_, 1); // for 1 patient in 1 Unroll part/GRU part, only 1 dimension of feature
}

void UnrollV3Layer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "This is UnrollV3Layer's ComputeFeature";
  // fill information in data_
  float* ptr = data_.mutable_cpu_data();
  memset(ptr, 0, sizeof(float) * data_.count());
  // fill information in laptime_info_
  float* ptr_time = laptime_info_.mutable_cpu_data();
  memset(ptr_time, 0, sizeof(float) * laptime_info_.count());
  // data info from src: DataLayer
  const float* idx = srclayers[0]->data(unroll_index()).cpu_data();
  for (int b = 0; b < batchsize_; b++) { // 3 demographical features + lap_time information: NOT included in data_
      //ptr[b * (feature_len_ - 3) + 0] = static_cast<float>( idx[b * feature_len_ + 0] );  // age
      //ptr[b * (feature_len_ - 3) + 1] = static_cast<float>( idx[b * feature_len_ + 1] );  // edu
      //ptr[b * (feature_len_ - 3) + 2] = static_cast<float>( idx[b * feature_len_ + 2] );  // gen

      // fill information into laptime_info_
      ptr_time[b * 1 + 0] = static_cast<float>( idx[b * feature_len_ + 2] );  // lap_time
//      LOG(ERROR) << "data_ for UnrollLayer: age " <<ptr[b * (feature_len_ - 2) + 0]  ;
//      LOG(ERROR) << "data_ for UnrollLayer: edu " <<ptr[b * (feature_len_ - 2) + 1]  ;
//      LOG(ERROR) << "data_ for UnrollLayer: gen " <<ptr[b * (feature_len_ - 2) + 2]  ;
//      LOG(ERROR) << "data_ for UnrollLayer: lt " <<ptr[b * (feature_len_ - 2) + 3]  ;
      for (int i=3; i<feature_len_-2; i++) { // (feature_len_ - 6) = 592 features
          ptr[b * (feature_len_ - 5) + (i - 3)] = static_cast<float>( idx[b * feature_len_ + i] );  // feature_value
//          LOG(ERROR) << "data_ for UnrollLayer: " <<ptr[b * (feature_len_ - 2) + i] ;
      }
  }
}

/*******DPMGruLayer - For Model 2**************/
DPMGruLayer::~DPMGruLayer() {
  delete weight_z_hx_; // params for update gate
  delete weight_z_hh_;
  delete bias_z_;

  delete weight_r_hx_; // params for reset gate
  delete weight_r_hh_;
  delete bias_r_;

  delete weight_c_hx_; // params for new memory
  delete weight_c_hh_;
  delete bias_c_;

  delete update_gate_; // gate information
  delete reset_gate_;
  delete new_memory_;

  // Parameters for time-related part
  delete weight_theta_;
  delete bias_theta_;

  delete new_update_gate_; // specific for time-related information
  delete time_part_;
  // delete reset_context_;
}

void DPMGruLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_LE(srclayers.size(), 2); // number of src layers no larger than 2
  const auto& src = srclayers[0]->data(this); // input from UnrollLayer

  batchsize_ = src.shape()[0];  // size of batch
  vdim_ = src.count() / (batchsize_);  // dimension of input

  hdim_ = layer_conf_.GetExtension(dpmgru_conf).dim_hidden();  // dimension of hidden state

  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  // one for grad from dst GRU, one for grad from upper layer
  gradvec_.push_back(new Blob<float>(grad_.shape())); //TODO(kaiping) why not check whether push 1 or 2 grads?

  // Initialize the parameters
  weight_z_hx_ = Param::Create(conf.param(0));
  weight_r_hx_ = Param::Create(conf.param(1));
  weight_c_hx_ = Param::Create(conf.param(2));

  weight_z_hh_ = Param::Create(conf.param(3));
  weight_r_hh_ = Param::Create(conf.param(4));
  weight_c_hh_ = Param::Create(conf.param(5));

  // Initialize the parameters specific for processing time information
  weight_theta_ = Param::Create(conf.param(6));

  if (conf.param_size() > 7) {
    bias_z_ = Param::Create(conf.param(7));
    bias_r_ = Param::Create(conf.param(8));
    bias_c_ = Param::Create(conf.param(9));
    // Initialize the parameters specific for processing time information
    bias_theta_ = Param::Create(conf.param(10));
  }

  weight_z_hx_->Setup(vector<int>{hdim_, vdim_});
  weight_r_hx_->Setup(vector<int>{hdim_, vdim_});
  weight_c_hx_->Setup(vector<int>{hdim_, vdim_});

  weight_z_hh_->Setup(vector<int>{hdim_, hdim_});
  weight_r_hh_->Setup(vector<int>{hdim_, hdim_});
  weight_c_hh_->Setup(vector<int>{hdim_, hdim_});
  // set up weight_theta_
  weight_theta_->Setup(vector<int>{hdim_, 1});

  if (conf.param_size() > 7) {
    bias_z_->Setup(vector<int>{hdim_});
    bias_r_->Setup(vector<int>{hdim_});
    bias_c_->Setup(vector<int>{hdim_});
    // set up bias_theta_
    bias_theta_->Setup(vector<int>{hdim_});
  }

  update_gate_ = new Blob<float>(batchsize_, hdim_);
  reset_gate_ = new Blob<float>(batchsize_, hdim_);
  new_memory_ = new Blob<float>(batchsize_, hdim_);
  // set up new gate information for processing time
  time_part_ = new Blob<float>(batchsize_, hdim_);
  new_update_gate_ = new Blob<float>(batchsize_, hdim_);
}

void DPMGruLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "This is DPMGruLayer's ComputeFeature";
  //LOG(ERROR) << "Unroll index of DPMGruLayer" << unroll_index();
  CHECK_LE(srclayers.size(), 2);

  // Do transpose
  Blob<float> *w_z_hx_t = Transpose(weight_z_hx_->data());
  Blob<float> *w_z_hh_t = Transpose(weight_z_hh_->data());
  Blob<float> *w_r_hx_t = Transpose(weight_r_hx_->data());
  Blob<float> *w_r_hh_t = Transpose(weight_r_hh_->data());
  Blob<float> *w_c_hx_t = Transpose(weight_c_hx_->data());
  Blob<float> *w_c_hh_t = Transpose(weight_c_hh_->data());
  // new parameters
  Blob<float> *w_theta_t = Transpose(weight_theta_->data());

  // Prepare the data input and the context
  const auto& src = srclayers[0]->data(this);
  const Blob<float> *context;
  if (srclayers.size() == 1) {  // only have data input
    context = new Blob<float>(batchsize_, hdim_); // ? (kaiping): When is this initialized (to all 0)?
  } else {  // have data input & context
    context = &srclayers[1]->data(this);
  }
  // Prepare the lap_time input
  Blob<float> lap;
  //LOG(ERROR) << "TEST" << srclayers[0]->user_type();
  if (srclayers[0]->user_type() == "kUnrollV3") { // DPMGruLayer needs to handle Model 2 & Model 3
      lap = dynamic_cast<UnrollV3Layer *>(srclayers[0])->laptime_info(); // information related to lap time
      // for testing - the output is backward order as it is in ComputeGradient
//      LOG(ERROR) << "Forward: ";
//      auto lap_ptr = lap.mutable_cpu_data();
//      for(int i = 0; i < lap.shape(0); i++) {
//          for (int j = 0; j < lap.shape(1); j++) {
//              LOG(ERROR) << "Lap time info: " << *(lap_ptr + i * lap.shape(1) + j);
//          }
//      }
  }
  else {
      lap = dynamic_cast<UnrollV2Layer *>(srclayers[0])->laptime_info(); // information related to lap time
  }
  //LOG(ERROR) << "Shape size for laptime_info_: " << lap.shape().size(); // for testing
  //LOG(ERROR) << "Shape 1 for laptime_info_: " << lap.shape(0);
  //LOG(ERROR) << "Shape 2 for laptime_info_: " << lap.shape(1);

  // Compute the update gate, new update gate and time part information
  // Original update gate
  singa::GEMM(1.0f, 0.0f, src, *w_z_hx_t, update_gate_);
  if (bias_z_ != nullptr)
      singa::MVAddRow(1.0f, 1.0f, bias_z_->data(), update_gate_);
  GEMM(1.0f, 1.0f, *context, *w_z_hh_t, update_gate_);
  singa::Map<singa::op::Sigmoid<float>, float>(*update_gate_, update_gate_);
  // LOG(ERROR) << "Update Gate: " << update_gate_->cpu_data()[0];

  // Time part computation & new update gate computation
  Blob<float> one_minus_timepart(batchsize_, hdim_);
  one_minus_timepart.SetValue(1.0f); // SetValue() is to fill every entry of Blob with the same value

  singa::GEMM(1.0f, 0.0f, lap, *w_theta_t, time_part_);
  if (bias_theta_ != nullptr)
      singa::MVAddRow(1.0f, 1.0f, bias_theta_->data(), time_part_);
  singa::Map<singa::op::Sigmoid<float>, float>(*time_part_, time_part_);
  singa::AXPY<float>(-1.0f, *time_part_, &one_minus_timepart);
  singa::Mult(*update_gate_, one_minus_timepart, new_update_gate_);

  // Compute the reset gate
  singa::GEMM(1.0f, 0.0f, src, *w_r_hx_t, reset_gate_);
  if (bias_r_ != nullptr)
      singa::MVAddRow(1.0f, 1.0f, bias_r_->data(), reset_gate_);
  singa::GEMM(1.0f, 1.0f, *context, *w_r_hh_t, reset_gate_);
  singa::Map<singa::op::Sigmoid<float>, float>(*reset_gate_, reset_gate_);
  // LOG(ERROR) << "Reset Gate: " << reset_gate_->cpu_data()[0];
  // Compute the new memory
  singa::GEMM(1.0f, 0.0f, *context, *w_c_hh_t, new_memory_);
  singa::Mult<float>(*reset_gate_, *new_memory_, new_memory_);
  singa::GEMM(1.0f, 1.0f, src, *w_c_hx_t, new_memory_);
  if (bias_c_ != nullptr)
      singa::MVAddRow(1.0f, 1.0f, bias_c_->data(), new_memory_);
  singa::Map<singa::op::Tanh<float>, float>(*new_memory_, new_memory_);

  singa::Sub(*context, *new_memory_, &data_);
  singa::Mult(data_, *new_update_gate_, &data_);
  singa::Add(data_, *new_memory_, &data_);

  // delete the pointers
  if (srclayers.size() == 1)
    delete context; // actually not used

  delete w_z_hx_t;
  delete w_z_hh_t;
  delete w_r_hx_t;
  delete w_r_hh_t;
  delete w_c_hx_t;
  delete w_c_hh_t;
  // new params
  delete w_theta_t;
  //delete one;
  //delete lap;
}

void DPMGruLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  CHECK_LE(srclayers.size(), 2);
  // agg grad from two dst layers, gradvec_[0] is grad_ (computed as src_grad in the direct upper layer)
  singa::AXPY(1.0f, *gradvec_[1], &grad_);
  float beta = 1.0f;  // agg param gradients

  Layer* ilayer = srclayers[0];  // input layer
  Layer* clayer = nullptr;  // context layer
  // Prepare the data input and the context
  const Blob<float>& src = ilayer->data(this);
  const Blob<float> *context;
  if (srclayers.size() == 1) {  // only have data input
    context = new Blob<float>(batchsize_, hdim_); // ? (kaiping): where to fill in with all 0s?
  } else {  // have data input & context
    clayer = srclayers[1];
    context = &(clayer->data(this));
  }

  // Retrieve time-related information
  //LOG(ERROR) << "TEST" << srclayers[0]->user_type();
  Blob<float> lap;
  if (srclayers[0]->user_type() == "kUnrollV3") { // DPMGruLayer needs to handle Model 2 & Model 3
      lap = dynamic_cast<UnrollV3Layer *>(srclayers[0])->laptime_info(); // information related to lap time
      // for testing - the output is backward order as it is in ComputeGradient
//      LOG(ERROR) << "Backward: ";
//      auto lap_ptr = lap.mutable_cpu_data();
//      for(int i = 0; i < lap.shape(0); i++) {
//          for (int j = 0; j < lap.shape(1); j++) {
//              LOG(ERROR) << "Lap time info: " << *(lap_ptr + i * lap.shape(1) + j);
//          }
//      }
  }
  else {
      lap = dynamic_cast<UnrollV2Layer *>(srclayers[0])->laptime_info(); // information related to lap time
  }

  //LOG(ERROR) << "Shape size for laptime_info_: " << lap.shape().size(); // for testing
  //LOG(ERROR) << "Shape 1 for laptime_info_: " << lap.shape(0);
  //LOG(ERROR) << "Shape 2 for laptime_info_: " << lap.shape(1);

  // Compute intermediate gradients which are used for other computations (gradient of gates to pre-activation sum)
  Blob<float> dugatedz(batchsize_, hdim_);
  singa::Map<singa::op::SigmoidGrad<float>, float>(*update_gate_, &dugatedz); // dugateddz should not be changed, used later
  Blob<float> drgatedr(batchsize_, hdim_);
  singa::Map<singa::op::SigmoidGrad<float>, float>(*reset_gate_, &drgatedr);
  Blob<float> dnewmdc(batchsize_, hdim_);
  singa::Map<singa::op::TanhGrad<float>, float>(*new_memory_, &dnewmdc);

  // Add intermediate gradients for newly-considered time part
  Blob<float> dtimedt(batchsize_, hdim_);
  singa::Map<singa::op::SigmoidGrad<float>, float>(*time_part_, &dtimedt);

  // kaiping: Compute gradient of Loss to pre-activated sum of Update Gate: dLdz
  Blob<float> dLdz(batchsize_, hdim_);
  Blob<float> one_minus_timepart2(batchsize_, hdim_);
  one_minus_timepart2.SetValue(1.0f); // SetValue() is to fill every entry of Blob with the same value
  singa::AXPY<float>(-1.0f, *time_part_, &one_minus_timepart2);
  singa::Sub<float>(*context, *new_memory_, &dLdz);
  singa::Mult<float>(dLdz, grad_, &dLdz);
  singa::Mult<float>(dLdz, dugatedz, &dLdz);
  singa::Mult<float>(dLdz, one_minus_timepart2, &dLdz); // Model 2: Add time information's influence in update gate computation

  // Model 2: Compute gradient of Loss to pre-activated sum of Time part (gate): dLdt
  Blob<float> dLdt(batchsize_, hdim_);
  Blob<float> zmin1(batchsize_, hdim_);
  zmin1.SetValue(-1.0f);
  singa::Sub<float>(*context, *new_memory_, &dLdt);
  singa::Mult<float>(dLdt, grad_, &dLdt);
  singa::Mult<float>(dLdt, *update_gate_, &dLdt);
  singa::Mult<float>(dLdt, zmin1, &dLdt);
  singa::Mult<float>(dLdt, dtimedt, &dLdt);

  // kaiping: Compute gradient of Loss to pre-activated sum of Context
  Blob<float> dLdc(batchsize_, hdim_);
  Blob<float> z1(batchsize_, hdim_);
  z1.SetValue(1.0f);
  singa::AXPY<float>(-1.0f, *new_update_gate_, &z1); // Add time information's influence in new mem computation (use new_update_gate_)
  singa::Mult(grad_, z1, &dLdc);
  singa::Mult(dLdc, dnewmdc, &dLdc);

  // kaiping: Compute the product of dLdc and reset_gate
  Blob<float> reset_dLdc(batchsize_, hdim_);
  singa::Mult(dLdc, *reset_gate_, &reset_dLdc); // As dLdc already changes, no explicit change here

  // kaiping: Compute gradient of Loss to pre-activated sum of Reset Gate
  Blob<float> dLdr(batchsize_, hdim_);
  Blob<float> cprev(batchsize_, hdim_);
  singa::GEMM(1.0f, 0.0f, *context, weight_c_hh_->data().T(), &cprev);
  singa::Mult(dLdc, cprev, &dLdr);
  singa::Mult(dLdr, drgatedr, &dLdr); // Model 2: As dLdc already changes, no explicit change here

  // Compute gradients for parameters of update gate (No explicit change here, already change dLdx part)
  Blob<float> *dLdz_t = Transpose(dLdz);
  singa::GEMM(1.0f, beta, *dLdz_t, src, weight_z_hx_->mutable_grad());
  singa::GEMM(1.0f, beta, *dLdz_t, *context, weight_z_hh_->mutable_grad());
  if (bias_z_ != nullptr)
      singa::MVSumRow<float>(1.0f, beta, dLdz, bias_z_->mutable_grad());
  delete dLdz_t;

  // Compute gradients for parameters of reset gate (No explicit change here, already change dLdx part)
  Blob<float> *dLdr_t = Transpose(dLdr);
  singa::GEMM(1.0f, beta, *dLdr_t, src, weight_r_hx_->mutable_grad());
  singa::GEMM(1.0f, beta, *dLdr_t, *context, weight_r_hh_->mutable_grad());
  if (bias_r_ != nullptr)
      singa::MVSumRow(1.0f, beta, dLdr, bias_r_->mutable_grad());
  delete dLdr_t;

  // Compute gradients for parameters of new memory (No explicit change here, already change dLdx part)
  Blob<float> *dLdc_t = Transpose(dLdc);
  singa::GEMM(1.0f, beta, *dLdc_t, src, weight_c_hx_->mutable_grad());
  if (bias_c_ != nullptr)
      singa::MVSumRow(1.0f, beta, dLdc, bias_c_->mutable_grad());
  delete dLdc_t;

  Blob<float> *reset_dLdc_t = Transpose(reset_dLdc);
  singa::GEMM(1.0f, beta, *reset_dLdc_t, *context, weight_c_hh_->mutable_grad());
  delete reset_dLdc_t;

  // Compute gradients for newly-added time-related parameters: weight_theta_, bias_theta_
  Blob<float> *dLdt_t = Transpose(dLdt);
  singa::GEMM(1.0f, beta, *dLdt_t, lap, weight_theta_->mutable_grad());
  if (bias_theta_ != nullptr)
      singa::MVSumRow<float>(1.0f, beta, dLdt, bias_theta_->mutable_grad());
  delete dLdt_t;

  // Compute gradients for data input layer (Model 2: no explicit change here)
  if (srclayers[0]->mutable_grad(this) != nullptr) {
      singa::GEMM(1.0f, 0.0f, dLdc, weight_c_hx_->data(), ilayer->mutable_grad(this));
      singa::GEMM(1.0f, 1.0f, dLdz, weight_z_hx_->data(), ilayer->mutable_grad(this));
      singa::GEMM(1.0f, 1.0f, dLdr, weight_r_hx_->data(), ilayer->mutable_grad(this));
  }

  // Compute gradients for context input layer (No explicit change here, already change dLdx part)
  if (clayer != nullptr && clayer->mutable_grad(this) != nullptr) {
    // Compute gradients for context layer
      singa::GEMM(1.0f, 0.0f, reset_dLdc, weight_c_hh_->data(),
        clayer->mutable_grad(this));
      singa::GEMM(1.0f, 1.0f, dLdr, weight_r_hh_->data(), clayer->mutable_grad(this));
      singa::GEMM(1.0f, 1.0f, dLdz, weight_z_hh_->data(), clayer->mutable_grad(this));
      singa::Add(clayer->grad(this), *update_gate_, clayer->mutable_grad(this));
    // LOG(ERROR) << "grad to prev gru " << Asum(clayer->grad(this));
  }

  if (srclayers.size() == 1) // kaiping: the first GRU unit
    delete context;
}

/*******DPMDemoLayer**************/
void DPMDemoLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  InputLayer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1); // DPMLabelLayer has only 1 src layer
  batchsize_ = srclayers.at(0)->data(0).shape(0);
  feature_len_ = dynamic_cast<DataLayer*>(srclayers[0])->feature_len();
  unroll_len_ = dynamic_cast<DataLayer*>(srclayers[0])->unroll_len();
  data_.Reshape(batchsize_, 3);
}

void DPMDemoLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  float* ptr = data_.mutable_cpu_data();
  // all unroll units have demo features, here look at the last unroll unit only
  const float* idx = srclayers[0]->data(unroll_len_-1).cpu_data();
  for (int b = 0; b < batchsize_; b++) {
    ptr[b * 3 + 0] = static_cast<float>( idx[b * feature_len_ + 0] );  // age
    ptr[b * 3 + 1] = static_cast<float>( idx[b * feature_len_ + 1] );  // edu
    ptr[b * 3 + 2] = static_cast<float>( idx[b * feature_len_ + 2] );  // gen
  }
}

/*******DPMLabelLayer**************/
DPMLabelLayer::~DPMLabelLayer() {
  fin_train_data.close();
  fin_train_label.close();
  fin_train_time.close();
  fin_test_data.close();
  fin_test_label.close();
  fin_test_time.close();
}

void DPMLabelLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  InputLayer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1); // DPMLabelLayer has only 1 src layer
  batchsize_ = srclayers.at(0)->data(0).shape(0);
  feature_len_ = dynamic_cast<DataLayer*>(srclayers[0])->feature_len();
  unroll_len_ = dynamic_cast<DataLayer*>(srclayers[0])->unroll_len();
  data_.Reshape(batchsize_, 1);
  fin_train_data.open("train_data");
  fin_train_label.open("train_label");
  fin_train_time.open("train_time");
  fin_test_data.open("test_data");
  fin_test_label.open("test_label");
  fin_test_time.open("test_time");
}

void DPMLabelLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  float* ptr = data_.mutable_cpu_data();
  // look at the last unroll unit only
  const float* idx = srclayers[0]->data(unroll_len_-1).cpu_data();
  for (int b = 0; b < batchsize_; b++) {
    ptr[b] = static_cast<int>(idx[b * feature_len_ + feature_len_ - 1]);  // mmscore
//    LOG(ERROR) << "data_ for DPMLabelLayer: " << ptr[b];
  }
  // printing for testing phase
  if(flag & 4 != 0) {
      // printing label information
      const float* test_label = srclayers[0]->data(unroll_len_-1).cpu_data();
      for (int test_label_bs = 0; test_label_bs < batchsize_; test_label_bs++) {
          fin_test_time << static_cast<int>(test_label[test_label_bs * feature_len_ + feature_len_ - 2]) << std::endl;
          fin_test_label << static_cast<int>(test_label[test_label_bs * feature_len_ + feature_len_ - 1]) << std::endl;
      }
      // printing data information
      for (int test_data_bs = 0; test_data_bs < batchsize_; test_data_bs++) {
          string dat = ""; // 1 sample/patient corresponds to 1 line in the output file
          for (int test_data_unroll = 0; test_data_unroll < unroll_len_; test_data_unroll++) {
              const float* test_data = srclayers[0]->data(test_data_unroll).cpu_data();
              for (int test_feature = 0; test_feature < feature_len_ - 2; test_feature++) {
                  dat += static_cast<float>(test_data[test_data_bs * feature_len_ + test_feature]);
                  if ((test_data_unroll == unroll_len_ - 1) && (test_feature == feature_len_ - 3)) { // the last feature
                      dat += "\n";
                  }
                  else {
                      dat += ",";
                  }
              }
          }
          fin_test_data << dat;
      }
  }
  // printing for training phase
  if(flag & 1 != 0) {
      // printing label information
      const float* train_label = srclayers[0]->data(unroll_len_-1).cpu_data();
      for (int train_label_bs = 0; train_label_bs < batchsize_; train_label_bs++) {
          fin_train_time << static_cast<int>(train_label[train_label_bs * feature_len_ + feature_len_ - 2]) << std::endl;
          fin_train_label << static_cast<int>(train_label[train_label_bs * feature_len_ + feature_len_ - 1]) << std::endl;
      }
      // printing data information
      for (int train_data_bs = 0; train_data_bs < batchsize_; train_data_bs++) {
          string dat2 = ""; // 1 sample/patient corresponds to 1 line in the output file
          for (int train_data_unroll = 0; train_data_unroll < unroll_len_; train_data_unroll++) {
              const float* train_data = srclayers[0]->data(train_data_unroll).cpu_data();
              for (int train_feature = 0; train_feature < feature_len_ - 2; train_feature++) {
                  dat2 += static_cast<float>(train_data[train_data_bs * feature_len_ + train_feature]);
                  if ((train_data_unroll == unroll_len_ - 1) && (train_feature == feature_len_ - 3)) { // the last feature
                      dat2 += "\n";
                  }
                  else {
                      dat2 += ",";
                  }
              }
          }
          fin_train_data << dat2;
      }
  }
}

/*******DPMTimeLayer**************/
void DPMTimeLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  InputLayer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1); // DPMLabelLayer has only 1 src layer
  batchsize_ = srclayers.at(0)->data(0).shape(0);
  feature_len_ = dynamic_cast<DataLayer*>(srclayers[0])->feature_len();
  unroll_len_ = dynamic_cast<DataLayer*>(srclayers[0])->unroll_len();
  data_.Reshape(batchsize_, 1);
}

void DPMTimeLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  float* ptr = data_.mutable_cpu_data();
  // look at the last unroll unit only
  const float* idx = srclayers[0]->data(unroll_len_-1).cpu_data();
  for (int b = 0; b < batchsize_; b++) {
    ptr[b] = static_cast<int>(idx[b * feature_len_ + feature_len_ - 2]);  // delta_time
//    LOG(ERROR) << "data_ for DPMTimeLayer: " << ptr[b];
  }
}

/*******CombinationLayer**************/
CombinationLayer::~CombinationLayer() {
  delete weight1_;
  delete weight2_;
  delete bias_;
}

void CombinationLayer::Setup(const LayerProto &conf,
                             const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 2); // CombinationLayer has 2 src layers
  const auto& src_gru = srclayers[0]->data(this); // 1st src layer is gru layer
  const auto& src_timespan = srclayers[1]->data(this); // 2nd layer is time_span layer
  batchsize_ = src_gru.shape()[0];
  vdim_ = src_gru.count() / batchsize_;
  hdim_ = conf.GetExtension(combination_conf).num_output();
  transpose_ = conf.GetExtension(combination_conf).transpose();
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
  else {// our case
    MMDot(srclayers[0]->data(this), weight1_->data().T(), &data_); // First part of data_
    MMDot(srclayers[1]->data(this), weight2_->data().T(), tmp); // Second part of data_
  }
  AXPY(1.0f, *tmp, &data_); // Combine 2 parts
  MVAddRow(bias_->data(), &data_); // Add bias
  delete tmp;
//  const float* printout = weight2_->data().cpu_data();
//  LOG(ERROR) << "Weight info: ";
//  LOG(ERROR) << "Weight info size: " << weight2_->data().shape().size();
//  LOG(ERROR) << "Weight info shape(0): " << weight2_->data().shape(0);
//  LOG(ERROR) << "Weight info shape(1): " << weight2_->data().shape(1);
//  for(int i = 0; i < weight2_->data().shape(0); i++) {
//      for(int j = 0; j < weight2_->data().shape(1); j++) {
//          LOG(ERROR) << "Weight info @ ( " << i << " , " << j << " ) is:   " << (*(printout + i * weight2_->data().shape(1) + j));
//      }
//  }
//  LOG(ERROR) << "Bias info: ";
//  LOG(ERROR) << "Bias info size: " << bias_->data().shape().size();
//  LOG(ERROR) << "Bias info shape(0): " << bias_->data().shape(0);
//  const float* printbias = bias_->data().cpu_data();
//  for(int k = 0; k < bias_->data().shape(0); k++) {
//      LOG(ERROR) << "Bias info @ " << k << " is:   " << (*(printbias + k));
//  }
  //LOG(ERROR) << "Weight info: ";
  //LOG(ERROR) << "Weight information for delta_T: " << weight2_->data();
  //LOG(ERROR) << "Bias information for delta_T: " << bias_->data();
}

void CombinationLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  float beta = 0.0f;
  //if(flag & singa::kAggGrad)
  //  beta = 1.0f;
  //LOG(ERROR) << "Beta info: " << beta; // the output beta is "1", after change, output is "1"
  // Compute gradients for parameters
  MVSumRow(1.0f, beta, grad_, bias_->mutable_grad());
  if(transpose_) {
    GEMM(1.0f, beta, srclayers[0]->data(this).T(), grad_,
         weight1_->mutable_grad());
    GEMM(1.0f, beta, srclayers[1]->data(this).T(), grad_,
         weight2_->mutable_grad());
  }
  else { // our case
    GEMM(1.0f, beta, grad_.T(), srclayers[0]->data(this),
         weight1_->mutable_grad());
    GEMM(1.0f, beta, grad_.T(), srclayers[1]->data(this),
         weight2_->mutable_grad());
  }

  if (srclayers[0]->mutable_grad(this) != nullptr) { // Compute the gradient for src_layer: gru layer; Note that gradient for src_layer: Delta_T is DataLayer, should be no grad_
    if (transpose_)
      MMDot(grad_, weight1_->data().T(), srclayers[0]->mutable_grad(this));
    else
      MMDot(grad_, weight1_->data(), srclayers[0]->mutable_grad(this));
  }
}

/*******CombinationV3Layer**************/
CombinationV3Layer::~CombinationV3Layer() {
  delete weight1_;
  delete weight2_;
  delete weight3_;
  delete bias_;
}

void CombinationV3Layer::Setup(const LayerProto &conf,
                             const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 3); // CombinationV3Layer has 3 src layers
  const auto& src_gru = srclayers[0]->data(this); // 1st src layer is gru layer
  const auto& src_timespan = srclayers[1]->data(this); // 2nd layer is time_span layer
  const auto& src_demo = srclayers[2]->data(this); // 3rd layer is demo layer
  batchsize_ = src_gru.shape()[0];
  vdim_ = src_gru.count() / batchsize_;
  vdim_demo_ = src_demo.count() / batchsize_;
  hdim_ = conf.GetExtension(combinationv3_conf).num_output();
  transpose_ = conf.GetExtension(combinationv3_conf).transpose();
  //if (partition_dim() > 0)
  //  hdim_ /= srclayers.at(0)->num_partitions();
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  weight1_ = Param::Create(conf.param(0));
  weight2_ = Param::Create(conf.param(1));
  weight3_ = Param::Create(conf.param(2));
  bias_ = Param::Create(conf.param(3));
  if(transpose_) {
    weight1_->Setup(vector<int>{vdim_,hdim_});
    weight2_->Setup(vector<int>{1,hdim_});
    weight3_->Setup(vector<int>{vdim_demo_,hdim_});
  }
  else { // this is our case
    weight1_->Setup(vector<int>{hdim_,vdim_});
    weight2_->Setup(vector<int>{hdim_,1});
    weight3_->Setup(vector<int>{hdim_,vdim_demo_});
  }
  bias_->Setup(vector<int>{hdim_});
}

void CombinationV3Layer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "Combination layer";
  Blob<float>* tmp_time = new Blob<float>(batchsize_, hdim_); // Use "tmp_time" to store the computation result from TimeSpanUnit
  Blob<float>* tmp_demo = new Blob<float>(batchsize_, hdim_); // Use "tmp_demo" to store the computation result from DemoUnit
  if(transpose_) {
    MMDot(srclayers[0]->data(this), weight1_->data(), &data_); // First part of data_
    MMDot(srclayers[1]->data(this), weight2_->data(), tmp_time); // Second part of data_
    MMDot(srclayers[2]->data(this), weight3_->data(), tmp_demo); // Third part of data_
  }
  else {// our case
    MMDot(srclayers[0]->data(this), weight1_->data().T(), &data_); // First part of data_
    MMDot(srclayers[1]->data(this), weight2_->data().T(), tmp_time); // Second part of data_
    MMDot(srclayers[2]->data(this), weight3_->data().T(), tmp_demo); // Third part of data_
  }
  AXPY(1.0f, *tmp_time, &data_); // Add second part
  AXPY(1.0f, *tmp_demo, &data_); // Add third part
  MVAddRow(bias_->data(), &data_); // Add bias
  delete tmp_time;
  delete tmp_demo;
//  const float* printout = weight2_->data().cpu_data();
//  LOG(ERROR) << "Weight info: ";
//  LOG(ERROR) << "Weight info size: " << weight2_->data().shape().size();
//  LOG(ERROR) << "Weight info shape(0): " << weight2_->data().shape(0);
//  LOG(ERROR) << "Weight info shape(1): " << weight2_->data().shape(1);
//  for(int i = 0; i < weight2_->data().shape(0); i++) {
//      for(int j = 0; j < weight2_->data().shape(1); j++) {
//          LOG(ERROR) << "Weight info @ ( " << i << " , " << j << " ) is:   " << (*(printout + i * weight2_->data().shape(1) + j));
//      }
//  }
//  LOG(ERROR) << "Bias info: ";
//  LOG(ERROR) << "Bias info size: " << bias_->data().shape().size();
//  LOG(ERROR) << "Bias info shape(0): " << bias_->data().shape(0);
//  const float* printbias = bias_->data().cpu_data();
//  for(int k = 0; k < bias_->data().shape(0); k++) {
//      LOG(ERROR) << "Bias info @ " << k << " is:   " << (*(printbias + k));
//  }
  //LOG(ERROR) << "Weight info: ";
  //LOG(ERROR) << "Weight information for delta_T: " << weight2_->data();
  //LOG(ERROR) << "Bias information for delta_T: " << bias_->data();
}

void CombinationV3Layer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  float beta = 0.0f;
  //if(flag & singa::kAggGrad)
  //  beta = 1.0f;
  //LOG(ERROR) << "Beta info: " << beta; // the output beta is "1", after change, output is "1"
  // Compute gradients for parameters
  MVSumRow(1.0f, beta, grad_, bias_->mutable_grad());
  if(transpose_) {
    GEMM(1.0f, beta, srclayers[0]->data(this).T(), grad_,
         weight1_->mutable_grad());
    GEMM(1.0f, beta, srclayers[1]->data(this).T(), grad_,
         weight2_->mutable_grad());
    GEMM(1.0f, beta, srclayers[2]->data(this).T(), grad_,
         weight3_->mutable_grad());
  }
  else { // our case
    GEMM(1.0f, beta, grad_.T(), srclayers[0]->data(this),
         weight1_->mutable_grad());
    GEMM(1.0f, beta, grad_.T(), srclayers[1]->data(this),
         weight2_->mutable_grad());
    GEMM(1.0f, beta, grad_.T(), srclayers[2]->data(this),
         weight3_->mutable_grad());
  }

  if (srclayers[0]->mutable_grad(this) != nullptr) { // Compute the gradient for src_layer[0]: gru layer; Note that gradient for src_layer[1]: Delta_T is DataLayer, should be no grad_
    if (transpose_)
      MMDot(grad_, weight1_->data().T(), srclayers[0]->mutable_grad(this));
    else
      MMDot(grad_, weight1_->data(), srclayers[0]->mutable_grad(this));
  }

  if (srclayers[2]->mutable_grad(this) != nullptr) { // Compute the gradient for src_layer[2]: demo layer;
    if (transpose_)
      MMDot(grad_, weight3_->data().T(), srclayers[2]->mutable_grad(this));
    else
      MMDot(grad_, weight3_->data(), srclayers[2]->mutable_grad(this));
  }
}


}   // end of namespace dpm
