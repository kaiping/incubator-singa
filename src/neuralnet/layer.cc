#include <glog/logging.h>
#include <memory>
#include <algorithm>
#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"
#include "neuralnet/layer.h"
#include "utils/singleton.h"
#include "../../include/neuralnet/layer.h"

using namespace mshadow;
using namespace mshadow::expr;

namespace singa {
inline Tensor<cpu, 4> Tensor4(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 4> tensor(blob->mutable_cpu_data(),
      Shape4(shape[0], shape[1], shape[2], shape[3]));
  return tensor;
}

inline Tensor<cpu, 3> Tensor3(Blob<float>* blob){
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 3> tensor(blob->mutable_cpu_data(),
      Shape3(shape[0], shape[1], blob->count() / shape[0] / shape[1]));
  return tensor;
}
inline Tensor<cpu, 2> Tensor2(Blob<float>* blob){
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 2> tensor(blob->mutable_cpu_data(),
      Shape2(shape[0], blob->count() / shape[0]));
  return tensor;
}
inline Tensor<cpu, 1> Tensor1(Blob<float>* blob){
  Tensor<cpu, 1> tensor(blob->mutable_cpu_data(), Shape1(blob->count()));
  return tensor;
}

/************ Implementation for ConvProductLayer*************************/
ConvolutionLayer::~ConvolutionLayer() {
  delete weight_;
  delete bias_;
}
void ConvolutionLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  ConvolutionProto conv_conf=proto.convolution_conf();
  kernel_=conv_conf.kernel();
  CHECK_GT(kernel_, 0) << "Filter size cannot be zero.";
  pad_=conv_conf.pad();
  stride_=conv_conf.stride();
  num_filters_=conv_conf.num_filters();
  if(partition_dim() > 0)
    num_filters_ /= npartitions;

  const vector<int>& srcshape=srclayers_[0]->data(this).shape();
  int dim=srcshape.size();
  CHECK_GT(dim, 2);
  width_=srcshape[dim-1];
  height_=srcshape[dim-2];
  if(dim>3)
    channels_=srcshape[dim-3];
  else if(dim>2)
    channels_=1;
  batchsize_=srcshape[0];
  conv_height_=(height_ + 2 * pad_ - kernel_) / stride_ + 1;
  conv_width_= (width_ + 2 * pad_ - kernel_) / stride_ + 1;
  col_height_=channels_*kernel_*kernel_;
  col_width_=conv_height_*conv_width_;
  vector<int> shape{batchsize_, num_filters_, conv_height_, conv_width_};
  data_.Reshape(shape);
  grad_.Reshape(shape);
  col_data_.Reshape(vector<int>{col_height_, col_width_});
  col_grad_.Reshape(vector<int>{col_height_, col_width_});

  weight_ = Param::Create(proto.param(0));
  bias_ = Param::Create(proto.param(1));
  weight_->Setup(proto.param(0), vector<int>{num_filters_, col_height_});
  bias_->Setup(proto.param(1), vector<int>{num_filters_});
}

void ConvolutionLayer::ComputeFeature(Phase phase, Metric* perf){
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor3(&data_);
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());

  for(int n=0;n<batchsize_;n++){
    if(pad_>0)
      col=unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col=unpack_patch2col(src[n], kernel_, stride_);
    data[n]=dot(weight, col);
  }
  data+=broadcast<1>(bias, data.shape);
}

void ConvolutionLayer::ComputeGradient(Phase phase) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());

  auto grad = Tensor3(&grad_);
  auto gcol = Tensor2(&col_grad_);
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());

  Blob<float>* gsrcblob=srclayers_[0]->mutable_grad(this);
  Tensor<cpu, 4> gsrc(nullptr, Shape4(batchsize_, channels_, height_, width_));
  if(gsrcblob!=nullptr)
    gsrc.dptr=gsrcblob->mutable_cpu_data();
  gbias=sumall_except_dim<1>(grad);

  gweight = 0.0f;
  Shape<3> padshp(gsrc.shape.SubShape());
  padshp[0] += 2 * pad_;
  padshp[1] += 2 * pad_;
  Shape<2> imgshp = Shape2(height_, width_);
  for(int n=0;n<batchsize_;n++){
    if(pad_>0)
      col=unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col=unpack_patch2col(src[n], kernel_, stride_);
    gweight += dot(grad[n], col.T());

    if(gsrcblob!=nullptr){
      gcol = dot(weight.T(), grad[n]);
      gsrc[n] = crop(pack_col2patch(gcol, padshp, kernel_, stride_), imgshp);
    }
  }
}

/****************** Implementation for DropoutLayer ***********************/
void DropoutLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(*srclayers_[0]->mutable_grad(this));
  mask_.Reshape(srclayers_[0]->data(this).shape());
  pdrop_ = proto.dropout_conf().dropout_ratio();
}

void DropoutLayer::ComputeFeature(Phase phase, Metric* perf) {
  // check training
  if(phase != kTrain){//!training){
    data_.CopyFrom(srclayers_[0]->data(this));
    return;
  }
  float pkeep=1-pdrop_;
  auto mask = Tensor1(&mask_);
  mask = F<op::threshold>(TSingleton<Random<cpu>>::Instance()\
      ->uniform(mask.shape), pkeep ) * (1.0f/pkeep);
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data = src * mask;
}

void DropoutLayer::ComputeGradient(Phase phase)  {
  auto mask = Tensor1(&mask_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc = grad * mask;
}
/**************** Implementation for RBMVisLayer********************/
RBMVisLayer::~RBMVisLayer() {
  delete weight_;
  delete bias_;
}
void RBMVisLayer::Setup(const LayerProto& proto,
      int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 2);
  // hid_idx_: index indicating which srclayer is is hidden layer
  // data_idx_: index indicating which srclayer is data layer
  for (unsigned int i = 0; i < srclayers_.size(); i++)
    for (unsigned int j = 0; j < (srclayers_[i]-> dstlayers()).size(); j++)
      if (strcmp(((srclayers_[i]->dstlayers()).at(j)->name().c_str()),
        (this->name()).c_str()) == 0)
        hid_idx_ = i;
  for (unsigned int i = 0; i < srclayers_.size(); i++)
    if (i != static_cast<unsigned int>(hid_idx_) )
      data_idx_ = i;
  const auto& src = srclayers_[data_idx_]->data(this);
  is_first_iteration_vis_ = true;
  batchsize_ = src.shape()[0];
  neg_batchsize_ = batchsize_;
  /*gibbs sampling size and input have the same size*/
  vdim_ = src.count()/batchsize_;
  hdim_ = proto.rbmvis_conf().num_output();
  data_.Reshape(vector<int>{batchsize_, vdim_});  // this is visible dimension
  vis_sample_.Reshape(vector<int>{neg_batchsize_, vdim_});
  weight_ = Param::Create(proto.param(0));
  bias_ = Param::Create(proto.param(1));
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
  bias_->Setup(proto.param(1), vector<int>{vdim_});
}

void RBMVisLayer::ComputeFeature(Phase phase, Metric* perf) {
  if (phase == kPositive) { /*positive phase*/
    auto data = Tensor2(&data_);
    CHECK_EQ(srclayers_[data_idx_]->data(this).count(), batchsize_*vdim_);
    auto src = Tensor2(srclayers_[data_idx_]->mutable_data(this));
    Copy(data, src);
  } else if (phase == kNegative) {   /*negative phase*/
      if (is_first_iteration_vis_) {
        CHECK_EQ(srclayers_[data_idx_]->data(this).count(), batchsize_*vdim_);
        auto src = Tensor2(srclayers_[data_idx_]->mutable_data(this));
        auto vis_sample = Tensor2(&vis_sample_);
        Copy(vis_sample, src);
        is_first_iteration_vis_ = false;
      } else {
          auto hid_sample =
                Tensor2(srclayers_[hid_idx_]->mutable_data(this, kNegative));
          // fetch sampling results from hidden layer
          auto vis_sample = Tensor2(&vis_sample_);
          auto weight = Tensor2(weight_->mutable_data());
          auto bias = Tensor1(bias_->mutable_data());
          vis_sample = dot(hid_sample, weight.T());
          vis_sample+=repmat(bias, neg_batchsize_);
          vis_sample = F<op::sigmoid>(vis_sample);
          TSingleton<Random<cpu>>::Instance()->SampleBinary(vis_sample);
        }
    }
}

void RBMVisLayer::ComputeGradient(Phase phase) {
  auto data = Tensor2(&data_);
  auto hid_data = Tensor2(srclayers_[hid_idx_]->mutable_data(this, kPositive));
  auto vis_sample = Tensor2(&vis_sample_);
  auto hid_sample =
       Tensor2(srclayers_[hid_idx_]->mutable_data(this, kNegative));
  // fetch sampling results from hidden layer
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());
  gbias = sum_rows(vis_sample);
  gbias -= sum_rows(data);
  gweight = dot(vis_sample.T(), hid_sample);
  gweight -= dot(data.T(), hid_data);
  gbias*=(1.0f)/(1.0f*batchsize_);
  gweight*=(1.0f)/(1.0f*batchsize_);
}

void RBMVisLayer::ComputeLoss(Metric* perf) {
  float loss = (0.0f);
  CHECK_EQ(srclayers_[data_idx_]->data(this).count(), batchsize_*vdim_);
  auto src = Tensor2(srclayers_[data_idx_]->mutable_data(this));
  auto hid_data = Tensor2(srclayers_[hid_idx_]->mutable_data(this, kPositive));
  // gibbs using u
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());
  Tensor<cpu, 2> reconstruct(Shape2(batchsize_, vdim_)); /*reconstruct error*/
  AllocSpace(reconstruct);
  reconstruct = dot(hid_data, weight.T());
  reconstruct+=repmat(bias, batchsize_);
  reconstruct = F<op::sigmoid>(reconstruct);
  float *src_dptr = src.dptr;
  float *reconstruct_dptr = reconstruct.dptr;
  for (int i = 0; i < vdim_*batchsize_; i++)
    loss += -(src_dptr[i]*log(reconstruct_dptr[i])
            +(1-src_dptr[i])*log(1-reconstruct_dptr[i]));
  loss/=batchsize_;
  FreeSpace(reconstruct);
  perf->Reset();
  perf->Add("reconstruct_error", loss);
}
/**************** Implementation for RBMHidLayer********************/
RBMHidLayer::~RBMHidLayer() {
  delete weight_;
  delete bias_;
}
void RBMHidLayer::Setup(const LayerProto& proto,
      int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src_data = srclayers_[0]->data(this, kPositive);
  const auto& src_sample = srclayers_[0]->data(this, kNegative);
  scale_ = static_cast<float> (1.0f);
  batchsize_ = src_data.shape()[0];
  neg_batchsize_ = src_sample.shape()[0];
  vdim_ = src_data.count()/batchsize_;
  hdim_ = proto.rbmhid_conf().hid_dim();
  data_.Reshape(vector<int>{batchsize_, hdim_});
  hid_sample_.Reshape(vector<int>{neg_batchsize_, hdim_});
  weight_ = Param::Create(proto.param(0));
  bias_ = Param::Create(proto.param(1));
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
  bias_->Setup(proto.param(1), vector<int>{hdim_});
}

void RBMHidLayer::ComputeFeature(Phase phase, Metric* perf) {
  if (phase == kPositive) {  /*postive phase*/
    auto data = Tensor2(&data_);
    CHECK_EQ(srclayers_[0]->data(this, kPositive).count(), batchsize_*vdim_);
    auto src = Tensor2(srclayers_[0]->mutable_data(this, kPositive));
    auto weight = Tensor2(weight_->mutable_data());
    auto bias = Tensor1(bias_->mutable_data());
    data = dot(src, weight);
    data += repmat(bias, batchsize_);
    data = F<op::sigmoid>(data);
  } else if (phase == kNegative) {   /*negative phase*/
      CHECK_EQ(srclayers_[0]->data(this, kNegative).count(),
         neg_batchsize_*vdim_);
      auto src_sample = Tensor2(srclayers_[0]->mutable_data(this, kNegative));
      auto hid_sample = Tensor2(&hid_sample_);
      auto bias = Tensor1(bias_->mutable_data());
      auto weight = Tensor2(weight_->mutable_data());
      hid_sample = dot(src_sample, weight);
      hid_sample += repmat(bias, neg_batchsize_);
      hid_sample = F<op::sigmoid>(hid_sample);
      TSingleton<Random<cpu>>::Instance()->SampleBinary(hid_sample);
    } else if (phase == kLoss) {   /*test phase*/
       auto data = Tensor2(&data_);  // data: sigmoid(Wv+b)
       TSingleton<Random<cpu>>::Instance()->SampleBinary(data);
      }
}
void RBMHidLayer::ComputeGradient(Phase phase) {
  auto data = Tensor2(&data_);
  auto hid_sample = Tensor2(&hid_sample_);
  auto gbias = Tensor1(bias_->mutable_grad());
  gbias = sum_rows(hid_sample);
  gbias -= sum_rows(data);
  gbias *= scale_/(1.0f*batchsize_);
}
/*********** Implementation for InnerProductLayer**********/
InnerProductLayer::~InnerProductLayer() {
  delete weight_;
  delete bias_;
}
void InnerProductLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src=srclayers_[0]->data(this);
  batchsize_=src.shape()[0];
  vdim_=src.count()/batchsize_;
  hdim_=proto.innerproduct_conf().num_output();
  if(partition_dim()>0)
    hdim_ /= npartitions;
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  weight_ = Param::Create(proto.param(0));
  bias_ = Param::Create(proto.param(1));
  weight_->Setup(proto.param(0), vector<int>{hdim_, vdim_});
  bias_->Setup(proto.param(1), vector<int>{hdim_});
}

void InnerProductLayer::ComputeFeature(Phase phase, Metric* perf) {
  auto data = Tensor2(&data_);
  auto src = Tensor2(srclayers_[0]->mutable_data(this));
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());
  data=dot(src, weight.T());
  // repmat: repeat bias vector into batchsize rows
  data+=repmat(bias, batchsize_);
}

void InnerProductLayer::ComputeGradient(Phase phas) {
  auto src = Tensor2(srclayers_[0]->mutable_data(this));
  auto grad = Tensor2(&grad_);
  auto weight = Tensor2(weight_->mutable_data());
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());

  gbias=sum_rows(grad);
  gweight=dot(grad.T(), src);
  if(srclayers_[0]->mutable_grad(this)!=nullptr){
    auto gsrc = Tensor2(srclayers_[0]->mutable_grad(this));
    gsrc=dot(grad, weight);
  }
}
/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::Setup(const LayerProto& proto, int npartitions){
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();  // batchsize() is a function of DataLayer
  data_.Reshape(vector<int>{batchsize});
}

void LabelLayer::ParseRecords(Phase phase, const vector<Record>& records,
    Blob<float>* blob){
  int rid=0;
  float *label= blob->mutable_cpu_data() ;
  for(const Record& record: records){
    label[rid++]=record.image().label();
    //  CHECK_LT(record.image().label(),10);
  }
  CHECK_EQ(rid, blob->shape()[0]);  // In the end, rid should be batchsize
}


// TODO(kaiping): Baseline 1 & 2, check later
/******************** Implementation for DPMLabelParserLayer******************/
void DPMLabelParserLayer::Setup(const LayerProto& proto, int npartitions){
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();  // batchsize() is a function of DataLayer
  data_.Reshape(vector<int>{batchsize});  // batchsize number of labels (corresponding to batchsize number of samples/patients)
}

void DPMLabelParserLayer::ParseRecords(Phase phase, const vector<Record>& records,
                                  Blob<float>* blob){
  int rid=0;
  float *label= blob->mutable_cpu_data() ;
  for(const Record& record: records){  // corresponding to a DPMMultiVectorRecord
     label[rid++]=record.dpm_multi_vector_record().label();
  }
  CHECK_EQ(rid, blob->shape()[0]);  // In the end, rid should be batchsize
}


/***************** Implementation for LRNLayer *************************/
void LRNLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  lsize_ = proto.lrn_conf().local_size();
  CHECK_EQ(lsize_ % 2, 1) << "LRN only supports odd values for Localvol";
  knorm_=proto.lrn_conf().knorm();
  alpha_ = proto.lrn_conf().alpha();
  beta_ = proto.lrn_conf().beta();

  const vector<int>& s=srclayers_[0]->data(this).shape();
  data_.Reshape(s);
  grad_.Reshape(s);
  norm_.Reshape(s);
  batchsize_=s[0];
  channels_=s[1];
  height_=s[2];
  width_=s[3];
}

void LRNLayer::ComputeFeature(Phase phase, Metric* perf) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor4(&data_);
  auto norm = Tensor4(&norm_);
  // stores normalizer without power
  norm= chpool<red::sum>( F<op::square>(src) , lsize_ ) * salpha + knorm_;
  data = src * F<op::power>(norm, -beta_ );
}

void LRNLayer::ComputeGradient(Phase phase) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto norm = Tensor4(&norm_);
  auto grad = Tensor4(&grad_);
  auto gsrc = Tensor4(srclayers_[0]->mutable_grad(this));

  gsrc = grad * F<op::power>( norm, -beta_ );
  gsrc += ( - 2.0f * beta_ * salpha ) * chpool<red::sum>(
      grad * src * F<op::power>( norm, -beta_-1.0f ), lsize_ )  * src;
}

/**************** Implementation for MnistImageLayer******************/

void MnistLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
  LOG_IF(ERROR, records.size()==0)<<"Empty records to parse";
  int ndim=records.at(0).image().shape_size();
  int inputsize =records.at(0).image().shape(ndim-1);
  CHECK_EQ(inputsize, blob->shape()[2]);

  float* dptr=blob->mutable_cpu_data();
  for(const Record& record: records){
    const SingleLabelImageRecord& imagerecord=record.image();
    if(imagerecord.pixel().size()) {
      string pixel=imagerecord.pixel();
      for(int i = 0, k = 0; i < inputsize; i++) {
        for(int j = 0; j < inputsize; j++) {
          // NOTE!!! must cast pixel to uint8_t then to float!!! waste a lot of
          // time to debug this
          float x =  static_cast<float>(static_cast<uint8_t>(pixel[k++]));
          x = x / norm_a_-norm_b_;
          *dptr = x;
          dptr++;
        }
      }
    } else {
      for(int i = 0, k = 0; i < inputsize; i++) {
        for(int j = 0; j < inputsize; j++) {
          *dptr = imagerecord.data(k++) / norm_a_ - norm_b_;
          dptr++;
        }
      }
    }
  }
  CHECK_EQ(dptr, blob->mutable_cpu_data()+blob->count());
}
void MnistLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers_[0])->sample();  // TODO(kaiping):? not understand
  kernel_=proto.mnist_conf().kernel();
  sigma_=proto.mnist_conf().sigma();
  alpha_=proto.mnist_conf().alpha();
  beta_=proto.mnist_conf().beta();
  gamma_=proto.mnist_conf().gamma();
  resize_=proto.mnist_conf().resize();
  norm_a_=proto.mnist_conf().norm_a();
  norm_b_=proto.mnist_conf().norm_b();
  elastic_freq_=proto.mnist_conf().elastic_freq();

  int ndim=sample.image().shape_size();
  CHECK_GE(ndim,2);
  if(resize_)
    data_.Reshape(vector<int>{batchsize, 1, resize_, resize_});
  else{
    int s=sample.image().shape(ndim-1);
    CHECK_EQ(s,sample.image().shape(ndim-2));
    data_.Reshape(vector<int>{batchsize, 1, s, s });
  }
}


// TODO(kaiping): Baseline 1 & 2, check later
/******************** Implementation for DPMFeatureParserLayer******************/
void DPMFeatureParserLayer::ParseRecords(Phase phase,
                                  const vector<Record>& records, Blob<float>* blob){
  LOG_IF(ERROR, records.size()==0)<<"Empty records to parse";
  float* dptr=blob->mutable_cpu_data();  // for assigning proper values to blob, i.e., data_
  for(int i = 0; i < records.size(); i++) {  // each is one dpm_multi_vector_record, corresponding to one patient, i.e., one row; the ith multi-vector
    for(int j = 0; j < window_num_; j++) {
      for(int k = 0; k < feature_num_; k++) {  // in each window, traverse all features; the kth feature
         int index = i * feature_num_ * window_num_ + j * feature_num_ + k;
         dptr[index] = records[i].dpm_multi_vector_record().vectors(j).data(k);
      }
    }
  }
}

void DPMFeatureParserLayer::Setup(const LayerProto &proto, int npartitions) {
   Layer::Setup(proto, npartitions);
   CHECK_EQ(srclayers_.size(),1);
   int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();
   feature_num_ = proto.dpmfeatureparser_conf().feature_num();
   window_num_ = proto.dpmfeatureparser_conf().window_num();
   int totallength = feature_num_ * window_num_;
   data_.Reshape(vector<int>{batchsize, totallength});
}


// TODO(kaiping): Model 1, check later
/******************** Implementation for DPMMultiDestFeatureParserLayer******************/
void DPMMultiDestFeatureParserLayer::ParseRecords(Phase phase, const vector<Record>& records,
            Blob<float>* blob) {
   LOG_IF(ERROR, records.size()==0)<<"Empty records to parse";
   float* dptr=blob->mutable_cpu_data();  // for assigning proper values to blob, i.e., data_
   float* win1_dptr=win1_data_.mutable_cpu_data();
   float* win2_dptr=win2_data_.mutable_cpu_data();
   float* win3_dptr=win3_data_.mutable_cpu_data();
   for(int i = 0; i < records.size(); i++) {  // each is one dpm_multi_vector_record, corresponding to one patient, i.e., one row; the ith multi-vector
      for(int j = 0; j < window_num_; j++) {
        for(int k = 0; k < feature_num_; k++) {  // in each window, traverse all features; the kth feature
          int index = i * feature_num_ * window_num_ + j * feature_num_ + k;
          dptr[index] = records[i].dpm_multi_vector_record().vectors(j).data(k);
          int index_for_win = i * feature_num_ + k;
          win1_dptr[index_for_win] = records[i].dpm_multi_vector_record().vectors(0).data(k);  // corresponding to different subvectors
          win2_dptr[index_for_win] = records[i].dpm_multi_vector_record().vectors(1).data(k);
          win3_dptr[index_for_win] = records[i].dpm_multi_vector_record().vectors(2).data(k);
        }
      }
   }
}

void DPMMultiDestFeatureParserLayer::Setup(const LayerProto& proto, int npartitions) {
   Layer::Setup(proto, npartitions);
   CHECK_EQ(srclayers_.size(),1);
   int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();
   feature_num_ = proto.dpm_multidest_featureparser_conf().feature_num();
   window_num_ = proto.dpm_multidest_featureparser_conf().window_num();
   int totallength = feature_num_ * window_num_;
   // Not add separately
   data_.Reshape(vector<int>{batchsize, totallength});
   // Add separately
   win1_data_.Reshape(vector<int>{batchsize, feature_num_});
   win2_data_.Reshape(vector<int>{batchsize, feature_num_});
   win3_data_.Reshape(vector<int>{batchsize, feature_num_});
}

    Blob<float>* DPMMultiDestFeatureParserLayer::mutable_data(const Layer* from, Phase phase) {
      // "from" here is the dest layer of DPMMultiDestFeatureParser
      if (from != nullptr){
        if ("fc1_window1@00" == from->name())
          return &win1_data_;
        else if ("fc1_window2@00" == from->name())
          return &win2_data_;
        else if ("fc1_window3@00" == from->name())
          return &win3_data_;
        else{
          LOG(ERROR)<<"mutable_data() - no mutable_data returned in the MultiSrcDatalayer return &data_";
          return &data_;
        }
      }
      else{
        LOG(ERROR)<<"mutable_data() - nullptr";
        return &data_;
      }
    }

    const Blob<float>& DPMMultiDestFeatureParserLayer::data(const Layer* from, Phase phase) const {
      if (from != nullptr){
        if ("fc1_window1@00" == from->name())
          return win1_data_;
        else if ("fc1_window2@00" == from->name())
          return win2_data_;
        else if ("fc1_window3@00" == from->name())
          return win3_data_;
        else{
          LOG(ERROR)<<"data() - no data returned in the MultiSrcDatalayer, return data_";
          return data_;
        }
      }
      else{
        LOG(ERROR)<<"data() - nullptr";
        return data_;
      }
    }

/******************** Implementation for PoolingLayer******************/
void PoolingLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  PoolingProto pool_conf = proto.pooling_conf();
  kernel_=pool_conf.kernel();
  stride_=pool_conf.stride();
  CHECK_LT(pad_, kernel_);
  pool_=proto.pooling_conf().pool();
  CHECK(pool_ == PoolingProto_PoolMethod_AVE
        || pool_ == PoolingProto_PoolMethod_MAX)
      << "Padding implemented only for average and max pooling.";

  const auto& srcshape=srclayers_[0]->data(this).shape();
  int dim=srcshape.size();
  CHECK_GT(dim,2);
  width_ = srcshape[dim-1];
  height_ = srcshape[dim-2];
  if(dim>3)
    channels_ = srcshape[dim-3];
  else
    channels_=1;
  batchsize_=srcshape[0];
  pooled_height_ = static_cast<int>((height_ - kernel_) / stride_) + 1;
  pooled_width_ = static_cast<int>(( width_ - kernel_) / stride_) + 1;
  data_.Reshape(vector<int>{batchsize_, channels_, pooled_height_, pooled_width_});
  grad_.ReshapeLike(data_);
}

void PoolingLayer::ComputeFeature(Phase phase, Metric* perf) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor4(&data_);
  if(pool_ == PoolingProto_PoolMethod_MAX)
    data=pool<red::maximum>(src, kernel_, stride_);
  else if(pool_ == PoolingProto_PoolMethod_AVE)
    data=pool<red::sum>(src, kernel_, stride_) *(1.0f/(kernel_*kernel_));
}

/*
 * partition only on num/channel dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient(Phase phase) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto gsrc = Tensor4(srclayers_[0]->mutable_grad(this));
  auto data = Tensor4(&data_);
  auto grad = Tensor4(&grad_);
  if(pool_ == PoolingProto_PoolMethod_MAX)
    gsrc = unpool<red::maximum>(src, data, grad, kernel_, stride_);
  else if(pool_ == PoolingProto_PoolMethod_AVE)
    gsrc = unpool<red::sum>(src, data, grad, kernel_, stride_)
      *(1.0f/(kernel_*kernel_));
}

/***************** Implementation for ReLULayer *****************************/

void ReLULayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(*(srclayers_[0]->mutable_grad(this)));
}

void ReLULayer::ComputeFeature(Phase phase, Metric* perf) {
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data=F<op::relu>(src);
}

void ReLULayer::ComputeGradient(Phase phase) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc=F<op::relu_grad>(data)*grad;
}

/*************** Implementation for RGBImageLayer *************************/

void RGBImageLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
  const vector<int>& s=blob->shape();
  auto images = Tensor4(&data_);
  const SingleLabelImageRecord& r=records.at(0).image();
  Tensor<cpu, 3> raw_image(Shape3(r.shape(0),r.shape(1),r.shape(2)));
  AllocSpace(raw_image);
  Tensor<cpu, 3> croped_image(nullptr, Shape3(s[1],s[2],s[3]));
  if(cropsize_)
    AllocSpace(croped_image);
    //CHECK(std::equal(croped_image.shape(), raw_image.shape());
  int rid=0;
  const float* meandptr=mean_.cpu_data();
  for(const Record& record: records){
    auto image=images[rid];
    bool do_crop=cropsize_>0&&(phase == kTrain);
    bool do_mirror=mirror_&&rand()%2&&(phase == kTrain);
    float* dptr=nullptr;
    if(do_crop||do_mirror)
      dptr=raw_image.dptr;
    else
      dptr=image.dptr;
    if(record.image().pixel().size()){
      string pixel=record.image().pixel();
      for(size_t i=0;i<pixel.size();i++)
        dptr[i]=static_cast<float>(static_cast<uint8_t>(pixel[i]));
    }else {
      memcpy(dptr, record.image().data().data(),
          sizeof(float)*record.image().data_size());
    }
    for(int i=0;i<mean_.count();i++)
      dptr[i]-=meandptr[i];

    if(do_crop){
      int hoff=rand()%(r.shape(1)-cropsize_);
      int woff=rand()%(r.shape(2)-cropsize_);
      Shape<2> cropshape=Shape2(cropsize_, cropsize_);
      if(do_mirror){
        croped_image=crop(raw_image, cropshape, hoff, woff);
        image=mirror(croped_image);
      }else{
        image=crop(raw_image, cropshape, hoff, woff);
      }
    }else if(do_mirror){
      image=mirror(raw_image);
    }
    rid++;
  }
  if(scale_)
    images=images*scale_;

  FreeSpace(raw_image);
  if(cropsize_)
    FreeSpace(croped_image);
}
void RGBImageLayer::Setup(const LayerProto& proto, int npartitions) {
  ParserLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  scale_=proto.rgbimage_conf().scale();
  cropsize_=proto.rgbimage_conf().cropsize();
  mirror_=proto.rgbimage_conf().mirror();
  int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers_[0])->sample();
  vector<int> shape;
  shape.push_back(batchsize);
  for(int x: sample.image().shape()){
    shape.push_back(x);
  }
  CHECK_EQ(shape.size(),4);
  if(cropsize_){
    shape[2]=cropsize_;
    shape[3]=cropsize_;
  }
  data_.Reshape(shape);
  mean_.Reshape({shape[1],shape[2],shape[3]});
  if(proto.rgbimage_conf().has_meanfile()){
    if(proto.rgbimage_conf().meanfile().find("binaryproto") != string::npos) {
      CaffeBlob mean;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &mean);
      CHECK_EQ(mean_.count(), mean.data_size());
      memcpy(mean_.mutable_cpu_data(), mean.data().data(),
          sizeof(float)*mean.data_size());
    } else {
      SingleLabelImageRecord mean;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &mean);
      CHECK_EQ(mean_.count(), mean.data_size());
      memcpy(mean_.mutable_cpu_data(), mean.data().data(),
          sizeof(float)*mean.data_size());
    }
  } else {
    memset(mean_.mutable_cpu_data(),0,sizeof(float)*mean_.count());
  }
}

/***************Implementation for ShardDataLayer**************************/
void ShardDataLayer::ComputeFeature(Phase phase, Metric* perf){
  if (shard_ == nullptr)
    shard_ = new DataShard(layer_proto_.sharddata_conf().path(),
        DataShard::kRead);
  if(random_skip_){
    int nskip = rand() % random_skip_;
    LOG(INFO)<<"Random Skip "<<nskip<<" records, there are "<<shard_->Count()
      <<" records in total";
    string key;
    for(int i=0;i<nskip;i++){
      shard_->Next(&key, &sample_);
    }
    random_skip_=0;
  }
  for(auto& record: records_){
    string key;
    if(!shard_->Next(&key, &record)){
      shard_->SeekToFirst();
      CHECK(shard_->Next(&key, &record));
    }
  }
}

void ShardDataLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  shard_= new DataShard(proto.sharddata_conf().path(), DataShard::kRead);
  string key;
  shard_->Next(&key, &sample_);
  delete shard_;
  shard_ = nullptr;
  batchsize_=proto.sharddata_conf().batchsize();
  if(partition_dim() == 0)
    batchsize_ /= npartitions;

  records_.resize(batchsize_);
  random_skip_=proto.sharddata_conf().random_skip();
}

ShardDataLayer::~ShardDataLayer() {
  if (shard_ != nullptr)
    delete shard_;
  shard_ = nullptr;
}
/*******************Implementation of TanLayer***************************/
void TanhLayer::Setup(const LayerProto& proto, int npartitions){
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(srclayers_[0]->grad(this));
}

void TanhLayer::ComputeFeature(Phase phase, Metric* perf) {
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data=F<op::stanh>(src);
}

void TanhLayer::ComputeGradient(Phase phase) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc=F<op::stanh_grad>(data)*grad;
}
/********** * Implementation for SoftmaxLossLayer*************************/
void SoftmaxLossLayer::Setup(const LayerProto& proto, int npartitions) {
  LossLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),2);
  data_.Reshape(srclayers_[0]->data(this).shape());
  batchsize_=data_.shape()[0];
  dim_=data_.count()/batchsize_;
  topk_=proto.softmaxloss_conf().topk();
  metric_.Reshape(vector<int>{2});
  scale_=proto.softmaxloss_conf().scale();
}
void SoftmaxLossLayer::ComputeFeature(Phase phase, Metric* perf) {
  Shape<2> s=Shape2(batchsize_, dim_);
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers_[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src);  // prob is the destination
  const float* label=srclayers_[1]->data(this).cpu_data();
  const float* probptr=prob.dptr;
  float loss=0, precision=0;
  for(int n=0;n<batchsize_;n++){
    int ilabel=static_cast<int>(label[n]);  // appropriate ground truth/label value
    //  CHECK_LT(ilabel,10);
    CHECK_GE(ilabel,0);
    float prob_of_truth=probptr[ilabel];
    loss-=log(std::max(prob_of_truth, FLT_MIN));
    vector<std::pair<float, int> > probvec;
    for (int j = 0; j < dim_; ++j) {
      probvec.push_back(std::make_pair(probptr[j], j));
    }
    std::partial_sort(
        probvec.begin(), probvec.begin() + topk_,
        probvec.end(), std::greater<std::pair<float, int> >());
    // check if true label is in top k predictions
    for (int k = 0; k < topk_; k++) {  // In our case, topk = 1, limit to only 1 result
      if (probvec[k].second == static_cast<int>(label[n])) {
        precision++;
        break;
      }
    }
    probptr+=dim_;  // change to the next row, corresponding to the next sample
  }
  CHECK_EQ(probptr, prob.dptr+prob.shape.Size());
  perf->Add("loss", loss*scale_/(1.0f*batchsize_));
  perf->Add("accuracy", precision*scale_/(1.0f*batchsize_));
}

void SoftmaxLossLayer::ComputeGradient(Phase phase) {
  const float* label=srclayers_[1]->data(this).cpu_data();
  Blob<float>* gsrcblob=srclayers_[0]->mutable_grad(this);
  gsrcblob->CopyFrom(data_);  // only copy value, for non-ground truth positions
  float* gsrcptr=gsrcblob->mutable_cpu_data();
  for(int n=0;n<batchsize_;n++){
    gsrcptr[n*dim_+static_cast<int>(label[n])]-=1.0f;
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));  // deal with all rows in the batch together
  gsrc*=scale_/(1.0f*batchsize_);  // TODO(kaiping): why need this?
}



// TODO(kaiping): Model 1, check later
/******************** Implementation for DPMCombineSoftmaxLossLayer******************/
    void DPMCombineSoftmaxLossLayer::Setup(const LayerProto& proto, int npartitions) {
      LossLayer::Setup(proto, npartitions);
      CHECK_EQ(srclayers_.size(),4);  // 3 innerproduct layers and 1 label layer, order: win1, win2, win3, label
      data_.Reshape(srclayers_[0]->data(this).shape());
      batchsize_=data_.shape()[0];
      dim_=data_.count()/batchsize_;  // i.e., neu2
      topk_=proto.dpm_combine_softmaxloss_conf().topk();
      scale_=proto.dpm_combine_softmaxloss_conf().scale();
      //metric_.Reshape(vector<int>{2});
      //scale_=proto.softmaxloss_conf().scale();
      win1_softmax_.Reshape(srclayers_[0]->data(this).shape());  // Reshape is like Initialization
      win2_softmax_.Reshape(srclayers_[0]->data(this).shape());
      win3_softmax_.Reshape(srclayers_[0]->data(this).shape());
    }
    void DPMCombineSoftmaxLossLayer::ComputeFeature(Phase phase, Metric* perf) {
      Shape<2> s=Shape2(batchsize_, dim_); // Softmax must use Tensor, i.e., [batch_size, neu2]

      auto prob_win1 = Tensor2(&win1_softmax_);  // for the layer with name "fc1_window1"
      Tensor<cpu, 2> src_win1(srclayers_[0]->mutable_data(this)->mutable_cpu_data(), s);
      Softmax(prob_win1, src_win1);

      auto prob_win2 = Tensor2(&win2_softmax_);  // for the layer with name "fc1_window2"
      Tensor<cpu, 2> src_win2(srclayers_[1]->mutable_data(this)->mutable_cpu_data(), s);
      Softmax(prob_win2, src_win2);

      auto prob_win3 = Tensor2(&win3_softmax_);  // for the layer with name "fc1_window3"
      Tensor<cpu, 2> src_win3(srclayers_[2]->mutable_data(this)->mutable_cpu_data(), s);
      Softmax(prob_win3, src_win3);

      // TODO(kaiping): to check whether 2 Tensor2 tensors can be added etc. element-wise
      auto data = Tensor2(&data_);
      data = (prob_win1 + prob_win2 + prob_win3) / 3.0;

      const float* label=srclayers_[3]->data(this).cpu_data();
      const float* probptr=data.dptr;
      float loss=0, precision=0;
      for(int n=0;n<batchsize_;n++){
        int ilabel=static_cast<int>(label[n]);  // appropriate ground truth/label value
        //  CHECK_LT(ilabel,10);
        CHECK_GE(ilabel,0);  // Change label information to be 0, 1, 2, 3
        CHECK_LT(ilabel,4);
        float prob_of_truth=probptr[ilabel];  // the probability of the ground truth, used in loss function
        loss-=log(std::max(prob_of_truth, FLT_MIN));
        vector<std::pair<float, int> > probvec;
        for (int j = 0; j < dim_; ++j) {
          probvec.push_back(std::make_pair(probptr[j], j));
        }
        std::partial_sort(
                probvec.begin(), probvec.begin() + topk_,
                probvec.end(), std::greater<std::pair<float, int> >());
        // check if true label is in top k predictions
        for (int k = 0; k < topk_; k++) {  // In our case, topk = 1, limit to only 1 result
          if (probvec[k].second == static_cast<int>(label[n])) {  // label[n] is the ground truth, means this is correct
            precision++;
            break;
          }
        }
        probptr+=dim_;  // change to the next row, corresponding to the next sample
      }
      CHECK_EQ(probptr, data.dptr+data.shape.Size());
      perf->Add("loss", loss*scale_/(1.0f*batchsize_));
      perf->Add("accuracy", precision*scale_/(1.0f*batchsize_));
    }

    void DPMCombineSoftmaxLossLayer::ComputeGradient(Phase phase) {
      const float* label=srclayers_[3]->data(this).cpu_data();
      float* softmax1 = win1_softmax_.mutable_cpu_data();
      float* softmax2 = win2_softmax_.mutable_cpu_data();
      float* softmax3 = win3_softmax_.mutable_cpu_data();
      float* datainfo = data_.mutable_cpu_data();

      Blob<float>* gsrcblob_1=srclayers_[0]->mutable_grad(this);
      Blob<float>* gsrcblob_2=srclayers_[1]->mutable_grad(this);
      Blob<float>* gsrcblob_3=srclayers_[2]->mutable_grad(this);

      // TODO(kaiping) need to check the formula and usage for this
      gsrcblob_1->CopyFrom(win1_softmax_);
      gsrcblob_2->CopyFrom(win2_softmax_);
      gsrcblob_3->CopyFrom(win3_softmax_);

      float* gsrcptr_1=gsrcblob_1->mutable_cpu_data();
      float* gsrcptr_2=gsrcblob_2->mutable_cpu_data();
      float* gsrcptr_3=gsrcblob_3->mutable_cpu_data();

      for(int n=0;n<batchsize_;n++){
        gsrcptr_1[n*dim_+static_cast<int>(label[n])]-=1.0f; // for the ground truth's position
        gsrcptr_2[n*dim_+static_cast<int>(label[n])]-=1.0f;
        gsrcptr_3[n*dim_+static_cast<int>(label[n])]-=1.0f;
      }

      for(int n = 0;n < batchsize_;n++){
        for(int idx = 0;idx < dim_; idx++){
            gsrcptr_1[n * dim_ + idx] = gsrcptr_1[n * dim_ + idx] * softmax1[n*dim_+static_cast<int>(label[n])] / (3.0 * datainfo[n*dim_+static_cast<int>(label[n])]);
            gsrcptr_2[n * dim_ + idx] = gsrcptr_2[n * dim_ + idx] * softmax2[n*dim_+static_cast<int>(label[n])] / (3.0 * datainfo[n*dim_+static_cast<int>(label[n])]);
            gsrcptr_3[n * dim_ + idx] = gsrcptr_3[n * dim_ + idx] * softmax3[n*dim_+static_cast<int>(label[n])] / (3.0 * datainfo[n*dim_+static_cast<int>(label[n])]);
        }
      }

      Tensor<cpu, 1> gsrc1(gsrcptr_1, Shape1(gsrcblob_1->count()));  // deal with all rows in the batch together
      gsrc1*=scale_/(1.0f*batchsize_);  // TODO(kaiping): why need this? Actually compute an average gsrc of the whole batch
      Tensor<cpu, 1> gsrc2(gsrcptr_2, Shape1(gsrcblob_2->count()));  // deal with all rows in the batch together
      gsrc2*=scale_/(1.0f*batchsize_);
      Tensor<cpu, 1> gsrc3(gsrcptr_3, Shape1(gsrcblob_3->count()));  // deal with all rows in the batch together
      gsrc3*=scale_/(1.0f*batchsize_);
    }


}  // namespace singa
