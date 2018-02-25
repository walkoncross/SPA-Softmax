#ifndef CAFFE_QUASI_ANGULAR_MARGIN_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_QUASI_ANGULAR_MARGIN_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "marginal fully-connected" layer, computes an marginal inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class QuasiAngularMarginInnerProductLayer : public Layer<Dtype> {
 public:
  explicit MarginInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QuasiAngularMarginInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  
  // common variables
  Blob<Dtype> x_norm_;
  Blob<Dtype> normalized_x_;
//   Blob<Dtype> normalized_x_diff_;
  Blob<Dtype> tmp_K_diff_;
//   Blob<Dtype> cos_theta_; // cos_theta_ is top_data[0] when using "normalize_feature_"

//   int iter_;
  int ip_type_; //must be in one of [0,1,2,3]
  Dtype alpha_;
  Dtype beta_;

  bool forward_without_labels_;
};

}  // namespace caffe

#endif  // CAFFE_QUASI_ANGULAR_MARGIN_INNER_PRODUCT_LAYER_HPP_
