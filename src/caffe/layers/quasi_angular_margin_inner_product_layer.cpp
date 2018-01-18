#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/quasi_angular_margin_inner_product_layer.hpp"

namespace caffe {

template <typename Dtype>
void QuasiAngularMarginInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ip_type_ = this->layer_param_.quasi_angular_margin_inner_product_param().ip_type();
  alpha_ = (this->layer_param_.quasi_angular_margin_inner_product_param().alpha();
  beta_ = this->layer_param_.quasi_angular_margin_inner_product_param().beta();

  if (ip_type_<0 || ip_type_>3){
    LOG(FATAL) << "Parameter 'tpye' must be one of [0,1,2,3]";
  }

  if (this->phase_==TRAIN 
  && (ip_type_ & 0x1)){
    CHECK_EQ(bottom.size(), 2)) << "In train phase, there must be 2 bottom blobs,"
      << "one for data, the other for class labels.";

    CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "Number of labels must match number of output; "
      << "DO NOT support multi-label this version."
      << "e.g., if prediction shape is (M X N), "
      << "label count (number of labels) must be M, "
      << "with integer values in {0, 1, ..., N-1}.";
  }

  forward_without_labels_ = false;
  if (this->phase_==TEST){
    if(this->layer_param_.quasi_angular_margin_inner_product_param().forward_without_labels()
    || bottom.size()==1) {
      LOG(INFO) << "Forward without labels.";
      forward_without_labels_ = true;
    }
  }

  if (ip_type_ & 0x1){
    if(alpha_ < 1) {
      LOG(FATAL) << "QuasiAngularMarginInnerProductLayer must have alpha>=1;";
    }
    if(beta_ > 0) {
      LOG(FATAL) << "QuasiAngularMarginInnerProductLayer must have beta<=0;";
    }
  }

  // beta_ = beta_ - (alpha-1)
  beta_ -= alpha_ - 1;

  const int num_output = this->layer_param_.quasi_angular_margin_inner_product_param().num_output();
  N_ = num_output;
  const int aX[i]s = bottom[0]->CanonicalAX[i]sIndex(
      this->layer_param_.quasi_angular_margin_inner_product_param().aX[i]s());
  // Dimensions starting from "aX[i]s" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and aX[i]s == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(aX[i]s);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.quasi_angular_margin_inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void QuasiAngularMarginInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int aX[i]s = bottom[0]->CanonicalAX[i]sIndex(
      this->layer_param_.quasi_angular_margin_inner_product_param().aX[i]s());
  const int new_K = bottom[0]->count(aX[i]s);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";

  // The first "aX[i]s" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, aX[i]s);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single aX[i]s with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(aX[i]s + 1);
  top_shape[aX[i]s] = N_;
  top[0]->Reshape(top_shape);

  // if needed, reshape top[1] to output lambda
  // if (top.size() == 2) {
  //   vector<int> alpha_shape(1, 1);
  //   top[1]->Reshape(alpha_shape);
  // }
  
  // common temp variables
  vector<int> shape_1_X_M(1, M_);
  x_norm_.Reshape(shape_1_X_M);

  vector<int> shape_1_X_K(1, K_);
  tmp_K_diff_.Reshape(shape_1_X_K);

  if ( this->phase_==TRAIN && ip_type_==2)
  {
  //   vector<int> shape_K_X_K(K_, K_);

  //   cos_theta_.Reshape(top_shape);
    normalized_x_.ReshapeLike(bottom[0]);
  //   normalized_x_diff_.Reshape(shape_K_X_K);
  }
}

template <typename Dtype>
void QuasiAngularMarginInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // iter_ += (Dtype)1.;

  /************************* normalize weight *************************/
  Dtype* norm_weight = this->blobs_[0]->mutable_cpu_data();
  Dtype temp_norm = (Dtype)0.;
  for (int i = 0; i < N_; i++) {
  	temp_norm = caffe_cpu_dot(K_, norm_weight + i * K_, norm_weight + i * K_);
  	temp_norm = (Dtype)1./sqrt(temp_norm);
  	caffe_scal(K_, temp_norm, norm_weight + i * K_);
  }

  /************************* common variables *************************/
  // x_norm_ = |x|
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* mutable_x_norm_data = x_norm_.mutable_cpu_data();

  for (int i = 0; i < M_; i++) {
    mutable_x_norm_data[i] = sqrt(caffe_cpu_dot(K_, bottom_data + i * K_, bottom_data + i * K_) + (Dtype)1e-6);
  }

  /************************* Forward *************************/
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* x_norm_data = x_norm_.cpu_data();

  if (ip_type_<2) {
    // Modified Inner Product TYPE #0:
    //   + normalized weights

    caffe_cpu_gemm(CblasNoTrans, CblasTrans, 
      M_, N_, K_, 
      (Dtype)1., bottom_data, weight, 
      (Dtype)0., top_data);

    // Modified Inner Product TYPE #1:
    //   + normalized weights
    //   + Quasi-Angular Marginal Softmax preprocess.

    // variables:
    //   X: size M_ X K_, bottom data (flattened), each row is data for one input sample 
    //   W: size N_ X K_, weight matrix, each row is a transform vector for one output class prediction
    //   top_data: size M_ X N_, output class prediction, each row is prediction vector output for one input sample
    //   i: range [0, M_-1], row index (batch index) for X
    //   j: range [0, N_-1], index of class prediction, or row index in W
    //   k: range [0, K_-1], col index for X or W
    //   cos_theta: size M_ X N_, cos_theta[i][j] is cosine of angle between X[i] and W[j]
    
    // conditions:
    //   |W[j]| = 1, for j in [0, N_-1]
    
    // top_data function (original):
    //   --init--
    //   top_data = zeros(N_, K_)
    
    //   --loops--
    //   for each i:
    //     for each j:
    //       if j==label[i]:
    //         top_data[i][j] = |W[j]| * |X[i]| * (alpha_ * cos_theta[i][j] + beta_)
    //           = X[i] * W[j]' * alpha_ + |X[i]| * beta_    (for forward and backward)
    //       else:
    //         top_data[i][j] = |W[j]| * |X[i]| * cos_theta[i][j]
    //           = X[i] * W[j]'    (for forward and backward)
    //     end
    //   end
    
    // top_data function (reformed):
    //   --init--
    //   for all i:
    //     for all j:
    //       top_data[i][j] = X[i] * W[j]'
    //     end
    //   end
    //   --init can be reformed as--
    //   top_data = X*W'
    
    //   --loops--
    //   for each i:
    //     for each j:
    //       if j==label[i]:
    //         top_data[i][j] = top_data[i][j] * alpha_ + |X[i]| * beta_
    //     end
    //   end
    //   --loops can be reformed as--
    //   for each i:
    //     j = label[i]
    //     top_data[i][j] = top_data[i][j] * alpha_ + |X[i]| * beta_
    //   end

    if ( ip_type_==1
    && !forward_without_labels_
    && (alpha_ > 1 || beta_ < 0)){
      const Dtype* label = bottom[1]->cpu_data();

      // the label[i]_th top_data
      for (int i = 0; i < M_; i++) {
        const int label_value = static_cast<int>(label[i]);
        top_data[i * N_ + label_value] = top_data[i * N_ + label_value] * alpha_ + x_norm_data[i] * beta_;
      }
    }
  } else {
    // Modified Inner Product TYPE #2:
    //   + normalized weights
    //   + normalized features

    // cos_theta = x'w/|x|
    // Dtype* cos_theta_data = cos_theta.mutable_cpu_data();
    Dtype* cos_theta_data = top[0]->mutable_cpu_data();

    caffe_cpu_gemm(CblasNoTrans, CblasTrans,
      M_, N_, K_,
      (Dtype)1., bottom_data, weight,
      (Dtype)0., cos_theta_data);
    
    // caffe_copy(M_*N_, bottom_data, normalized_x_);

    for (int i = 0; i < M_; i++) {
      caffe_scal(N_, (Dtype)1./mutable_x_norm_data[i], cos_theta_data + i * N_);

      // x/|x|
      // caffe_scal(N_, (Dtype)1./mutable_x_norm_data[i], normalized_x_ + i * N_);
    }

    // x/|x|
    if (this->phase_==TRAIN && ip_type_==2)
    {
      caffe_copy(M_*N_, bottom_data, normalized_x_->mutable_cpu_data());
      for (int i = 0; i < M_; i++) {
        caffe_scal(N_, (Dtype)1./mutable_x_norm_data[i], normalized_x_ + i * N_);
      } 
    }
    // caffe_copy(M_ * N_, cos_theta->cpu_data(), top_data);

    // Modified Inner Product TYPE #4:
    //   + normalized weights
    //   + normalized features
    //   + Quasi-Angular Marginal Softmax preprocess.
    //
    // variables:
    //   X: size M_ X K_, bottom data (flattened), each row is data for one input sample 
    //   W: size N_ X K_, weight matrix, each row is a transform vector for one output class prediction
    //   top_data: size M_ X N_, output class prediction, each row is prediction vector output for one input sample
    //   i: range [0, M_-1], row index (batch index) for X
    //   j: range [0, N_-1], index of class prediction, or row index in W
    //   k: range [0, K_-1], col index for X or W
    //   cos_theta: size M_ X N_, cos_theta[i][j] is cosine of angle between X[i] and W[j]
    
    // conditions: 
    //    X_1[i] = X[i] / |X[i]|, |X_1[i]| = 1, for i in [0, M_-1]
    //    |W[j]| = 1, for j in [0, N_-1]
    
    // top_data function (original):
    //   --init--
    //   top_data = zeros(N_, K_)
    
    //   --loops--
    //   for each i:
    //     for each j:
    //       if j==label[i]:
    //         top_data[i][j] = |W[j]| * |X_1[i]| * (alpha_ * cos_theta[i][j] + beta_)
    //           = cos_theta[i][j] * alpha_ + beta_         (for forward)
    //           = X_1[i] * W[j]' * alpha_ + beta_
    //           = X[i] * W[j]' / |X[i]| * alpha_ + beta_    (for backward)
    //       else:
    //         top_data[i][j] = |W[j]| * |X_1[i]| * cos_theta[i][j]
    //           = cos_theta[i][j]          (for forward)
    //           = X_1[i] * W[j]'
    //           = X[i] * W[j]' / |X[i]|    (for backward)
    //     end
    //   end
    
    // top_data function (reformed):
    //   --init--
    //   for all i:
    //     for all j:
    //       top_data[i][j] = cos_theta[i][j]
    //     end
    //   end
    //   --init can be reformed as--
    //   Omit the index i and j:
    //   top_data = cos_theta
    
    //   --loops--
    //   for each i:
    //     for each j:
    //       if j==label[i]:
    //         top_data[i][j] = top_data[i][j] * alpha_ + beta_
    //     end
    //   end
    //   --loops can be reformed as--
    //   for each i:
    //     j = label[i]
    //     top_data[i][j] = top_data[i][j] * alpha_ + beta_
    //   end

    if ( ip_type_==3
    && !forward_without_labels_ 
    && (alpha_ > 1 || beta_ < 0) ){
      const Dtype* label = bottom[1]->cpu_data();

      // the label[i]_th top_data
      for (int i = 0; i < M_; i++) {
        const int label_value = static_cast<int>(label[i]);
        top_data[i * N_ + label_value] = top_data[i * N_ + label_value] * alpha_ + beta_;
      }
    }
  }
}

template <typename Dtype>
void QuasiAngularMarginInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_data = top[0]->cpu_data();  
  const Dtype* top_diff = top[0]->cpu_diff();

  const Dtype* bottom_data = bottom[0]->cpu_data();

  const Dtype* x_norm_data = x_norm_.cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
 
  Dtype* W_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* bottom_diff = this->bottom[0]->mutable_cpu_diff();

  const Dtype* label = NULL;

  if(ip_type_&0x01)
  {
    label = bottom[1]->cpu_data();
  }

  // Gradient with respect to weight
  if (this->param_propagate_down_[0]) {
    switch (ip_type_) {
      case 0:
        // Modified Inner Product TYPE #0:
        //   + normalized weights

        // the same as original Inner Product
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_, K_, M_,
            (Dtype)1., top_diff, bottom_data,
            (Dtype)1., W_diff);
        break;

      case 1: 
        // Modified Inner Product TYPE #1:
        //   + normalized weights
        //   + Quasi-Angular Marginal Softmax preprocess.

        // variables:
        //   X: size M_ X K_, bottom data (flattened), each row is data for one input sample 
        //   W: size N_ X K_, weight matrix, each row is a transform vector for one output class prediction
        //   top_diff: size M_ X N_
        //   W_diff: size N_ X K_
        //   i: range [0, M_-1], row index (batch index) for X
        //   j: range [0, N_-1], index of class prediction, or row index in W
        //   k: range [0, K_-1], col index for X or W
        //   cos_theta: size M_ X N_, cos_theta[i][j] is cosine of angle between X[i] and W[j]
        
        // conditions: 
        //   |W[j]| = 1, for j in [0, N_-1]
        
        // W_diff function:
        //   --init--
        //   W_diff = zeros(N_, K_),
          
        //   --original loops--
        //   for each i:
        //     for each j:
        //       for each k:
        //         if label[i]==j:
        //             W_diff[j][k] += top_diff[i][j] * X[i][k] * alpha_
        //         else:
        //             W_diff[j][k] += top_diff[i][j] * X[i][k]
        //         end
        //     end
        //   end        

        //   --loops reform 1--
        //   for each j:
        //     for each i:
        //       if label[i]==j:
        //         for each k:
        //           W_diff[j][k] += top_diff[i][j] * X[i][k] * alpha_
        //         end
        //       else:
        //         for each k:
        //           W_diff[j][k] += top_diff[i][j] * X[i][k]
        //         end
        //     end
        //   end
          
        //   --loops reform 2--
        //   for each i:
        //     for each j:
        //       if label[i]==j:
        //         W_diff[j] += X[i] * top_diff[i][j] * alpha_
        //       else:
        //         W_diff[j] += X[i] * top_diff[i][j]
        //     end
        //   end
    
        caffe_gpu_set(N_ * K_, Dtype(0), W_diff);

        // the label[i]_th top_data
        for (int i = 0; i < M_; i++) {
          const int label_value = static_cast<int>(label[i]);
          for (int j = 0; j < N_; j++) {
            Dtype tmp_alpha = (label_value==j) ? (alpha_ * top_diff[i * N_ + j]) : top_diff[i * N_ + j];
            caffe_axpy(K_, tmp_alpha,
              bottom_data + i * K_, W_diff + j * K_);
          } 
        }
      break;
    case 2:
      // Modified Inner Product TYPE #3:
      //   + normalized weights
      //   + normalized features
      
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, normalized_x_,
          (Dtype)1., W_diff);
      break;

    case 3:
      // Modified Inner Product TYPE #4:
      //   + normalized weights
      //   + normalized features
      //   + Quasi-Angular Marginal Softmax preprocess.

      // variables:
      //   X: size M_ X K_, bottom data (flattened), each row is data for one input sample 
      //   W: size N_ X K_, weight matrix, each row is a transform vector for one output class prediction
      //   top_diff: size M_ X N_
      //   W_diff: size N_ X K_
      //   i: range [0, M_-1], row index (batch index) for X
      //   j: range [0, N_-1], index of class prediction, or row index in W
      //   k: range [0, K_-1], col index for X or W
      //   cos_theta: size M_ X N_, cos_theta[i][j] is cosine of angle between X[i] and W[j]

      // conditions: 
      //   X_1[i] = X[i] / |X[i]|, |X_1[i]| = 1, for i in [0, M_-1]
      //   |W[j]| = 1, for j in [0, N_-1]
      
      // W_diff function:
      //   --init--
      //   W_diff = zeros(N_, K_),

      //   --original loops--
      //   for each i:
      //     for each j:
      //       for each k:
      //         if label[i]==j:
      //             W_diff[j][k] += top_diff[i][j] * X_1[i][k] * alpha_
      //         else:
      //             W_diff[j][k] += top_diff[i][j] * X_1[i][k]
      //         end
      //     end
      //   end        

      //   --loops reform 1--
      //   for each j:
      //     for each i:
      //       if label[i]==j:
      //         for each k:
      //           W_diff[j][k] += X_1[i][k] * top_diff[i][j] * alpha_
      //         end
      //       else:
      //         for each k:
      //           W_diff[j][k] += X_1[i][k] * top_diff[i][j]
      //         end
      //     end
      //   end
        
      //   --loops reform 2--
      //   for each i:
      //     for each j:
      //       if label[i]==j:
      //         W_diff[j] += X_1[i] * top_diff[i][j] * alpha_
      //       else:
      //         W_diff[j] += X_1[i] * top_diff[i][j]
      //     end
      //   end
      caffe_gpu_set(N_ * K_, Dtype(0), W_diff);

      // the label[i]_th top_data
      for (int i = 0; i < M_; i++) {
        const int label_value = static_cast<int>(label[i]);

        for (int j = 0; j < N_; j++) {
          Dtype tmp_alpha = (label_value==j) ? alpha_ : ((Dtype)1.);
          caffe_axpy(K_, tmp_alpha,
            x_norm_data + i * K_, W_diff + j * K_);
        }
      }
      break;

    default:
      LOG(FATAL) << "Unknown Inner Product type.";
    }
  }

  // Gradient with respect to bottom data
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* x_norm_data = x_norm_.cpu_data();

    switch (ip_type_) {
      case 0:
        // Modified Inner Product TYPE #1:
        //   + normalized weights

        // the same as original Inner product
        // top_diff * W
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 
          M_, K_, N_, 
          (Dtype)1., top_diff, weight,
          (Dtype)0., bottom_diff);
        break;

      case 1:
        // Modified Inner Product TYPE #2:
        //   + normalized weights
        //   + Quasi-Angular Marginal Softmax preprocess.

        // variables:
        //   X: size M_ X K_, bottom data (flattened), each row is data for one input sample 
        //   W: size N_ X K_, weight matrix, each row is a transform vector for one output class prediction
        //   top_diff: size M_ X N_
        //   bottom_diff: size M_ X K_
        //   i: range [0, M_-1], row index (batch index) for X
        //   j: range [0, N_-1], index of class prediction, or row index in W
        //   k: range [0, K_-1], col index for X or W
        //   cos_theta: size M_ X N_, cos_theta[i][j] is cosine of angle between X[i] and W[j]
        
        // conditions:
        //   |W[j]| = 1, for j in [0, N_-1]
        
        // derivative for vector's norm xi_norm=|X[i]| (here xi_norm is a scalar, and X[i] is a row vector with size 1xK_): 
        //     norm_diff = d(xi_norm) / d(X[i])    (size: 1xK_)
        //     norm_diff[k] = d(xi_norm) / d(X[i][k]) = d(|X[i]|) / d(X[i][k]) = X[i][k] / |X[i]| 
        //   or in vector form:
        //     norm_diff = d(xi_norm) / d(X[i]) = d(|X[i]|) / d(X[i]) = X[i]/|X[i]|

        // bottom_diff function:
        //   --init--
        //   bottom_diff = zeros(M_, K_),
          
        //   --original loops--
        //   for each i:
        //     for each k:
        //       for each j:
        //         if label[i]==j:
        //           bottom_diff[i][k] += top_diff[i][j] * (W[j][k] * alpha_ - X[i][k] / |X[i]| * beta_)
        //         else:
        //           bottom_diff[i][k] += top_diff[i][j] * W[j][k]
        //       end
        //     end
        //   end
          
        //   --loops reform 1--
        //   for each i:
        //     for each j:
        //       if label[i]==j:
        //         for each k:
        //           bottom_diff[i][k] += top_diff[i][j] * (W[j][k] * alpha_ - X[i][k] / |X[i]| * beta_)
        //         end
        //       else:
        //         for each k:
        //           bottom_diff[i][k] += top_diff[i][j] * W[j][k]
        //         end
        //     end
        //   end        
          
        //   --loops reform 2--
        //   for each i:
        //     for each j:
        //       if label[i]==j:
        //         bottom_diff[i] += top_diff[i][j] * (W[j] * alpha_ - X[i] * beta_ / |X[i]|)
        //       else:
        //         bottom_diff[i] += top_diff[i][j] * W[j]
        //     end
        //   end

        caffe_gpu_set(M_ * K_, Dtype(0), bottom_diff);

        // the label[i]_th top_data
        for (int i = 0; i < M_; i++) {
          const int label_value = static_cast<int>(label[i]);

          for (int j = 0; j < N_; j++) {
            top_diff_val = top_diff[i * N_ + j];

            if(label_value==j){
              caffe_copy(K_, weight + j * K_, tmp_K_diff_);

              caffe_cpu_axpby(K_,
                top_diff_val * beta_ / x_norm_[i] , bottom_data + i * K_, 
                top_diff_val * alpha_, tmp_K_diff_);
              
              caffe_axpy(K_, (Dtype)1., tmp_K_diff_, bottom_diff_ + i * K_);
            } else {
              caffe_axpy(K_, top_diff_val, weight + j * K_, bottom_diff_ + i * K_);
            }
          } 
        }
        break;

      case 2:
        // Modified Inner Product TYPE #3:
        //   + normalized weights
        //   + normalized features

        caffe_set(bottom[0]->count_, Dtype(0), bottom_diff);

        // the same  as the case 3, but in a faster way
        for (int i = 0; i < M_; i++) {
          for (int j = 0; j < N_; j++) {
            Dtype tmp_1 = caffe_cpu_dot(K_, weight + j*K_, bottom_data + i*K_);
            Dtype tmp_2 = top_diff[i*N_+j] / x_norm_[i];
            Dtype tmp_3 = (tmp_1 * tmp_2) / ( x_norm_[i] * x_norm_[i]);

            caffe_copy(K_, bottom_data + i*K_, tmp_K_diff_);
            caffe_cpu_axpby(K_, tmp_2, tmp_K_diff_, tmp_3, tmp_K_diff_);
            caffe_cpu_axpy(K_, 1.0, tmp_K_diff_, bottom_diff + i * K_);
          }
        }
        break;

      case 3:
        // Modified Inner Product TYPE #4:
        //   + normalized weights
        //   + normalized features
        //   + Quasi-Angular Marginal Softmax preprocess.

        // variables:
        //   X: size M_ X K_, bottom data (flattened), each row is data for one input sample 
        //   W: size N_ X K_, weight matrix, each row is a transform vector for one output class prediction
        //   top_diff: size M_ X N_
        //   bottom_diff: size M_ X K_
        //   i: range [0, M_-1], row index (batch index) for X
        //   j: range [0, N_-1], index of class prediction, or row index in W
        //   k: range [0, K_-1], col index for X or W
        //   cos_theta: size M_ X N_, cos_theta[i][j] is cosine of angle between X[i] and W[j]

        // conditions: 
        //   X_1[i] = X[i] / |X[i]|, |X_1[i]| = 1, for i in [0, M_-1]
        //   |W[j]| = 1, for j in [0, N_-1]

        // derivatives:
        // (1) for vector's norm xi_norm=|X[i]| (here xi_norm is a scalar, and X[i] is a row vector with size 1xK_): 
        //     norm_diff = d(xi_norm) / d(X[i])    (size: 1xK_)
        //     norm_diff[k] = d(xi_norm) / d(X[i][k]) = d(|X[i]|) / d(X[i][k]) = X[i][k] / |X[i]| 
        //   or in vector form:
        //     norm_diff = d(xi_norm) / d(X[i]) = d(|X[i]|) / d(X[i]) = X[i]/|X[i]|
        // (2) for normalized vector Yi = X[i]/|X[i]| (here Yi and X[i] are row vectors with size 1xK_):
        //     diff_Yi_Xi = d(Yi)/d(X[i])    (size: K_ x K_)
        //     diff_Yi_Xi[k1, k2] = d(Yi[k1]) / d(X[i][k2])
        //       = d(X[i][k1]/|X[i]|) / d(X[i][k2])
        //          /  (|X[i]| - X[i][k1] * (X[i][k1] / |X[i]|)) / |X[i]|^2,    (if k1 = k2)
        //       = { 
        //          \_ -X[i][k1] * (X[i][k2] / |X[i]|) / |X[i]|^2,              (otherwise)
        //          /  1.0/|X[i]| - (X[i][k1] * X[i][k1]) / |X[i]|^3) ,    (if k1 = k2)
        //       = { 
        //          \_ -(X[i][k1] * X[i][k1]) / |X[i]|^3,              (otherwise)
        //   or in K_ x K_ matrix form:
        //     (refer to: http://blog.mmacklin.com/2012/05/)
        //     diff_Yi_Xi = d(Yi)/d(X[i])
        //       = I * (1.0 / |X[i]|) - X[i]' * X[i] * (1.0 / |X[i]|^3)
        //       = (I - (X[i]'* X[i]) / |X[i]|^2)) / |X[i]|

        // bottom_diff function:
        //   --init--
        //   bottom_diff = zeros(M_, K_),
          
        //   --original loops--
        //   for each i:
        //     for each k:
        //       for each j:
        //         if label[i]==j:
        //           bottom_diff[i][k] += top_diff[i][j] * W[j] * col(diff_Yi_Xi, k) * alpha_
        //           (1 x 1)                (1 x 1)      (1 x K_)      (K_ x 1)
        //         else:
        //           bottom_diff[i][k] += top_diff[i][j] * W[j] * col(diff_Yi_Xi, k)
        //           (1 x 1)                (1 x 1)      (1 x K_)      (K_ x 1)
        //       end
        //     end
        //   end
          
        //   --loops reform 1--
        //   for each i:
        //     for each j:
        //       tmp_diff_ik = zeros(1, K_)
        //       for each k:
        //         tmp = W[j] * col(diff_Yi_Xi, k) * top_diff[i][j]
        //             = (W[j][k] - W[j] * X[i]' * X[i][k] / |X[i]|^2) / |X[i]| * top_diff[i][j]
        //         tmp_diff_ik[k] += tmp
        //       end
        //       if label[i]==j:
        //         bottom_diff[i] += tmp_diff_ik * alpha_
        //         (1 x K_)          (1 x K_)
        //       else:
        //         bottom_diff[i] += tmp_diff_ik
        //         (1 x K_)          (1 x K_)
        //     end
        //   end

        //   --loops reform 2--
        //   for each i:
        //     for each j:
        //       tmp_diff_ik = zeros(1, K_)
        //       tmp_1 = W[j] * X[i]'
        //       tmp_2 = top_diff[i][j] / |X[i]|
        //       tmp_3 = tmp_1 * tmp_2 / |X[i]|^2
        //
        //       if label[i]==j:
        //         tmp_alpha = alpha_
        //       else:
        //         tmp_alpha = 1.0
        //
        //       for each k:
        //         tmp = W[j][k] * tmp_2 - X[i][k] * tmp_3
        //         tmp_diff_ik[k] += tmp 
        //       end
        //
        //       bottom_diff[i] += tmp_diff_ik * tmp_alpha
        //       (1 x K_)          (1 x K_)
        //     end
        //   end
        caffe_gpu_set(M_ * K_, Dtype(0), bottom_diff);

        for (int i = 0; i < M_; i++) {
          const int label_value = static_cast<int>(label[i]);
          for (int j = 0; j < N_; j++) {
            Dtype tmp_1 = caffe_cpu_dot(K_, weight + j*K_, bottom_data + i*K_);
            Dtype tmp_2 = top_diff[i*N_+j] / x_norm_[i];
            Dtype tmp_3 = (tmp_1 * tmp_2) / ( x_norm_[i] * x_norm_[i]);
            Dtype tmp_alpha = (label_value==j) ? alpha_ : 1.0;

            caffe_set(K_, Dtype(0), tmp_K_diff_);

            for (int k = 0; k < K_; k++) {
              tmp_K_diff_[k] += weight[j*K_+k] * tmp_2 + bottom_data[j*K_+k] * tmp_3;
            }
            caffe_cpu_axpy(K_, tmp_alpha, tmp_K_diff_, bottom_diff + i * K_);
          }
        }
        break;

      default:
        LOG(FATAL) << "Unknown Inner Product type.";
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(QuasiAngularMarginInnerProductLayer);
#endif

INSTANTIATE_CLASS(QuasiAngularMarginInnerProductLayer);
REGISTER_LAYER_CLASS(QuasiAngularMarginInnerProduct);

}  // namespace caffe
