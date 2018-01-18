#include <vector>
#include <cfloat>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/quasi_angular_margin_inner_product_layer.hpp"

namespace caffe {
template <typename Dtype>
__global__ void Weight_norm_gpu(int nthreads, const int K_,
    Dtype* weight) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  	Dtype sum_square = 0.;
  	for (int i = 0; i < K_; i++) {
  	  sum_square += weight[index * K_ + i] * weight[index * K_ + i];
  	}
  	sum_square = sqrt(sum_square);
    for (int i = 0; i < K_; i++) {
  	  weight[index * K_ + i] = weight[index * K_ + i] / sum_square;
  	}
  }
}

template <typename Dtype>
__global__ void Compute_bottom_norm_gpu(int nthreads, const int K_,
    const Dtype* bottom, Dtype* x_norm) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype sum_square = 0.;
    for (int i = 0; i < K_; i++) {
      sum_square += bottom[index * K_ + i] * bottom[index * K_ + i];
    }
    x_norm[index] = sqrt(sum_square);
  }
}

template <typename Dtype>
__global__ void Compute_cos_theta_gpu(int nthreads, const int N_,
    const Dtype* x_norm, Dtype* cos_theta) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / N_;
    cos_theta[index] = cos_theta[index] / x_norm[i];
  }
}

template <typename Dtype>
__global__ void Margin_Type1_forward_gpu(int nthreads, const int N_,
    Dtype alpha, Dtype beta, const Dtype* label, const Dtype* x_norm,
    Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // the label[i]_th top_data
    const int label_value = static_cast<int>(label[index]);
    top[index*N_ + label_value] = top[index*N_ + label_value] * alpha
                                   + x_norm[index] * beta;
  }
}

template <typename Dtype>
__global__ void Margin_Type2_forward_gpu(int nthreads, const int N_, 
    Dtype alpha, Dtype beta, const Dtype* label, Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // the label[i]_th top_data
    const int label_value = static_cast<int>(label[index]);
    top[index*N_ + label_value] = top[index*N_ + label_value] * alpha + beta;
  }
}

// template <typename Dtype>
// __global__ void Margin_Type1_weight_backward_gpu(int nthreads, const int N_, const int K_, 
//             Dtype lambda,
//             const Dtype* bottom, const Dtype* weight, const Dtype* top_diff, const Dtype* label,
//             const Dtype* x_norm, Dtype* bottom_diff) {
//   CUDA_KERNEL_LOOP(index, nthreads) {
//     const int i = index / K_;
//     const int j = index % K_;
//     bottom_diff[index] = (Dtype)0.;
//     const int label_value = static_cast<int>(label[i]);
//     for (int n = 0; n < N_; n++) {
//       if (label_value != n) {
//         bottom_diff[index] += top_diff[i * N_ + n] * weight[n * K_ + j];
//       } else {
//         Dtype coeff_w = (Dtype)4. * sign_0[i * N_ + n] * cos_theta[i * N_ + n];
//         Dtype coeff_x = - (Dtype)1./ x_norm[i] * ((Dtype)2. * sign_0[i * N_ + n] *  
//                      cos_theta_quadratic[i * N_ + n] + (Dtype)1.);
//         Dtype coeff_norm = sqrt(coeff_w * coeff_w + coeff_x * coeff_x);
//         coeff_w = coeff_w / coeff_norm;
//         coeff_x = coeff_x / coeff_norm;
//         bottom_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[i * N_ + n] * 
//                               (coeff_w * weight[n * K_ + j] + coeff_x * bottom[index]);
//         bottom_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[i * N_ + n] * weight[n * K_ + j];
//       }
//     }
//   }
// }

// template <typename Dtype>
// __global__ void Margin_Type1_bottom_backward_gpu(int nthreads, const int N_, const int K_, 
//             Dtype lambda,
//             const Dtype* bottom, const Dtype* weight, const Dtype* top_diff, const Dtype* label,
//             const Dtype* x_norm, Dtype* bottom_diff) {
//   CUDA_KERNEL_LOOP(index, nthreads) {
//     const int i = index / K_;
//     const int j = index % K_;
//     bottom_diff[index] = (Dtype)0.;
//     const int label_value = static_cast<int>(label[i]);
//     for (int n = 0; n < N_; n++) {
//       if (label_value != n) {
//         bottom_diff[index] += top_diff[i * N_ + n] * weight[n * K_ + j];
//       } else {
//         Dtype coeff_w = (Dtype)4. * sign_0[i * N_ + n] * cos_theta[i * N_ + n];
//         Dtype coeff_x = - (Dtype)1./ x_norm[i] * ((Dtype)2. * sign_0[i * N_ + n] *  
//                      cos_theta_quadratic[i * N_ + n] + (Dtype)1.);
//         Dtype coeff_norm = sqrt(coeff_w * coeff_w + coeff_x * coeff_x);
//         coeff_w = coeff_w / coeff_norm;
//         coeff_x = coeff_x / coeff_norm;
//         bottom_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[i * N_ + n] * 
//                               (coeff_w * weight[n * K_ + j] + coeff_x * bottom[index]);
//         bottom_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[i * N_ + n] * weight[n * K_ + j];
//       }
//     }
//   }
// }

template <typename Dtype>
void QuasiAngularInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  /************************* normalize weight *************************/
  int nthreads = N_;
  Weight_norm_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_,
                                this->blobs_[0]->mutable_gpu_data());

  /************************* common variables *************************/
  // x_norm_ = |x|
  nthreads = M_;
  Compute_bottom_norm_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom_data,
                                x_norm_.mutable_gpu_data());

  /************************* Forward *************************/
  if (ip_type_<2) {
  // Modified Inner Product TYPE #0:
  //   + normalized weights

    caffe_gpu_gemm(CblasNoTrans, CblasTrans, 
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
      const Dtype* label = bottom[1]->gpu_data();

      // // the label[i]_th top_data
      // for (int i = 0; i < M_; i++) {
      //   const int label_value = static_cast<int>(label[i]);
      //   top_data[i * N_ + label_value] = top_data[i * N_ + label_value] * alpha_ + x_norm_data[i] * beta_;
      // }
      nthreads = M_;

      Margin_Type1_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, alpha_, beta_, label, x_norm_.gpu_data(), top_data);
      
    }
  } else {
    // Modified Inner Product TYPE #2:
    //   + normalized weights
    //   + normalized features

    // cos_theta = x'w/|x|
    // Dtype* cos_theta_data = cos_theta.mutable_gpu_data();
    Dtype* cos_theta_data = top[0]->mutable_gpu_data();

    nthreads = M_ * N_;
    // cos_theta = x'w / |x|
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
        bottom_data, weight, (Dtype)0., cos_theta_data);
    Compute_cos_theta_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, x_norm_.gpu_data(), cos_theta_data);
    
    // caffe_copy(bottom[0].count(), bottom_data, normalized_x_);

    for (int i = 0; i < M_; i++) {
      caffe_gpu_scal(N_, (Dtype)1./mutable_x_norm_data[i], cos_theta_data + i * N_);

      // x/|x|
      // caffe_scal(N_, (Dtype)1./mutable_x_norm_data[i], normalized_x_ + i * N_);
    }
    // x/|x|
    if (this->phase_==TRAIN && ip_type_==2)
    {
      caffe_copy(M_*N_, bottom_data, normalized_x_->mutable_gpu_data());
      for (int i = 0; i < M_; i++) {
        caffe_gpu_scal(N_, (Dtype)1./mutable_x_norm_data[i], normalized_x_ + i * N_);
      } 
    }
    // caffe_copy(M_ * N_, cos_theta->gpu_data(), top_data);

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
      const Dtype* label = bottom[1]->gpu_data();

      // // the label[i]_th top_data
      // for (int i = 0; i < M_; i++) {
      //   const int label_value = static_cast<int>(label[i]);
      //   top_data[i * N_ + label_value] = top_data[i * N_ + label_value] * alpha_ + beta_;
      // }

      nthreads = M_;

      Margin_Type2_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, alpha_, beta_, label, top_data);
    }
  }
}

template <typename Dtype>
void QuasiAngularInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_data = top[0]->gpu_data();  
  const Dtype* top_diff = top[0]->gpu_diff();

  const Dtype* bottom_data = bottom[0]->gpu_data();

  const Dtype* x_norm_data = x_norm_.gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
 
  Dtype* W_diff = this->blobs_[0]->mutable_gpu_data();
  Dtype* bottom_diff = this->bottom[0]->mutable_gpu_diff();

  const Dtype* label = NULL;

  if(ip_type_&0x01)
  {
    label = bottom[1]->gpu_data();
  }

  // Gradient with respect to weight
  if (this->param_propagate_down_[0]) {
    switch (ip_type_) {
      case 0:
        // Modified Inner Product TYPE #0:
        //   + normalized weights

        // the same as original Inner Product
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
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
            if(label_value==j){
              caffe_gpu_axpy(K_, alpha_ * top_diff[i * N_ + j],
                bottom_data + i * K_, W_diff + j * K_);
            } else {
              caffe_gpu_axpy(K_, (Dtype)1., 
                bottom_data + i * K_, W_diff + j * K_);
            }
          } 
        }
      break;

    case 2:
      // Modified Inner Product TYPE #3:
      //   + normalized weights
      //   + normalized features
      
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, x_norm_data,
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
          caffe_gpu_axpy(K_, tmp_alpha,
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
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* x_norm_data = x_norm_.gpu_data();

    switch (ip_type_) {
      case 0:
        // Modified Inner Product TYPE #1:
        //   + normalized weights

        // the same as original Inner product
        // top_diff * W
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 
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
              caffe_gpu_memcpy(K_, weight + j * K_, tmp_K_diff_);

              caffe_gpu_axpby(K_,
                top_diff_val * beta_ / x_norm_[i] , bottom_data + i * K_, 
                top_diff_val * alpha_, tmp_K_diff_);
              
              caffe_gpu_axpy(K_, (Dtype)1., tmp_K_diff_, bottom_diff_ + i * K_);
            } else {
              caffe_gpu_axpy(K_, top_diff_val, weight + j * K_, bottom_diff_ + i * K_);
            }
          } 
        }
        break;

      case 2:
        // Modified Inner Product TYPE #3:
        //   + normalized weights
        //   + normalized features

        caffe_gpu_set(M_ * K_, Dtype(0), bottom_diff);

        // the same  as the case 3, but in a faster way
        for (int i = 0; i < M_; i++) {
          for (int j = 0; j < N_; j++) {
            Dtype tmp_1 = caffe_gpu_dot(K_, weight + j*K_, bottom_data + i*K_);
            Dtype tmp_2 = top_diff[i*N_+j] / x_norm_[i];
            Dtype tmp_3 = (tmp_1 * tmp_2) / ( x_norm_[i] * x_norm_[i]);

            caffe_gpu_memcpy(K_, bottom_data + i*K_, tmp_K_diff_);
            caffe_gpu_axpby(K_, tmp_2, tmp_K_diff_, tmp_3, tmp_K_diff_);
            caffe_gpu_axpy(K_, 1.0, tmp_K_diff_, bottom_diff + i * K_);
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
            Dtype tmp_1 = caffe_gpu_dot(K_, weight + j*K_, bottom_data + i*K_);
            Dtype tmp_2 = top_diff[i*N_+j] / x_norm_[i];
            Dtype tmp_3 = (tmp_1 * tmp_2) / ( x_norm_[i] * x_norm_[i]);
            Dtype tmp_alpha = (label_value==j) ? alpha_ : 1.0;

            caffe_gpu_set(K_, Dtype(0), tmp_K_diff_);

            for (int k = 0; k < K_; k++) {
              tmp_K_diff_[k] += weight[j*K_+k] * tmp_2 + bottom_data[j*K_+k] * tmp_3;
            }
            caffe_gpu_axpy(K_, tmp_alpha, tmp_K_diff_, bottom_diff + i * K_);
          }
        }
        break;

      default:
        LOG(FATAL) << "Unknown Inner Product type.";
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(QuasiAngularInnerProductLayer);

}  // namespace caffe
