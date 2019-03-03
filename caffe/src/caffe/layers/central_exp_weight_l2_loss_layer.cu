// written by Shengke Xue, Ph.D., Zhejiang University, 2019, email: xueshengke@zju.edu.cn
#include <vector>

#include "caffe/layers/central_exp_weight_l2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CentralExpWeightL2LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff.mutable_gpu_data());
  caffe_gpu_mul(
      count,
      W.gpu_data(),
      diff.mutable_gpu_data(),
      diff.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff.gpu_data(), diff.gpu_data(), &dot);
  Dtype loss = dot / num / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CentralExpWeightL2LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Blob<Dtype> grad(num, ch, hei, wid);
  // memset(grad.mutable_cpu_data(), 0, sizeof(Dtype)*count);
  Dtype* pgrad = grad.mutable_gpu_data();
  caffe_gpu_mul(count, W.gpu_data(), W.gpu_data(), pgrad);
  caffe_gpu_mul(count, diff.gpu_data(), pgrad, pgrad);
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / num;
      caffe_gpu_axpby(
          count,             			  // count
          alpha,                          // alpha
          pgrad,                          // a
          Dtype(0),                       // beta
          bottom[i]->mutable_gpu_diff()); // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CentralExpWeightL2LossLayer);

}  // namespace caffe
