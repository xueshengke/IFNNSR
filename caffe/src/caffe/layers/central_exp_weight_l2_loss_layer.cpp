// written by Shengke Xue, Ph.D., Zhejiang University, 2019, email: xueshengke@zju.edu.cn
#include <vector>

#include "caffe/layers/central_exp_weight_l2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CentralExpWeightL2LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff.ReshapeLike(*bottom[0]);
  vector<int> dims = bottom[0]->shape();
  count = bottom[0]->count();
  num = dims[0];
  ch  = dims[1];
  hei = dims[2];
  wid = dims[3];
  
  W.Reshape(dims);
  memset(W.mutable_cpu_data(), 0, sizeof(Dtype)*count);
  Dtype* pW = W.mutable_cpu_data();
  Blob<Dtype> mat(1, 1, hei, wid);
  memset(mat.mutable_cpu_data(), 0, sizeof(Dtype)*hei*wid);
  Dtype* pmat = mat.mutable_cpu_data();
  
  Dtype x[hei];
  Dtype y[wid];
  memset(x, 0, sizeof(x));
  memset(y, 0, sizeof(y));
  for(int i = 0; i <= hei/2; i ++)
  {
	x[i] = exp( ((hei/2-i) * 1.0 / (hei/2)) * ((hei/2-i) * 1.0 / (hei/2)) );
	x[hei-1-i] = x[i];
  }
  for(int j = 0; j <= wid/2; j ++)
  {
	y[j] = exp( ((wid/2-j) * 1.0 / (wid/2)) * ((wid/2-j) * 1.0 / (wid/2)) );
	y[wid-1-j] = y[j];
  }
  
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, hei, wid, 1, 
  				 Dtype(1), &x[0], &y[0], Dtype(0), pmat); // outer product x * y^T
  // caffe_sqrt(hei*wid, pmat, pmat);	// sqrt weight here, it will be square in l2 norm
  for(int i = 0; i < num; i ++)
  {
	for(int j = 0; j < ch; j ++)
	{
	  Dtype* pW_part = pW + (i * ch + j) * hei * wid;
	  caffe_copy(hei*wid, pmat, pW_part); // copy matrix to weight blob
	}
  }
}

template <typename Dtype>
void CentralExpWeightL2LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_sub(count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff.mutable_cpu_data());
  caffe_mul(count,
      W.cpu_data(),
      diff.mutable_cpu_data(),
      diff.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff.cpu_data(), diff.cpu_data());
  Dtype loss = dot / num / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CentralExpWeightL2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Blob<Dtype> grad(num, ch, hei, wid);
  // memset(grad.mutable_cpu_data(), 0, sizeof(Dtype)*count);
  Dtype* pgrad = grad.mutable_cpu_data();
  caffe_mul(count, W.cpu_data(), W.cpu_data(), pgrad);
  caffe_mul(count, diff.cpu_data(), pgrad, pgrad);
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / num;
      caffe_cpu_axpby(
          count,                          // count
          alpha,                          // alpha
          pgrad,                          // a
          Dtype(0),                       // beta
          bottom[i]->mutable_cpu_diff()); // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CentralExpWeightL2LossLayer);
#endif

INSTANTIATE_CLASS(CentralExpWeightL2LossLayer);
REGISTER_LAYER_CLASS(CentralExpWeightL2Loss);

}  // namespace caffe
