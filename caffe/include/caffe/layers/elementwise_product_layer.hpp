// written by Shengke Xue, Ph.D., Zhejiang University, 2018, email: xueshengke@zju.edu.cn
#ifndef CAFFE_ELEMENTWISE_PRODUCT_LAYER_HPP_
#define CAFFE_ELEMENTWISE_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Use a weighting matrix which has the same size as 
 *        the input blob, pointwise multiyply with the input
 *        
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class ElementWiseProductLayer : public Layer<Dtype> {
 public:
  explicit ElementWiseProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ElementWiseProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // parameters defined here
	int num;
	int ch;
	int hei;
	int wid;
	int dim;
	int share;
	bool bias_term;
};

}  // namespace caffe

#endif  // CAFFE_ELEMENTWISE_PRODUCT_LAYER_HPP_
