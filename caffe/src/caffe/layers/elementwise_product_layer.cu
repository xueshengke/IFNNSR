// written by Shengke Xue, Ph.D., Zhejiang University, 2018, email: xueshengke@zju.edu.cn
#include <vector>

#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/elementwise_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void ElementWiseProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();

	for ( int i = 0; i < num ; i++ )
	{
		for( int j = 0; j < ch; j++ )
		{
			const Dtype* bottom_data_part = bottom_data + (i * ch + j) * dim;
			Dtype* top_data_part = top_data + (i * ch + j) * dim;
			const Dtype* weight_part = weight + (j / share) * dim;
			caffe_gpu_mul( dim, bottom_data_part, weight_part, top_data_part );
			if (bias_term) { 
			    const Dtype* bias = this->blobs_[1]->gpu_data();	
			    const Dtype* bias_part = bias + (j / share) * dim; 
			    caffe_gpu_axpy( dim, Dtype(1), bias_part, top_data_part );   
			}
		}
	}
}

template<typename Dtype>
void ElementWiseProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* weight = this->blobs_[0]->gpu_data();

	Blob<Dtype> mid_weight_result(1, ch/share, hei, wid);
	memset(mid_weight_result.mutable_cpu_data(), 0, sizeof(Dtype)*(ch/share)*hei*wid);
    Dtype* weight_diff = mid_weight_result.mutable_gpu_data();

    Blob<Dtype> sin_weight_result(1, 1, hei, wid);
    memset(sin_weight_result.mutable_cpu_data(), 0, sizeof(Dtype)*hei*wid);
    Dtype* sin_weight_diff = sin_weight_result.mutable_gpu_data();

  	Blob<Dtype> mid_bias_result(1, ch/share, hei, wid);
	memset(mid_bias_result.mutable_cpu_data(), 0, sizeof(Dtype)*(ch/share)*hei*wid);
	Dtype* bias_diff = mid_bias_result.mutable_gpu_data();

	Blob<Dtype> sin_bias_result(1, 1, hei, wid);
	memset(sin_bias_result.mutable_cpu_data(), 0, sizeof(Dtype)*hei*wid);
	Dtype* sin_bias_diff = sin_bias_result.mutable_gpu_data();
	
	// Gradient with respect to weight and bias
	for ( int i = 0; i < num ; i++ )
	{
        caffe_gpu_set((ch/share)*dim, Dtype(0), weight_diff);
		if (bias_term) { 
		  caffe_gpu_set((ch/share)*dim, Dtype(0), bias_diff);  
		}
	    for( int j = 0; j < ch; j++ )
	    {
            const Dtype* top_diff_part = top_diff + (i * ch + j) * dim;
		    const Dtype* bottom_data_part = bottom_data + (i * ch + j) * dim;
		    caffe_gpu_mul( dim, top_diff_part, bottom_data_part, sin_weight_diff );
		    Dtype* weight_diff_part = weight_diff + (j / share) * dim;
            caffe_gpu_axpy( dim, Dtype(1), sin_weight_diff, weight_diff_part );
            if (bias_term)
			{
				caffe_gpu_axpby( dim, Dtype(1), top_diff_part, Dtype(0), sin_bias_diff );
				Dtype* bias_diff_part = bias_diff + (j / share) * dim;
				caffe_gpu_axpy( dim, Dtype(1), sin_bias_diff, bias_diff_part );
			}
	    }
	    // caffe_gpu_scal((ch/share)*dim, 1.0/share, weight_diff);
		// caffe_gpu_scal((ch/share)*dim, 1.0/share, bias_diff);
		caffe_gpu_axpy( (ch/share)*dim, Dtype(1), weight_diff, this->blobs_[0]->mutable_gpu_diff() );
		if (bias_term) {
		  caffe_gpu_axpy( (ch/share)*dim, Dtype(1), bias_diff, this->blobs_[1]->mutable_gpu_diff() );
		}
	}

  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    for ( int i = 0; i < num ; i++ )
    {
        for( int j = 0; j < ch; j++ )
        {
            const Dtype* top_diff_part = top_diff + (i * ch + j) * dim;
            Dtype* bottom_diff_part = bottom_diff + (i * ch + j) * dim;
            const Dtype* weight_part = weight + (j / share) * dim;
            caffe_gpu_mul( dim, top_diff_part, weight_part, bottom_diff_part );
        }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ElementWiseProductLayer);

}  // namespace caffe
