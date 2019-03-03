// written by Shengke Xue, Ph.D., Zhejiang University, 2018, email: xueshengke@zju.edu.cn
#include <vector>

#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/elementwise_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void ElementWiseProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 1)<< "ElementWiseProduct layer takes a single blob as input.";
	CHECK_EQ(top.size(), 1) << "ElementWiseProduct layer takes a single blob as output.";
	bias_term = this->layer_param_.elementwise_product_param().bias_term();
	share = this->layer_param_.elementwise_product_param().share();
	// Compute the dimensions
	vector<int> dims = bottom[0]->shape(); 	
	num = bottom[0]->num();         // number of samples
	ch = dims[1];					// channel
	hei = dims[2];					// height
	wid = dims[3];					// width
 	dim = hei * wid;
	
	CHECK_LE(share, ch) << "Parameter share cannot be larger than channel (second dimension).";
	CHECK_EQ(ch % share, 0) << "Channel should be divided by share.";
// 	for(int t = 0; t < dims.size(); t ++)
// 	  LOG(INFO) << "bottom dimension " << dims[t];
	
	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
	    // parameter initialization
		if (bias_term) {
			this->blobs_.resize(2);
		} else {
			this->blobs_.resize(1);
		}
		// Intialize the weight
		this->blobs_[0].reset(new Blob<Dtype>(1, ch/share, hei, wid));
		LOG(INFO) << "weight shape " << this->blobs_[0]->shape_string();
		// fill the weights
		shared_ptr<Filler<Dtype> > weight_filler(
				GetFiller<Dtype>(this->layer_param_.elementwise_product_param().weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
		// If necessary, intiialize and fill the bias term
		if (bias_term) {
			this->blobs_[1].reset(new Blob<Dtype>(1, ch/share, hei, wid));
			shared_ptr<Filler<Dtype> > bias_filler(
					GetFiller<Dtype>(this->layer_param_.elementwise_product_param().bias_filler()));
			bias_filler->Fill(this->blobs_[1].get());
		}
	}  
}

template <typename Dtype>
void ElementWiseProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	vector<int> bottom_shape = bottom[0]->shape();
	top[0]->Reshape(bottom_shape);
// 	top[0]->ReshapeLike(*bottom[0]);
// 	LOG(INFO) << "ElementWiseProductLayer";
// 	LOG(INFO) << "bottom shape " << bottom[0]->shape_string(); 
// 	LOG(INFO) << "top shape " << top[0]->shape_string();
}

template<typename Dtype>
void ElementWiseProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	
	for ( int i = 0; i < num ; i++ )
	{
		for( int j = 0; j < ch; j++ )
		{
			const Dtype* bottom_data_part = bottom_data + (i * ch + j) * dim;
			Dtype* top_data_part = top_data + (i * ch + j) * dim;
			const Dtype* weight_part = weight + (j / share) * dim; 
			caffe_mul( dim, bottom_data_part, weight_part, top_data_part );
			if (bias_term) { 
			    const Dtype* bias = this->blobs_[1]->cpu_data();
			    const Dtype* bias_part = bias + (j / share) * dim; 
			    caffe_axpy( dim, Dtype(1), bias_part, top_data_part );   
			}
		}
	}
}

template<typename Dtype>
void ElementWiseProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* weight = this->blobs_[0]->cpu_data();

	Blob<Dtype> mid_weight_result(1, ch/share, hei, wid);
	memset(mid_weight_result.mutable_cpu_data(), 0, sizeof(Dtype)*(ch/share)*hei*wid);
    Dtype* weight_diff = mid_weight_result.mutable_cpu_data();

    Blob<Dtype> sin_weight_result(1, 1, hei, wid);
    memset(sin_weight_result.mutable_cpu_data(), 0, sizeof(Dtype)*hei*wid);
    Dtype* sin_weight_diff = sin_weight_result.mutable_cpu_data();
	
  	Blob<Dtype> mid_bias_result(1, ch/share, hei, wid);
	memset(mid_bias_result.mutable_cpu_data(), 0, sizeof(Dtype)*(ch/share)*hei*wid);
	Dtype* bias_diff = mid_bias_result.mutable_cpu_data();

	Blob<Dtype> sin_bias_result(1, 1, hei, wid);
	memset(sin_bias_result.mutable_cpu_data(), 0, sizeof(Dtype)*hei*wid);
	Dtype* sin_bias_diff = sin_bias_result.mutable_cpu_data();
	
	// Gradient with respect to weight and bias
	for ( int i = 0; i < num ; i++ )
	{
        caffe_set((ch/share)*dim, Dtype(0), weight_diff);
		if (bias_term) { 
		  caffe_set((ch/share)*dim, Dtype(0), bias_diff);  
		}
	    for( int j = 0; j < ch; j++ )
	    {
            const Dtype* top_diff_part = top_diff + (i * ch + j) * dim;
		    const Dtype* bottom_data_part = bottom_data + (i * ch + j) * dim;
		    caffe_mul( dim, top_diff_part, bottom_data_part, sin_weight_diff );
		    Dtype* weight_diff_part = weight_diff + (j / share) * dim;
            caffe_axpy( dim, Dtype(1), sin_weight_diff, weight_diff_part );
            if (bias_term)
			{
				caffe_cpu_axpby( dim, Dtype(1), top_diff_part, Dtype(0), sin_bias_diff );
				Dtype* bias_diff_part = bias_diff + (j / share) * dim;
				caffe_axpy( dim, Dtype(1), sin_bias_diff, bias_diff_part );
			}
	    }
	    // caffe_scal((ch/share)*dim, 1.0/share, weight_diff);
	    // caffe_scal((ch/share)*dim, 1.0/share, bias_diff);
		caffe_axpy( (ch/share)*dim, Dtype(1), weight_diff, this->blobs_[0]->mutable_cpu_diff() );
		if (bias_term) {
		  caffe_axpy( (ch/share)*dim, Dtype(1), bias_diff, this->blobs_[1]->mutable_cpu_diff() );
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
				caffe_mul( dim, top_diff_part, weight_part, bottom_diff_part );
			}
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(ElementWiseProductLayer);
#endif

INSTANTIATE_CLASS(ElementWiseProductLayer);
REGISTER_LAYER_CLASS(ElementWiseProduct);

}  // namespace caffe
