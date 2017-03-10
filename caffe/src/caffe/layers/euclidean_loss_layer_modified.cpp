#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  Dtype *diff_prior = diff_.mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data(); // network output
  const Dtype* bottom_label = bottom[1]->cpu_data(); // label
   //(n, k, h, w)
  const int num = bottom[0]->num();
  const int channel = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const float alpha = 0.1;
  Dtype c, t, d, l, r;
  Dtype lc, lt, ltr, ld, ll, lld, lr;
  Dtype g1, g2, g3, g4, gtg1, gtg3, gtg4;
  for (int n = 0; n < num; ++n) {
    for (int k = 0; k < channel; ++k) {
      for (int h = 1; h < height - 1; ++h) {
        for (int w = 1; w < width - 1; ++w) {
          c = bottom_data[((n*channel+k)*height+h)*width+w];
          t = bottom_data[((n*channel+k)*height+h-1)*width+w];
          d = bottom_data[((n*channel+k)*height+h+1)*width+w];
          l = bottom_data[((n*channel+k)*height+h)*width+w-1];
          r = bottom_data[((n*channel+k)*height+h)*width+w+1];
          lc = bottom_label[((n*channel+k)*height+h)*width+w];
          lt = bottom_label[((n*channel+k)*height+h-1)*width+w];
          ltr = bottom_label[((n*channel+k)*height+h-1)*width+w+1];
          ld = bottom_label[((n*channel+k)*height+h+1)*width+w];
          ll = bottom_label[((n*channel+k)*height+h)*width+w-1];
          lld = bottom_label[((n*channel+k)*height+h+1)*width+w-1];
          lr = bottom_label[((n*channel+k)*height+h)*width+w+1];
          if (abs(c-d) < 0.002)
            g1 = 0;
          else
            g1 = copysign(alpha*pow(abs(c-d),alpha-1), c-d);
          if (abs(c-r) < 0.002)
            g2 = 0;
          else
            g2 = copysign(alpha*pow(abs(c-r),alpha-1), c-r);
          if (abs(t-c) < 0.002)
            g3 = 0;
          else
            g3 = copysign(alpha*pow(abs(t-c), alpha-1), t-c);   
          if (abs(l-c) < 0.002)
            g4 = 0;
          else
            g4 = copysign(alpha*pow(abs(l-c), alpha-1), l-c);
          gtg1 = sqrt((lc-ld)*(lc-ld)+(lc-lr)*(lc-lr)); 
          gtg3 = sqrt((lt-lc)*(lt-lc)+(lt-ltr)*(lt-ltr));
          gtg4 = sqrt((ll-lc)*(ll-lc)+(ll-lld)*(ll-lld));
          gtg1 = pow(2.71,-20*gtg1*gtg1);
          gtg3 = pow(2.71,-20*gtg3*gtg3);
          gtg4 = pow(2.71,-20*gtg4*gtg4);                    
          //if (gtg < 0.1)
            diff_prior[((n*channel+k)*height+h)*width+w] = diff_prior[((n*channel+k)*height+h)*width+w] + 0.01*(gtg1*g1+gtg1*g2-gtg3*g3-gtg4*g4); //((n * K + k) * H + h) * W + w
                }
            }
        }
    }
  caffe_copy(count, diff_prior, diff_.mutable_cpu_data()); 
  top[0]->mutable_cpu_data()[0] = loss; 
}

template <typename Dtype>
void MEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(MEuclideanLossLayer);
REGISTER_LAYER_CLASS(MEuclideanLoss);

}  // namespace caffe
