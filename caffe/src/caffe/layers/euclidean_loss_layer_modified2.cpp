#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void M2EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  chroma_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void M2EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
  caffe_copy(count, bottom[0]->cpu_data(), chroma_.mutable_cpu_data()); 
  Dtype *chroma = chroma_.mutable_cpu_data(); 
  const Dtype* bottom_data = bottom[0]->cpu_data();
   //(n, k, h, w)
  const int num = bottom[0]->num();
  const int channel = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const float eps = 0.00001;
  const float alpha = 1;
  const float th = 0.001;
  Dtype r_, g_, b_;
  for (int n = 0; n < num; ++n) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        r_ = bottom_data[((n*channel+0)*height+h)*width+w];
        g_ = bottom_data[((n*channel+1)*height+h)*width+w];
        b_ = bottom_data[((n*channel+2)*height+h)*width+w];
        chroma[((n*channel+0)*height+h)*width+w] = chroma[((n*channel+0)*height+h)*width+w]/pow((pow(r_,2)+pow(g_,2)+pow(b_,2)),0.5);//((n * K + k) * H + h) * W + w
        chroma[((n*channel+1)*height+h)*width+w] = chroma[((n*channel+1)*height+h)*width+w]/pow((pow(r_,2)+pow(g_,2)+pow(b_,2)),0.5);
        chroma[((n*channel+2)*height+h)*width+w] = chroma[((n*channel+2)*height+h)*width+w]/pow((pow(r_,2)+pow(g_,2)+pow(b_,2)),0.5);
              }
          }
      }
  Dtype c, t, d, l, r;
  Dtype cr, cg, cb, tr, tg, tb, dr, dg, db, lr, lg, lb, rr, rg, rb;
  Dtype w1, w2, w3, w4;
  for (int n = 0; n < num; ++n) {
    for (int h = 1; h < height - 1; ++h) {
      for (int w = 1; w < width - 1; ++w) {
        cr = chroma[((n*channel+0)*height+h)*width+w];
        cg = chroma[((n*channel+1)*height+h)*width+w];
        cb = chroma[((n*channel+2)*height+h)*width+w];                
        tr = chroma[((n*channel+0)*height+h-1)*width+w];
        tg = chroma[((n*channel+1)*height+h-1)*width+w];
        tb = chroma[((n*channel+2)*height+h-1)*width+w];
        dr = chroma[((n*channel+0)*height+h+1)*width+w];
        dg = chroma[((n*channel+1)*height+h+1)*width+w];
        db = chroma[((n*channel+2)*height+h+1)*width+w];
        lr = chroma[((n*channel+0)*height+h)*width+w-1];
        lg = chroma[((n*channel+1)*height+h)*width+w-1];
        lb = chroma[((n*channel+2)*height+h)*width+w-1];
        rr = chroma[((n*channel+0)*height+h)*width+w+1];
        rg = chroma[((n*channel+1)*height+h)*width+w+1];
        rb = chroma[((n*channel+2)*height+h)*width+w+1];
        if (2*(1-cr*dr+cg*dg+cb*db) < th)
          w1 = 1;
        else
          w1 = 0;
        if (2*(1-(cr*rr+cg*rg+cb*rb)) < th)
          w2 = 1;
        else
          w2 = 0;
        if (2*(1-(tr*cr+tg*cg+tb*cb)) < th)
          w3 = 1;
        else
          w3 = 0;
        if (2*(1-(lr*cr+lg*cg+lb*cb)) < th)
          w4 = 1;
        else
          w4 = 0;
        for (int k = 0; k < channel; ++k) {
          c = bottom_data[((n*channel+k)*height+h)*width+w];
          t = bottom_data[((n*channel+k)*height+h-1)*width+w];
          d = bottom_data[((n*channel+k)*height+h+1)*width+w];
          l = bottom_data[((n*channel+k)*height+h)*width+w-1];
          r = bottom_data[((n*channel+k)*height+h)*width+w+1];          
          diff_prior[((n*channel+k)*height+h)*width+w] = diff_prior[((n*channel+k)*height+h)*width+w] + 0.1*(w1*copysign(alpha*pow(abs(c-d) + eps,alpha-1), c-d) + w2*copysign(alpha*pow(abs(c-r) + eps,alpha-1), c-r) - w3*copysign(alpha*pow(abs(t-c) + eps, alpha-1), t-c) - w4*copysign(alpha*pow(abs(l-c) + eps, alpha-1), l-c)); //((n * K + k) * H + h) * W + w
                }
            }
        }
    }
  caffe_copy(count, diff_prior, diff_.mutable_cpu_data()); 
  top[0]->mutable_cpu_data()[0] = loss; 
}

template <typename Dtype>
void M2EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(M2EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(M2EuclideanLossLayer);
REGISTER_LAYER_CLASS(M2EuclideanLoss);

}  // namespace caffe
