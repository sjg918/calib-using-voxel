
//original paper:
//Embedded Real-time Stereo Estimation via Semi-Global Matching on the{GPU}

//original code:
//https://github.com/dhernandez0/sgm

#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cuda.h>
#include <torch/extension.h>
#include "configuration.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

void SemiGlobalMatchingCudaLauncher(
    const uint8_t * left, const uint8_t * right, uint32_t * d_transform0, uint32_t * d_transform1, uint8_t * d_cost,
    uint8_t * d_L0, uint8_t * d_L1, uint8_t * d_L2, uint8_t * d_L3, uint8_t * d_L4, uint8_t * d_L5, uint8_t * d_L6, uint8_t * d_L7,
    uint8_t * d_disparity, uint8_t * d_disparity_filtered_uchar,
    const int p1, const int p2, const int rows, const int cols, const int pathA
);

at::Tensor sgm(const at::Tensor left, const at::Tensor right, const int p1, const int p2, const int rows, const int cols, const int pathA) {
    //CHECK_INPUT(left);
    //CHECK_INPUT(right);

    const uint8_t * left_ = left.contiguous().data_ptr<uint8_t>();
    const uint8_t * right_ = right.contiguous().data_ptr<uint8_t>();

    auto d_transform0 = at::zeros({rows, cols}, left.options().dtype(at::ScalarType::Int));
    auto d_transform1 = at::zeros({rows, cols}, left.options().dtype(at::ScalarType::Int));
    auto d_cost = at::zeros({rows, cols, MAX_DISPARITY}, left.options());
    auto d_disparity = at::zeros({rows, cols}, left.options());
    auto d_disparity_filtered_uchar = at::zeros({rows, cols}, left.options());
    auto d_L0 = at::zeros({rows, cols, MAX_DISPARITY}, left.options());
    auto d_L1 = at::zeros({rows, cols, MAX_DISPARITY}, left.options());
    auto d_L2 = at::zeros({rows, cols, MAX_DISPARITY}, left.options());
    auto d_L3 = at::zeros({rows, cols, MAX_DISPARITY}, left.options());
    auto d_L4 = at::zeros({rows, cols, MAX_DISPARITY}, left.options());
    auto d_L5 = at::zeros({rows, cols, MAX_DISPARITY}, left.options());
    auto d_L6 = at::zeros({rows, cols, MAX_DISPARITY}, left.options());
    auto d_L7 = at::zeros({rows, cols, MAX_DISPARITY}, left.options());
    uint32_t * d_transform0_ = reinterpret_cast<uint32_t*>(d_transform0.data_ptr<int32_t>());
    uint32_t * d_transform1_ = reinterpret_cast<uint32_t*>(d_transform1.data_ptr<int32_t>());
    uint8_t * d_cost_ = d_cost.data_ptr<uint8_t>();
    uint8_t * d_disparity_ = d_disparity.data_ptr<uint8_t>();
    uint8_t * d_disparity_filtered_uchar_ = d_disparity_filtered_uchar.data_ptr<uint8_t>();
    uint8_t * d_L0_ = d_L0.data_ptr<uint8_t>();
    uint8_t * d_L1_ = d_L1.data_ptr<uint8_t>();
    uint8_t * d_L2_ = d_L2.data_ptr<uint8_t>();
    uint8_t * d_L3_ = d_L3.data_ptr<uint8_t>();
    uint8_t * d_L4_ = d_L4.data_ptr<uint8_t>();
    uint8_t * d_L5_ = d_L5.data_ptr<uint8_t>();
    uint8_t * d_L6_ = d_L6.data_ptr<uint8_t>();
    uint8_t * d_L7_ = d_L7.data_ptr<uint8_t>();

    const int rows_ = rows;
    const int cols_ = cols;

    SemiGlobalMatchingCudaLauncher(
        left_, right_, d_transform0_, d_transform1_, d_cost_,
        d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_, d_L7_,
        d_disparity_, d_disparity_filtered_uchar_,
        p1, p2, rows_, cols_, pathA
    );

    return d_disparity_filtered_uchar;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgm", &sgm,
        "semi global matching on gpu");
}
