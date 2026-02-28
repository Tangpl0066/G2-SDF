#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> ray_matching_cuda(
                               const at::Tensor w_h_24,
							   const at::Tensor w_h_3,
							   const at::Tensor grid,
							   const int width, 
							   const int height,
							   const float bounding_box_min,
							   const float bounding_box_max,
							   const int grid_res,
							   const float dw,
							   const float dh,
							   const float r_x,  
							   const float r_y,  
							   const float r_z,
							   const float d_x,  
							   const float d_y,  
							   const float d_z,
                 const int point_max,
                 const float step_size);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> ray_matching(
                               const at::Tensor w_h_24,
							   const at::Tensor w_h_3,
							   const at::Tensor grid,
							   const int width, 
							   const int height,
							   const float bounding_box_min,
							   const float bounding_box_max,
							   const int grid_res,
							   const float dw,
							   const float dh,
							   const float r_x,  
							   const float r_y,  
							   const float r_z,
							   const float d_x,  
							   const float d_y,  
							   const float d_z,
                 const int point_max,
                 const float step_size) {
    CHECK_INPUT(w_h_3);
    CHECK_INPUT(w_h_24);
	CHECK_INPUT(grid);

    return ray_matching_cuda(w_h_24, w_h_3,
								grid, width, height,
								bounding_box_min, bounding_box_max,
								grid_res, dw, dh,
								r_x, r_y, r_z,
								d_x, d_y, d_z, point_max, step_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ray_matching", &ray_matching, "Ray Matching");
}