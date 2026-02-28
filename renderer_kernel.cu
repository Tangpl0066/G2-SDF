#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <ATen/ATen.h>

namespace {
    __device__ __forceinline__ int flat(float const x, float const y, float const z, int const grid_res) {
        return __float2int_rd(z) + __float2int_rd(y) * grid_res + __float2int_rd(x) * grid_res * grid_res;
    }

	__device__ float CalSDF(
				const float* grid,
				const float point_x,
				const float point_y,
				const float point_z,
				const float voxel_size,
				const float bounding_box_min,
				const int grid_res){
		
		// S1: min_index
		const float p1_x = floorf((point_x - bounding_box_min) / voxel_size);
		const float p1_y = floorf((point_y - bounding_box_min) / voxel_size);
		const float p1_z = floorf((point_z - bounding_box_min) / voxel_size);
		
		// S2.1: proportion
		const float xd = (point_x - (p1_x * voxel_size + bounding_box_min)) / voxel_size;
		const float yd = (point_y - (p1_y * voxel_size + bounding_box_min)) / voxel_size;
		const float zd = (point_z - (p1_z * voxel_size + bounding_box_min)) / voxel_size;
		
		// S2.2: sdf 8
		const float sdf1 = grid[flat(p1_x,     p1_y,     p1_z,     grid_res)];
		const float sdf2 = grid[flat(p1_x,     p1_y + 1, p1_z,     grid_res)];
		const float sdf3 = grid[flat(p1_x + 1, p1_y + 1, p1_z,     grid_res)];
		const float sdf4 = grid[flat(p1_x + 1, p1_y,     p1_z,     grid_res)];
		const float sdf5 = grid[flat(p1_x,     p1_y,     p1_z + 1, grid_res)];
		const float sdf6 = grid[flat(p1_x,     p1_y + 1, p1_z + 1, grid_res)];
		const float sdf7 = grid[flat(p1_x + 1, p1_y + 1, p1_z + 1, grid_res)];
		const float sdf8 = grid[flat(p1_x + 1, p1_y,     p1_z + 1, grid_res)];
		
		// S2.3: interpolation
		const float sdf_a = (1-yd)*sdf4 + yd*sdf3;
		const float sdf_b = (1-yd)*sdf1 + yd*sdf2;
		const float sdf_c = (1-yd)*sdf5 + yd*sdf6;
		const float sdf_d = (1-yd)*sdf8 + yd*sdf7;
		
		const float sdf_e = (1-xd)*sdf_c + xd*sdf_d;
		const float sdf_f = (1-xd)*sdf_b + xd*sdf_a;
		
		return (1-zd)*sdf_f + zd*sdf_e;
	}

    __global__ void GenerateRay(
                float* origins, 
                float* directions,
                const int width, 
                const int height, 
		        const float dw,
		        const float dh,
                const float r_x, 
                const float r_y, 
                const float r_z,
		        const float d_x,
                const float d_y, 
                const float d_z) {

        const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (pixel_index < width * height) {
            const int col = pixel_index % width;
            const int row = pixel_index / width;
            const int i = 3 * pixel_index;
			
			// direction	
			directions[i] = r_y * d_z - d_y * r_z;
            directions[i+1] = d_x * r_z - r_x * d_z;
            directions[i+2] = r_x * d_y - d_x * r_y;

            // origin
            const float u = (height/2 - row) * dh - dh/2;
			const float v = (col - width/2) * dw + dw/2;

            origins[i] = u * r_x + v * d_x;
            origins[i+1] = u * r_y + v * d_y;
            origins[i+2] = u * r_z + v * d_z;
            
        }
    }  
	
	
	__global__ void SurfacePoint(
        const float* grid,
        const float* origins,
        const float* directions,
        const float bounding_box_min,
        const float bounding_box_max,
        const int grid_res,
        float* voxel_position,
        float* intersection_pos, 
        const int width, 
        const int height,
        const int point_max,
        const float step_size) {
		
		const float voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1);
		const float length_max = sqrtf(3 * powf((bounding_box_max - bounding_box_min), 2));
		const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (pixel_index < width * height) {
			const int i = 3 * pixel_index;
			const int j = point_max * pixel_index;
			int point_num = 0;
			float step = -length_max/2;
			
			float pre_point_x = origins[i] + step * directions[i];
			float pre_point_y = origins[i+1] + step * directions[i+1];
			float pre_point_z = origins[i+2] + step * directions[i+2];
			float pre_sdf = 1;
			
			float sample_point_x = 10.0;
			float sample_point_y = 0.0;
			float sample_point_z = 0.0;
			float sample_point_sdf = 0.0;

			
			
			while (step <= length_max/2){
				// S1: Sample point
				sample_point_x = origins[i] + step * directions[i];
				sample_point_y = origins[i+1] + step * directions[i+1];
				sample_point_z = origins[i+2] + step * directions[i+2];
				
				
				// S2: point in bounding box 
				if (sample_point_x <= bounding_box_min || sample_point_x >= bounding_box_max || sample_point_y <= bounding_box_min || sample_point_y >= bounding_box_max || sample_point_z <= bounding_box_min || sample_point_z >= bounding_box_max){
					sample_point_sdf = 100;
				}
				else{
					// S3: cal sdf
					sample_point_sdf = CalSDF(grid, sample_point_x, sample_point_y, sample_point_z, 
												voxel_size, bounding_box_min, grid_res);
					
					// S4: surface point
					if (pre_sdf * sample_point_sdf < 0){
						
						// S4.1: positive sdf
						if (pre_sdf < 0){
							pre_sdf = -pre_sdf;
						}
						else{
							sample_point_sdf = -sample_point_sdf;
						}
						
						// S4.2: compare sdf
						float p1_x = 0;
						float p1_y = 0;
						float p1_z = 0;
						float p1_sdf = 0;

						float p2_x = 0;
						float p2_y = 0;
						float p2_z = 0;
						float p2_sdf = 0;

						if (pre_sdf < sample_point_sdf){
							p1_x = pre_point_x;
							p1_y = pre_point_y;
							p1_z = pre_point_z;
							p1_sdf = pre_sdf;
							
							p2_x = sample_point_x;
							p2_y = sample_point_y;
							p2_z = sample_point_z;
							p2_sdf = sample_point_sdf;
						}
						else{
							p1_x = sample_point_x;
							p1_y = sample_point_y;
							p1_z = sample_point_z;
							p1_sdf = sample_point_sdf;
							
							p2_x = pre_point_x;
							p2_y = pre_point_y;
							p2_z = pre_point_z;
							p2_sdf = pre_sdf;
						}
						
						int flag = 1;
						float new_point_x = 0;
						float new_point_y = 0;
						float new_point_z = 0;
						float new_point_sdf = 0;
						
						while(p1_sdf > 0.01 && flag == 1){
						
							// S4.3: new_point
							new_point_x = p1_x + (p2_x - p1_x) * p1_sdf;
							new_point_y = p1_y + (p2_y - p1_y) * p1_sdf;
							new_point_z = p1_z + (p2_z - p1_z) * p1_sdf;
							new_point_sdf = CalSDF(grid, new_point_x, new_point_y, new_point_z,voxel_size, bounding_box_min, grid_res);
							if (new_point_sdf < 0){
								new_point_sdf = -new_point_sdf;
							}
							
							// S4.4: compare sdf
							if (new_point_sdf < p1_sdf){
							    if (CalSDF(grid, p1_x, p1_y, p1_z,voxel_size, bounding_box_min, grid_res) * CalSDF(grid, new_point_x, new_point_y, new_point_z,voxel_size, bounding_box_min, grid_res) < 0) {
                                    p2_x = p1_x;
                                    p2_y = p1_y;
                                    p2_z = p1_z;
                                    p2_sdf = p1_sdf;
							    }

								
								p1_x = new_point_x;
								p1_y = new_point_y;
								p1_z = new_point_z;
								p1_sdf = new_point_sdf;
							}
							else if (new_point_sdf < p2_sdf){
								p2_x = new_point_x;
								p2_y = new_point_y;
								p2_z = new_point_z;
								p2_sdf = new_point_sdf;
							}
							else{
								flag = 0;
							}
						}
						
						// S4.5: record
						
						intersection_pos[j + point_num] = p1_x;
						intersection_pos[j + point_num + 1] = p1_y;
						intersection_pos[j + point_num + 2] = p1_z;
						
						voxel_position[j + point_num] = floorf((p1_x - bounding_box_min) / voxel_size);
						voxel_position[j + point_num + 1] = floorf((p1_y - bounding_box_min) / voxel_size);
						voxel_position[j + point_num + 2] = floorf((p1_z - bounding_box_min) / voxel_size);
						
						point_num = point_num + 3;
					}

					sample_point_sdf = CalSDF(grid, sample_point_x, sample_point_y, sample_point_z,
													voxel_size, bounding_box_min, grid_res);
				}
				
				// S4.6: 
				pre_point_x = sample_point_x;
				pre_point_y = sample_point_y;
				pre_point_z = sample_point_z;


				pre_sdf = sample_point_sdf;
				
				step = step + step_size;
			}
			
			// S5: 
			while(point_num < point_max){
				intersection_pos[j + point_num] = -1;
				intersection_pos[j + point_num + 1] = -1;
				intersection_pos[j + point_num + 2] = -1;
				
				voxel_position[j + point_num] = -1;
				voxel_position[j + point_num + 1] = -1;
				voxel_position[j + point_num + 2] = -1;
				
				point_num = point_num + 3;
			}
		}
	}
} 


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
                           const float step_size) {

    const int thread = 512;

    at::Tensor origins = at::zeros_like(w_h_3);
    at::Tensor directions = at::zeros_like(w_h_3);

        GenerateRay<<<(width * height + thread - 1) / thread, thread>>>(
                                     origins.data<float>(), 
                                     directions.data<float>(), 
                                     width, 
                                     height,
									 dw,
									 dh,
									 r_x,
									 r_y,
									 r_z,
									 d_x,
									 d_y,
									 d_z);     
	
	at::Tensor voxel_position = at::zeros_like(w_h_24);
    at::Tensor intersection_pos = at::zeros_like(w_h_24);
	
		SurfacePoint<<<(width * height + thread - 1) / thread, thread>>>(
                                                    grid.data<float>(), 
                                                    origins.data<float>(), 
                                                    directions.data<float>(),
                                                    bounding_box_min,
                                                    bounding_box_max,
                                                    grid_res, 
                                                    voxel_position.data<float>(), 
                                                    intersection_pos.data<float>(), 
                                                    width, 
                                                    height,
                                                    point_max,
                                                    step_size);

    return {origins, directions, voxel_position, intersection_pos};
}



