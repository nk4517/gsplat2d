#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "floatN.cuh"

// for f : R(n) -> R(m), J in R(m, n),
// v is cotangent in R(m), e.g. dL/df in R(m),
// compute vjp i.e. vT J -> R(n)
__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float2* __restrict__ extents,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float3* __restrict__ v_conic,
    float3* __restrict__ v_cov2d,
    float2* __restrict__ v_mean2d
);


__global__ void project_gaussians_backward_kernel_cholesky(
    const int num_points,
    const float2* __restrict__ extents,
    const float3* __restrict__ cholesky,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float3* __restrict__ v_conic,
    float3* __restrict__ v_cholesky,
    float2* __restrict__ v_mean2d
);

template<bool WITH_UPSCALE_GRADS>
__global__ void rasterize_backward_kernel_unified(
    const uint32_t num_images,
    const uint32_t num_tiles_per_image,
    const uint32_t tile_size,
    const uint32_t n_isects,
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_grouped,
    const int32_t* __restrict__ tile_offsets,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const int* __restrict__ final_index,
    const float* __restrict__ out_T,
    const float* __restrict__ out_T_dx,
    const float* __restrict__ out_T_dy,
    const float* __restrict__ out_T_dxy,
    const float* __restrict__ out_S_xy_cross,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_T,
    const float3* __restrict__ v_output_dx,
    const float3* __restrict__ v_output_dy,
    const float3* __restrict__ v_output_dxy,
    const float* __restrict__ v_T_dx,
    const float* __restrict__ v_T_dy,
    const float* __restrict__ v_T_dxy,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_xy_abs,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
);
