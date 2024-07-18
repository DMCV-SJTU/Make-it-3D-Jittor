#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>
#include <float.h>

#include "jt_helper.h"
#include "cuda_util.h"

//#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
//#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
//#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
//#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }


template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(const float x) {
    return copysignf(1.0, x);
}

inline __host__ __device__ float clamp(const float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}

inline __host__ __device__ void swapf(float& a, float& b) {
    float c = a; a = b; b = c;
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
    int exponent;
    frexpf(mx, &exponent); // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, [2, 4) --> 2, ...
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __device__ int mip_from_dt(const float dt, const float H, const float max_cascade) {
    const float mx = dt * H * 0.5;
    int exponent;
    frexpf(mx, &exponent);
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
	uint32_t xx = __expand_bits(x);
	uint32_t yy = __expand_bits(y);
	uint32_t zz = __expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

__global__ void kernel_near_far_from_aabb(
    const PackedVar32<float,2> __restrict__ rays_o,
    const PackedVar32<float,2> __restrict__ rays_d,
    const PackedVar32<float,1> __restrict__ aabb,
    const uint32_t N,
    const float min_near,
    PackedVar32<float,1> nears, PackedVar32<float,1> fars
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
//    rays_o += n * 3;
//    rays_d += n * 3;

    const float ox = rays_o[n][0], oy = rays_o[n][1], oz = rays_o[n][2];
    const float dx = rays_d[n][0], dy = rays_d[n][1], dz = rays_d[n][2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near = (aabb[0] - ox) * rdx;
    float far = (aabb[3] - ox) * rdx;
    if (near > far) swapf(near, far);

    float near_y = (aabb[1] - oy) * rdy;
    float far_y = (aabb[4] - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);

    if (near > far_y || near_y > far) {
        nears[n] = fars[n] = FLT_MAX;//std::numeric_limits<float>::max();
        return;
    }

    if (near_y > near) near = near_y;
    if (far_y < far) far = far_y;

    float near_z = (aabb[2] - oz) * rdz;
    float far_z = (aabb[5] - oz) * rdz;
    if (near_z > far_z) swapf(near_z, far_z);

    if (near > far_z || near_z > far) {
        nears[n] = fars[n] = FLT_MAX;//std::numeric_limits<float>::max();
        return;
    }

    if (near_z > near) near = near_z;
    if (far_z < far) far = far_z;

    if (near < min_near) near = min_near;

    nears[n] = near;
    fars[n] = far;
}


// coords: int32, [N, 3]
// indices: int32, [N]
__global__ void kernel_morton3D(
    const PackedVar32<int32_t, 2> coords,
    const uint32_t N,
    PackedVar32<int32_t, 1> indices
) {
    // parallelkernel_morton3D
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;
    // locate
//    coords += n * 3;
    indices[n] = __morton3D(coords[n][0], coords[n][1], coords[n][2]);
}


// grid: float, [C, H, H, H]
// N: int, C * H * H * H / 8
// density_thresh: float
// bitfield: uint8, [N]
__global__ void kernel_packbits(
    const PackedVar32<float,2> __restrict__ grid,
    const uint32_t N,
    const float density_thresh,
    PackedVar32<uint8_t,1> bitfield
) {
    // parallel per byte
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
//    grid += n * 8;

    uint8_t bits = 0;
    const float* grid_ptr = &grid[0][0];

    #pragma unroll
    for (uint8_t i = 0; i < 8; i++) {
        bits |= (grid_ptr[n*8+i] > density_thresh) ? ((uint8_t)1 << i) : 0;
    }

    bitfield[n] = (uint8_t)bits;
}


// rays_o/d: [N, 3]
// grid: [CHHH / 8]
// xyzs, dirs, deltas: [M, 3], [M, 3], [M, 2]
// dirs: [M, 3]
// rays: [N, 3], idx, offset, num_steps
__global__ void kernel_march_rays_train(
    const PackedVar32<float,2> __restrict__ rays_o,
    const PackedVar32<float,2> __restrict__ rays_d,
    const PackedVar32<uint8_t,1> __restrict__ grid,
    const float bound,
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M,
    PackedVar32<float, 1> __restrict__ nears,
    PackedVar32<float, 1> __restrict__ fars,
    PackedVar32<float, 2> xyzs, PackedVar32<float, 2> dirs, PackedVar32<float, 2> deltas,
    PackedVar32<int32_t, 2> rays,
    PackedVar32<int32_t, 1> counter,
    const PackedVar32<float,1> __restrict__ noises
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
//    rays_o += n * 3;
//    rays_d += n * 3;

    // ray marching
    const float ox = rays_o[n][0], oy = rays_o[n][1], oz = rays_o[n][2];
    const float dx = rays_d[n][0], dy = rays_d[n][1], dz = rays_d[n][2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;

    const float near = nears[n];
    const float far = fars[n];
    const float noise = noises[n];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;

    float t0 = near;

    // perturb
    t0 += clamp(t0 * dt_gamma, dt_min, dt_max) * noise;

    // first pass: estimation of num_steps
    float t = t0;
    uint32_t num_steps = 0;

    //if (t < far) printf("valid ray %d t=%f near=%f far=%f \n", n, t, near, far);

    while (t < far && num_steps < max_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;

        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        //if (n == 0) printf("t=%f density=%f vs thresh=%f step=%d\n", t, density, density_thresh, num_steps);

        if (occ) {
            num_steps++;
            t += dt;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;

            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do {
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }

    //printf("[n=%d] num_steps=%d, near=%f, far=%f, dt=%f, max_steps=%f\n", n, num_steps, near, far, dt_min, (far - near) / dt_min);

    // second pass: really locate and write points & dirs
    int32_t* counter_ptr = &counter[0];
    uint32_t point_index = atomicAdd(counter_ptr, num_steps);
    uint32_t ray_index = atomicAdd(counter_ptr + 1, 1);

    //printf("[n=%d] num_steps=%d, point_index=%d, ray_index=%d\n", n, num_steps, point_index, ray_index);

    // write rays
    rays[ray_index][0] = n;
    rays[ray_index][1] = point_index;
    rays[ray_index][2] = num_steps;

    if (num_steps == 0) return;
    if (point_index + num_steps > M) return;

    float* xyzs_ptr = &xyzs[0][0];
    xyzs_ptr += point_index * 3;
    float* dirs_ptr = &dirs[0][0];
    dirs_ptr += point_index * 3;
    float* deltas_ptr = &deltas[0][0];
    deltas_ptr += point_index * 2;

    t = t0;
    uint32_t step = 0;

    float last_t = t;

    while (t < far && step < num_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;

        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        // query grid
        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs_ptr[0] = x;
            xyzs_ptr[1] = y;
            xyzs_ptr[2] = z;
            dirs_ptr[0] = dx;
            dirs_ptr[1] = dy;
            dirs_ptr[2] = dz;
            t += dt;
            deltas_ptr[0] = dt;
            deltas_ptr[1] = t - last_t; // used to calc depth
            last_t = t;
            xyzs_ptr += 3;
            dirs_ptr += 3;
            deltas_ptr += 2;
            step++;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do {
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }
}

// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N], final pixel alpha
// depth: [N,]
// image: [N, 3]
__global__ void kernel_composite_rays_train_forward(
    const PackedVar32<float,1> __restrict__ sigmas,
    const PackedVar32<float,2> __restrict__ rgbs,
    const PackedVar32<float,2> __restrict__ deltas,
    const PackedVar32<int32_t,2> __restrict__ rays,
    const uint32_t M, const uint32_t N, const float T_thresh,
    PackedVar32<float,1> weights_sum,
    PackedVar32<float,1> depth,
    PackedVar32<float,2> image
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    uint32_t index = rays[n][0];
    uint32_t offset = rays[n][1];
    uint32_t num_steps = rays[n][2];

    // empty ray, or ray that exceed max step count.
    if (num_steps == 0 || offset + num_steps > M) {
        weights_sum[index] = 0;
        depth[index] = 0;
        image[index][0] = 0;
        image[index][1] = 0;
        image[index][2] = 0;
        return;
    }

    const float* sigmas_ptr = &sigmas[0];
    sigmas_ptr += offset;
    const float* rgbs_ptr = &rgbs[0][0];
    rgbs_ptr += offset * 3;
    const float* deltas_ptr = &deltas[0][0];
    deltas_ptr += offset * 2;

    // accumulate
    uint32_t step = 0;

    float T = 1.0f;
    float r = 0, g = 0, b = 0, ws = 0, t = 0, d = 0;

    while (step < num_steps) {

        const float alpha = 1.0f - __expf(- sigmas_ptr[0] * deltas_ptr[0]);
        const float weight = alpha * T;

        r += weight * rgbs_ptr[0];
        g += weight * rgbs_ptr[1];
        b += weight * rgbs_ptr[2];

        t += deltas_ptr[1]; // real delta
        d += weight * t;

        ws += weight;

        T *= 1.0f - alpha;

        // minimal remained transmittence
        if (T < T_thresh) break;

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // locate
        sigmas_ptr++;
        rgbs_ptr += 3;
        deltas_ptr += 2;

        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // write
    weights_sum[index] = ws; // weights_sum
    depth[index] = d;
    image[index][0] = r;
    image[index][1] = g;
    image[index][2] = b;
}

// grad_weights_sum: [N,]
// grad: [N, 3]
// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N,], weights_sum here
// image: [N, 3]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
__global__ void kernel_composite_rays_train_backward(
    const PackedVar32<float,1> __restrict__ grad_weights_sum,
    const PackedVar32<float,2> __restrict__ grad_image,
    const PackedVar32<float,1> __restrict__ sigmas,
    const PackedVar32<float,2> __restrict__ rgbs,
    const PackedVar32<float,2> __restrict__ deltas,
    const PackedVar32<int32_t,2> __restrict__ rays,
    const PackedVar32<float,1> __restrict__ weights_sum,
    const PackedVar32<float,2> __restrict__ image,
    const uint32_t M, const uint32_t N, const float T_thresh,
    PackedVar32<float,1> grad_sigmas,
    PackedVar32<float,2> grad_rgbs
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    uint32_t index = rays[n][0];
    uint32_t offset = rays[n][1];
    uint32_t num_steps = rays[n][2];

    if (num_steps == 0 || offset + num_steps > M) return;

    const float* grad_weights_sum_ptr = &grad_weights_sum[0];
    grad_weights_sum_ptr += index;
    const float* grad_image_ptr = &grad_image[0][0];
    grad_image_ptr += index * 3;
    const float* weights_sum_ptr = &weights_sum[0];
    weights_sum_ptr += index;
    const float* image_ptr = &image[0][0];
    image_ptr += index * 3;
    const float* sigmas_ptr = &sigmas[0];
    sigmas_ptr += offset;
    const float* rgbs_ptr = &rgbs[0][0];
    rgbs_ptr += offset * 3;
    const float* deltas_ptr = &deltas[0][0];
    deltas_ptr += offset * 2;
    float* grad_sigmas_ptr = &grad_sigmas[0];
    grad_sigmas_ptr += offset;
    float* grad_rgbs_ptr = &grad_rgbs[0][0];
    grad_rgbs_ptr += offset * 3;

    // accumulate
    uint32_t step = 0;

    float T = 1.0f;
    const float r_final = image_ptr[0], g_final = image_ptr[1], b_final = image_ptr[2], ws_final = weights_sum_ptr[0];
    float r = 0, g = 0, b = 0, ws = 0;

    while (step < num_steps) {

        //TODO: deltas_ptr[1] unused?
        const float alpha = 1.0f - __expf(- sigmas_ptr[0] * deltas_ptr[0]);
        const float weight = alpha * T;

        r += weight * rgbs_ptr[0];
        g += weight * rgbs_ptr[1];
        b += weight * rgbs_ptr[2];
        ws += weight;

        T *= 1.0f - alpha;

        // check https://note.kiui.moe/others/nerf_gradient/ for the gradient calculation.
        // write grad_rgbs
        grad_rgbs_ptr[0] = grad_image_ptr[0] * weight;
        grad_rgbs_ptr[1] = grad_image_ptr[1] * weight;
        grad_rgbs_ptr[2] = grad_image_ptr[2] * weight;

        // write grad_sigmas
        grad_sigmas_ptr[0] = deltas_ptr[0] * (
            grad_image_ptr[0] * (T * rgbs_ptr[0] - (r_final - r)) +
            grad_image_ptr[1] * (T * rgbs_ptr[1] - (g_final - g)) +
            grad_image_ptr[2] * (T * rgbs_ptr[2] - (b_final - b)) +
            grad_weights_sum_ptr[0] * (1 - ws_final)
        );

        //printf("[n=%d] num_steps=%d, T=%f, grad_sigmas=%f, r_final=%f, r=%f\n", n, step, T, grad_sigmas[0], r_final, r);
        // minimal remained transmittence
        if (T < T_thresh) break;

        // locate
        sigmas_ptr++;
        rgbs_ptr += 3;
        deltas_ptr += 2;
        grad_sigmas_ptr++;
        grad_rgbs_ptr += 3;

        step++;
    }
}

__global__ void kernel_march_rays(
    const uint32_t n_alive,
    const uint32_t n_step,
    const PackedVar32<int32_t,1> __restrict__ rays_alive,
    const PackedVar32<float,1> __restrict__ rays_t,
    const PackedVar32<float,2> __restrict__ rays_o,
    const PackedVar32<float,2> __restrict__ rays_d,
    const float bound,
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t C, const uint32_t H,
    const PackedVar32<uint8_t,1> __restrict__ grid,
    const PackedVar32<float,1> __restrict__ nears,
    const PackedVar32<float,1> __restrict__ fars,
    PackedVar32<float,2> xyzs, PackedVar32<float,2> dirs, PackedVar32<float,2> deltas,
    const PackedVar32<float,1> __restrict__ noises
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    const float noise = noises[n];

    // locate
    const float* rays_o_ptr = &rays_o[0][0];
    rays_o_ptr += index * 3;
    const float* rays_d_ptr = &rays_d[0][0];
    rays_d_ptr += index * 3;
    float* xyzs_ptr = &xyzs[0][0];
    xyzs_ptr += n * n_step * 3;
    float* dirs_ptr = &dirs[0][0];
    dirs_ptr += n * n_step * 3;
    float* deltas_ptr = &deltas[0][0];
    deltas_ptr += n * n_step * 2;

    const float ox = rays_o_ptr[0], oy = rays_o_ptr[1], oz = rays_o_ptr[2];
    const float dx = rays_d_ptr[0], dy = rays_d_ptr[1], dz = rays_d_ptr[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;

    float t = rays_t[index]; // current ray's t
    const float near = nears[index], far = fars[index];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;

    // march for n_step steps, record points
    uint32_t step = 0;

    // introduce some randomness
    t += clamp(t * dt_gamma, dt_min, dt_max) * noise;

    float last_t = t;

    while (t < far && step < n_step) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1, level), bound);
        const float mip_rbound = 1 / mip_bound;

        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs_ptr[0] = x;
            xyzs_ptr[1] = y;
            xyzs_ptr[2] = z;
            dirs_ptr[0] = dx;
            dirs_ptr[1] = dy;
            dirs_ptr[2] = dz;
            // calc dt
            t += dt;
            deltas_ptr[0] = dt;
            deltas_ptr[1] = t - last_t; // used to calc depth
            last_t = t;
            // step
            xyzs_ptr += 3;
            dirs_ptr += 3;
            deltas_ptr += 2;
            step++;

        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do {
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }
}

__global__ void kernel_composite_rays(
    const uint32_t n_alive,
    const uint32_t n_step,
    const float T_thresh,
    PackedVar32<int32_t,1> rays_alive,
    PackedVar32<float,1> rays_t,
    const PackedVar32<float,1> __restrict__ sigmas,
    const PackedVar32<float,2> __restrict__ rgbs,
    const PackedVar32<float,2> __restrict__ normals,
    const PackedVar32<float,2> __restrict__ deltas,
    PackedVar32<float,1> weights_sum, PackedVar32<float,1> depth,
    PackedVar32<float,2> image, PackedVar32<float,2> normal
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id

    // locate
    const float* sigmas_ptr = &sigmas[0];
    sigmas_ptr += n * n_step;
    const float* rgbs_ptr = &rgbs[0][0];
    rgbs_ptr += n * n_step * 3;
    const float* normals_ptr = &normals[0][0];
    normals_ptr += n * n_step * 3;
    const float* deltas_ptr = &deltas[0][0];
    deltas_ptr += n * n_step * 2;

    float* rays_t_ptr = &rays_t[0];
    rays_t_ptr += index;
    float* weights_sum_ptr = &weights_sum[0];
    weights_sum_ptr += index;
    float* depth_ptr = &depth[0];
    depth_ptr += index;
    float* image_ptr = &image[0][0];
    image_ptr += index * 3;
    float* normal_ptr = &normal[0][0];
    normal_ptr += index * 3;

    float t = rays_t_ptr[0];
    float d = depth_ptr[0], r = image_ptr[0], g = image_ptr[1], b = image_ptr[2], x_ = normal_ptr[0], y_ = normal_ptr[1], z_ = normal_ptr[2], weight_sum = weights_sum_ptr[0];

    // accumulate
    uint32_t step = 0;
    while (step < n_step) {

        // ray is terminated if t == 0
        if (deltas_ptr[0] == 0) break;

        const float alpha = 1.0f - __expf(- sigmas_ptr[0] * deltas_ptr[0]);

        /*
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        -->
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const float T = 1 - weight_sum;
        const float weight = alpha * T;
        weight_sum += weight;

        t += deltas_ptr[1];
        d += weight * t; // real depth
        r += weight * rgbs_ptr[0];
        g += weight * rgbs_ptr[1];
        b += weight * rgbs_ptr[2];
        x_ += weight * normals_ptr[0];
        y_ += weight * normals_ptr[1];
        z_ += weight * normals_ptr[2];

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // ray is terminated if T is too small
        // use a larger bound to further accelerate inference
        if (T < T_thresh) break;

        // locate
        sigmas_ptr++;
        rgbs_ptr += 3;
        normals_ptr += 3;
        deltas_ptr += 2;
        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // rays_alive = -1 means ray is terminated early.
    if (step < n_step) {
        rays_alive[n] = -1;
    } else {
        rays_t_ptr[0] = t;
    }

    weights_sum_ptr[0] = weight_sum; // this is the thing I needed!
    depth_ptr[0] = d;
    image_ptr[0] = r;
    image_ptr[1] = g;
    image_ptr[2] = b;
    normal_ptr[0] = x_;
    normal_ptr[1] = y_;
    normal_ptr[2] = z_;
}

template <typename scalar_t>
__global__ void kernel_composite_sdf_rays(
    const uint32_t n_alive,
    const uint32_t n_step,
    const float T_thresh,
    int* rays_alive,
    scalar_t* rays_t,
    const scalar_t* __restrict__ sigmas,
    const scalar_t* __restrict__ rgbs,
    const scalar_t* __restrict__ deltas,
    scalar_t* weights_sum, scalar_t* depth, scalar_t* image
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id

    // locate
    sigmas += n * n_step;
    rgbs += n * n_step * 3;
    deltas += n * n_step * 2;

    rays_t += index;
    weights_sum += index;
    depth += index;
    image += index * 3;

    scalar_t t = rays_t[0]; // current ray's t

    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];
    scalar_t r = image[0];
    scalar_t g = image[1];
    scalar_t b = image[2];

    // accumulate
    uint32_t step = 0;
    while (step < n_step) {

        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;

        const scalar_t alpha = sigmas[0]; //1.0f - __expf(- sigmas[0] * deltas[0]);

        /*
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        -->
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        t += deltas[1]; // real delta
        d += weight * t;
        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // ray is terminated if T is too small
        // use a larger bound to further accelerate inference
        if (T < T_thresh) break;

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // rays_alive = -1 means ray is terminated early.
    if (step < n_step) {
        rays_alive[n] = -1;
    } else {
        rays_t[0] = t;
    }

    weights_sum[0] = weight_sum; // this is the thing I needed!
    depth[0] = d;
    image[0] = r;
    image[1] = g;
    image[2] = b;
}