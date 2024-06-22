#pragma once

#include <cstdio>
#include <tuple>
#include "rasterization_utils.h"

#include "jt_helper.h"
#include "cuda_util.h"

namespace {
// A little structure for holding details about a pixel.
struct Pix {
  float z; // Depth of the reference point.
  int32_t idx; // Index of the reference point.
  float dist2; // Euclidean distance square to the reference point.
};

__device__ inline bool operator<(const Pix& a, const Pix& b) {
  return a.z < b.z;
}

// This function checks if a pixel given by xy location pxy lies within the
// point with index p and batch index n. One of the inputs is a list (q)
// which contains Pixel structs with the indices of the points which intersect
// with this pixel sorted by closest z distance. If the pixel pxy lies in the
// point, the list (q) is updated and re-orderered in place. In addition
// the auxiliary variables q_size, q_max_z and q_max_idx are also modified.
// This code is shared between RasterizePointsNaiveCudaKernel and
// RasterizePointsFineCudaKernel.
template <typename PointQ>
__device__ void CheckPixelInsidePoint(
    const float* points, // (P, 3)
    const int p_idx,
    int& q_size,
    float& q_max_z,
    int& q_max_idx,
    PointQ& q,
    const float* radius,
    const float xf,
    const float yf,
    const int K) {
  const float px = points[p_idx * 3 + 0];
  const float py = points[p_idx * 3 + 1];
  const float pz = points[p_idx * 3 + 2];
  const float p_radius = radius[p_idx];
  const float radius2 = p_radius * p_radius;
  if (pz < 0)
    return; // Don't render points behind the camera
  const float dx = xf - px;
  const float dy = yf - py;
  const float dist2 = dx * dx + dy * dy;
  if (dist2 < radius2) {
    if (q_size < K) {
      // Just insert it
      q[q_size] = {pz, p_idx, dist2};
      if (pz > q_max_z) {
        q_max_z = pz;
        q_max_idx = q_size;
      }
      q_size++;
    } else if (pz < q_max_z) {
      // Overwrite the old max, and find the new max
      q[q_max_idx] = {pz, p_idx, dist2};
      q_max_z = pz;
      for (int i = 0; i < K; i++) {
        if (q[i].z > q_max_z) {
          q_max_z = q[i].z;
          q_max_idx = i;
        }
      }
    }
  }
}
}

__global__ void RasterizePointsFineCudaKernel(
    const PackedVar32<float,2> points, // (P, 3)
    const PackedVar32<int32_t,4> bin_points, // (N, BH, BW, T)
    const PackedVar32<float,1> radius,
    const int bin_size,
    const int N,
    const int BH, // num_bins y
    const int BW, // num_bins x
    const int M,
    const int H,
    const int W,
    const int K,
    PackedVar32<int32_t,4> point_idxs, // (N, H, W, K)
    PackedVar32<float,4> zbuf, // (N, H, W, K)
    PackedVar32<float,4> pix_dists) { // (N, H, W, K)
  // This can be more than H * W if H or W are not divisible by bin_size.
  const int num_pixels = N * BH * BW * bin_size * bin_size;
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const float* points_ptr = &points[0][0];
  const int32_t* bin_points_ptr = &bin_points[0][0][0][0];
  const float* radius_ptr = &radius[0];
  int32_t* point_idxs_ptr = &point_idxs[0][0][0][0];
  float* zbuf_ptr = &zbuf[0][0][0][0];
  float* pix_dists_ptr = &pix_dists[0][0][0][0];

  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    // Convert linear index into bin and pixel indices. We make the within
    // block pixel ids move the fastest, so that adjacent threads will fall
    // into the same bin; this should give them coalesced memory reads when
    // they read from points and bin_points.
    int i = pid;
    const int n = i / (BH * BW * bin_size * bin_size);
    i %= BH * BW * bin_size * bin_size;
    const int by = i / (BW * bin_size * bin_size);
    i %= BW * bin_size * bin_size;
    const int bx = i / (bin_size * bin_size);
    i %= bin_size * bin_size;

    const int yi = i / bin_size + by * bin_size;
    const int xi = i % bin_size + bx * bin_size;

    if (yi >= H || xi >= W)
      continue;

    const float xf = PixToNonSquareNdc(xi, W, H);
    const float yf = PixToNonSquareNdc(yi, H, W);

    // This part looks like the naive rasterization kernel, except we use
    // bin_points to only look at a subset of points already known to fall
    // in this bin. TODO abstract out this logic into some data structure
    // that is shared by both kernels?
    Pix q[kMaxPointsPerPixel];
    int q_size = 0;
    float q_max_z = -1000;
    int q_max_idx = -1;
    for (int m = 0; m < M; ++m) {
      const int p = bin_points_ptr[n * BH * BW * M + by * BW * M + bx * M + m];
      if (p < 0) {
        // bin_points uses -1 as a sentinal value
        continue;
      }
      CheckPixelInsidePoint(
          points_ptr, p, q_size, q_max_z, q_max_idx, q, radius_ptr, xf, yf, K);
    }
    // Now we've looked at all the points for this bin, so we can write
    // output for the current pixel.
    BubbleSort(q, q_size);

    // Reverse ordering of the X and Y axis as the camera coordinates
    // assume that +Y is pointing up and +X is pointing left.
    const int yidx = H - 1 - yi;
    const int xidx = W - 1 - xi;

    const int pix_idx = n * H * W * K + yidx * W * K + xidx * K;
    for (int k = 0; k < q_size; ++k) {
      point_idxs_ptr[pix_idx + k] = q[k].idx;
      zbuf_ptr[pix_idx + k] = q[k].z;
      pix_dists_ptr[pix_idx + k] = q[k].dist2;
    }
  }
}

__global__ void RasterizePointsBackwardCudaKernel(
    const PackedVar32<float,2> points, // (P, 3)
    const PackedVar32<int32_t,4> idxs, // (N, H, W, K)
    const int N,
    const int P,
    const int H,
    const int W,
    const int K,
    const PackedVar32<float,4> grad_zbuf, // (N, H, W, K)
    const PackedVar32<float,4> grad_dists, // (N, H, W, K)
    PackedVar32<float,2> grad_points) { // (P, 3)
  // Parallelized over each of K points per pixel, for each pixel in images of
  // size H * W, for each image in the batch of size N.
  int num_threads = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const float* points_ptr = &points[0][0];
  const int32_t* idxs_ptr = &idxs[0][0][0][0];
  const float* grad_zbuf_ptr = &grad_zbuf[0][0][0][0];
  const float* grad_dists_ptr = &grad_dists[0][0][0][0];
  float* grad_points_ptr = &grad_points[0][0];
  for (int i = tid; i < N * H * W * K; i += num_threads) {
    // const int n = i / (H * W * K); // batch index (not needed).
    const int yxk = i % (H * W * K);
    const int yi = yxk / (W * K);
    const int xk = yxk % (W * K);
    const int xi = xk / K;
    // k = xk % K (We don't actually need k, but this would be it.)
    // Reverse ordering of X and Y axes.
    const int yidx = H - 1 - yi;
    const int xidx = W - 1 - xi;

    const float xf = PixToNonSquareNdc(xidx, W, H);
    const float yf = PixToNonSquareNdc(yidx, H, W);

    const int p = idxs_ptr[i];
    if (p < 0)
      continue;
    const float grad_dist2 = grad_dists_ptr[i];
    const int p_ind = p * 3; // index into packed points tensor
    const float px = points_ptr[p_ind + 0];
    const float py = points_ptr[p_ind + 1];
    const float dx = px - xf;
    const float dy = py - yf;
    const float grad_px = 2.0f * grad_dist2 * dx;
    const float grad_py = 2.0f * grad_dist2 * dy;
    const float grad_pz = grad_zbuf_ptr[i];
    atomicAdd(grad_points_ptr + p_ind + 0, grad_px);
    atomicAdd(grad_points_ptr + p_ind + 1, grad_py);
    atomicAdd(grad_points_ptr + p_ind + 2, grad_pz);
  }
}