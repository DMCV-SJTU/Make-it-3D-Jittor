/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "jt_helper.h"
#include "cuda_util.h"


#include <stdio.h>
#include <vector>

__constant__ const float kEpsilon = 1e-9;

template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
// result: []
// features:[C,P]
// alphas:[N, points_per_pixel, image_size,image_size]
// point_idx:[N, points_per_pixel, image_size, image_size]
__global__ void alphaCompositeCudaForwardKernel(
    // clang-format off
    const PackedVar32<float, 2> __restrict__ features,
    const PackedVar32<float, 4>  __restrict__ alphas,
    const PackedVar32<int64_t, 4>  __restrict__ points_idx,
    PackedVar32<float, 4> __restrict__ result // __restrict__是做什么的？
) {
  // clang-format on
  const int64_t batch_size = result.size(0);
  const int64_t C = features.size(0);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);

  // Get the batch and index
  const int batch = blockIdx.x;

  const int num_pixels = C * H * W;
  const int num_threads = gridDim.y * blockDim.x;
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;

  // Iterate over each feature in each pixel
  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    int ch = pid / (H * W);
    int j = (pid % (H * W)) / W;
    int i = (pid % (H * W)) % W;

    // alphacomposite the different values
    float cum_alpha = 1.;
    // Iterate through the closest K points for this pixel
    for (int k = 0; k < points_idx.size(1); ++k) {
      int n_idx = points_idx[batch][k][j][i];

      // Sentinel value is -1 indicating no point overlaps the pixel
      if (n_idx < 0) {
        continue;
      }

      float alpha = alphas[batch][k][j][i];
      // TODO(gkioxari) It might be more efficient to have threads write in a
      // local variable, and move atomicAdd outside of the loop such that
      // atomicAdd is executed once per thread.

      atomicAdd(
          &result[batch][ch][j][i], features[ch][n_idx] * cum_alpha * alpha);

      cum_alpha = cum_alpha * (1 - alpha);
    }
  }
}

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
__global__ void alphaCompositeCudaBackwardKernel(
    // clang-format off
    const PackedVar32<float, 4> __restrict__ grad_outputs,
    const PackedVar32<float, 2> __restrict__ features,
    const PackedVar32<float, 4> __restrict__  alphas,
    const PackedVar32<int64_t, 4> __restrict__  points_idx,
    PackedVar32<float, 2> __restrict__ grad_features,
    PackedVar32<float, 4> __restrict__  grad_alphas) {
  // clang-format on
  const int64_t batch_size = points_idx.size(0);
  const int64_t C = features.size(0);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);

  // Get the batch and index
  const int batch = blockIdx.x;

  const int num_pixels = C * H * W;
  const int num_threads = gridDim.y * blockDim.x;
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;

  // Parallelize over each feature in each pixel in images of size H * W,
  // for each image in the batch of size batch_size
  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    int ch = pid / (H * W);
    int j = (pid % (H * W)) / W;
    int i = (pid % (H * W)) % W;

    // alphacomposite the different values
    float cum_alpha = 1.;
    // Iterate through the closest K points for this pixel
    for (int k = 0; k < points_idx.size(1); ++k) {
      int n_idx = points_idx[batch][k][j][i];

      // Sentinel value is -1 indicating no point overlaps the pixel
      if (n_idx < 0) {
        continue;
      }
      float alpha = alphas[batch][k][j][i];

      // TODO(gkioxari) It might be more efficient to have threads write in a
      // local variable, and move atomicAdd outside of the loop such that
      // atomicAdd is executed once per thread.
      atomicAdd(
          &grad_alphas[batch][k][j][i],
          cum_alpha * features[ch][n_idx] * grad_outputs[batch][ch][j][i]);
      atomicAdd(
          &grad_features[ch][n_idx],
          cum_alpha * alpha * grad_outputs[batch][ch][j][i]);

      // Iterate over all (K-1) nearest points to update gradient
      for (int t = 0; t < k; ++t) {
        int t_idx = points_idx[batch][t][j][i];
        // Sentinel value is -1, indicating no point overlaps this pixel
        if (t_idx < 0) {
          continue;
        }
        float alpha_tvalue = alphas[batch][t][j][i];
        // TODO(gkioxari) It might be more efficient to have threads write in a
        // local variable, and move atomicAdd outside of the loop such that
        // atomicAdd is executed once per thread.
        atomicAdd(
            &grad_alphas[batch][t][j][i],
            -grad_outputs[batch][ch][j][i] * features[ch][n_idx] * cum_alpha *
                alpha / (1 - alpha_tvalue + kEpsilon));
      }

      cum_alpha = cum_alpha * (1 - alphas[batch][k][j][i]);
    }
  }
}
