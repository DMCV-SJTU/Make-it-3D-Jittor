
#include <float.h>
#include <math.h>
#include <tuple>
#include "rasterization_utils.h"

#include "jt_helper.h"
#include "cuda_util.h"
#include "bitmask.h"

__global__ void PointBoundingBoxKernel(
    const PackedVar32<float,2> points, // (P, 3)
    const PackedVar32<float,1> radius, // (P,)
    const int P,
    PackedVar32<float,2> bboxes, // (4, P)
    PackedVar32<bool,1> skip_points) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;
    const float* points_ptr = &points[0][0];
//    float* bboxes_ptr = &bboxes[0][0];
    for (int p = tid; p < P; p += num_threads) {
        const float x = points_ptr[p * 3 + 0];
        const float y = points_ptr[p * 3 + 1];
        const float z = points_ptr[p * 3 + 2];
        const float r = radius[p];
        // TODO: change to kEpsilon to match triangles?
        const bool skip = z < 0;
        bboxes[0][p] = x - r;
        bboxes[1][p] = x + r;
        bboxes[2][p] = y - r;
        bboxes[3][p] = y + r;
        skip_points[p] = skip;
    }
}


__global__ void RasterizeCoarseCudaKernel(
    const PackedVar32<float,2> bboxes, // (4, E) (xmin, xmax, ymin, ymax)
    const PackedVar32<bool,1> should_skip, // (E,)
    const PackedVar64<int64_t,1> elem_first_idxs,
    const PackedVar64<int64_t,1> elems_per_batch,
    const int N,
    const int E,
    const int H,
    const int W,
    const int bin_size,
    const int chunk_size,
    const int max_elem_per_bin,
    PackedVar32<int32_t,3> elems_per_bin,
    PackedVar32<int32_t,4> bin_elems) {
    extern __shared__ char sbuf[];
    const int M = max_elem_per_bin;
    // Integer divide round up
    const int num_bins_x = 1 + (W - 1) / bin_size;
    const int num_bins_y = 1 + (H - 1) / bin_size;

    // NDC range depends on the ratio of W/H
    // The shorter side from (H, W) is given an NDC range of 2.0 and
    // the other side is scaled by the ratio of H:W.
    const float NDC_x_half_range = NonSquareNdcRange(W, H) / 2.0f;
    const float NDC_y_half_range = NonSquareNdcRange(H, W) / 2.0f;

    // Size of half a pixel in NDC units is the NDC half range
    // divided by the corresponding image dimension
    const float half_pix_x = NDC_x_half_range / W;
    const float half_pix_y = NDC_y_half_range / H;

    // This is a boolean array of shape (num_bins_y, num_bins_x, chunk_size)
    // stored in shared memory that will track whether each elem in the chunk
    // falls into each bin of the image.
    BitMask binmask((unsigned int*)sbuf, num_bins_y, num_bins_x, chunk_size);

    // Have each block handle a chunk of elements
    const int chunks_per_batch = 1 + (E - 1) / chunk_size;
    const int num_chunks = N * chunks_per_batch;

//    const float* bboxes_ptr = &bboxes[0][0];
    int32_t* elems_per_bin_ptr = &elems_per_bin[0][0][0];
    int32_t* bin_elems_ptr = &bin_elems[0][0][0][0];
    
    for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
      const int batch_idx = chunk / chunks_per_batch; // batch index
      const int chunk_idx = chunk % chunks_per_batch;
      const int elem_chunk_start_idx = chunk_idx * chunk_size;

      binmask.block_clear();
      const int64_t elem_start_idx = elem_first_idxs[batch_idx];
      const int64_t elem_stop_idx = elem_start_idx + elems_per_batch[batch_idx];

      // Have each thread handle a different face within the chunk
      for (int e = threadIdx.x; e < chunk_size; e += blockDim.x) {
        const int e_idx = elem_chunk_start_idx + e;

        // Check that we are still within the same element of the batch
        if (e_idx >= elem_stop_idx || e_idx < elem_start_idx) {
          continue;
        }

        if (should_skip[e_idx]) {
          continue;
        }
        const float xmin = bboxes[0][e_idx];
        const float xmax = bboxes[1][e_idx];
        const float ymin = bboxes[2][e_idx];
        const float ymax = bboxes[3][e_idx];

        // Brute-force search over all bins; TODO(T54294966) something smarter.
        for (int by = 0; by < num_bins_y; ++by) {
          // Y coordinate of the top and bottom of the bin.
          // PixToNdc gives the location of the center of each pixel, so we
          // need to add/subtract a half pixel to get the true extent of the bin.
          // Reverse ordering of Y axis so that +Y is upwards in the image.
          const float bin_y_min =
              PixToNonSquareNdc(by * bin_size, H, W) - half_pix_y;
          const float bin_y_max =
              PixToNonSquareNdc((by + 1) * bin_size - 1, H, W) + half_pix_y;
          const bool y_overlap = (ymin <= bin_y_max) && (bin_y_min < ymax);

          for (int bx = 0; bx < num_bins_x; ++bx) {
            // X coordinate of the left and right of the bin.
            // Reverse ordering of x axis so that +X is left.
            const float bin_x_max =
                PixToNonSquareNdc((bx + 1) * bin_size - 1, W, H) + half_pix_x;
            const float bin_x_min =
                PixToNonSquareNdc(bx * bin_size, W, H) - half_pix_x;

            const bool x_overlap = (xmin <= bin_x_max) && (bin_x_min < xmax);
            if (y_overlap && x_overlap) {
              binmask.set(by, bx, e);
            }
          }
        }
      }
      __syncthreads();
      // Now we have processed every elem in the current chunk. We need to
      // count the number of elems in each bin so we can write the indices
      // out to global memory. We have each thread handle a different bin.
      for (int byx = threadIdx.x; byx < num_bins_y * num_bins_x;
          byx += blockDim.x) {
        const int by = byx / num_bins_x;
        const int bx = byx % num_bins_x;
        const int count = binmask.count(by, bx);
        const int elems_per_bin_idx =
            batch_idx * num_bins_y * num_bins_x + by * num_bins_x + bx;

        // This atomically increments the (global) number of elems found
        // in the current bin, and gets the previous value of the counter;
        // this effectively allocates space in the bin_faces array for the
        // elems in the current chunk that fall into this bin.
        const int start = atomicAdd(elems_per_bin_ptr + elems_per_bin_idx, count);
        if (start + count > M) {
          // The number of elems in this bin is so big that they won't fit.
          // We print a warning using CUDA's printf. This may be invisible
          // to notebook users, but apparent to others. It would be nice to
          // also have a Python-friendly warning, but it is not obvious
          // how to do this without slowing down the normal case.
          const char* warning =
              "Bin size was too small in the coarse rasterization phase. "
              "This caused an overflow, meaning output may be incomplete. "
              "To solve, "
              "try increasing max_faces_per_bin / max_points_per_bin, "
              "decreasing bin_size, "
              "or setting bin_size to 0 to use the naive rasterization.";
          printf(warning);
          continue;
        }

        // Now loop over the binmask and write the active bits for this bin
        // out to bin_faces.
        int next_idx = batch_idx * num_bins_y * num_bins_x * M +
            by * num_bins_x * M + bx * M + start;
        for (int e = 0; e < chunk_size; ++e) {
          if (binmask.get(by, bx, e)) {
            // TODO(T54296346) find the correct method for handling errors in
            // CUDA. Throw an error if num_faces_per_bin > max_faces_per_bin.
            // Either decrease bin size or increase max_faces_per_bin
            bin_elems_ptr[next_idx] = elem_chunk_start_idx + e;
            next_idx++;
          }
        }
      }
      __syncthreads();
    }
}