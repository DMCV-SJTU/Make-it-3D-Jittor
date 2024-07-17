from typing import List, Optional, Tuple, Union

import numpy as np
import jittor as jt

from pointclouds import Pointclouds
from .utils import parse_image_size
from .global_header import proj_path

kMaxPointsPerBin = 22


def rasterize_points(
    pointclouds: Pointclouds,
    image_size: Union[int, List[int], Tuple[int, int]] = 256,
    radius: Union[float, List, Tuple, jt.Var] = 0.01,
    points_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_points_per_bin: Optional[int] = None,
):
    """
    Each pointcloud is rasterized onto a separate image of shape
    (H, W) if `image_size` is a tuple or (image_size, image_size) if it
    is an int.

    If the desired image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration. There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The camera can be used to set the pixel aspect ratio. In the rasterizer,
    we assume square pixels, but variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera aspect ratio to
    1.0 (i.e. square pixels) and only vary the
    `image_size` (i.e. the output image dimensions in pix

    Args:
        pointclouds: A Pointclouds object representing a batch of point clouds to be
            rasterized. This is a batch of N pointclouds, where each point cloud
            can have a different number of points; the coordinates of each point
            are (x, y, z). The coordinates are expected to
            be in normalized device coordinates (NDC): [-1, 1]^3 with the camera at
            (0, 0, 0); In the camera coordinate frame the x-axis goes from right-to-left,
            the y-axis goes from bottom-to-top, and the z-axis goes from back-to-front.
        image_size: Size in pixels of the output image to be rasterized.
            Can optionally be a tuple of (H, W) in the case of non square images.
        radius (Optional): The radius (in NDC units) of the disk to
            be rasterized. This can either be a float in which case the same radius is used
            for each point, or a torch.Tensor of shape (N, P) giving a radius per point
            in the batch.
        points_per_pixel (Optional): We will keep track of this many points per
            pixel, returning the nearest points_per_pixel points along the z-axis
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts to
            set it heuristically based on the shape of the input. This should not
            affect the output, but can affect the speed of the forward pass.
        max_points_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maximum number of points allowed within each
            bin. This should not affect the output values, but can affect
            the memory usage in the forward pass.

    Returns:
        3-element tuple containing

        - **idx**: int32 Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the indices of the nearest points at each pixel, in ascending
          z-order. Concretely `idx[n, y, x, k] = p` means that `points[p]` is the kth
          closest point (along the z-direction) to pixel (y, x) - note that points
          represents the packed points of shape (P, 3).
          Pixels that are hit by fewer than points_per_pixel are padded with -1.
        - **zbuf**: Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the z-coordinates of the nearest points at each pixel, sorted in
          z-order. Concretely, if `idx[n, y, x, k] = p` then
          `zbuf[n, y, x, k] = points[n, p, 2]`. Pixels hit by fewer than
          points_per_pixel are padded with -1
        - **dists2**: Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the squared Euclidean distance (in NDC units) in the x/y plane
          for each point closest to the pixel. Concretely if `idx[n, y, x, k] = p`
          then `dists[n, y, x, k]` is the squared distance between the pixel (y, x)
          and the point `(points[n, p, 0], points[n, p, 1])`. Pixels hit with fewer
          than points_per_pixel are padded with -1.

        In the case that image_size is a tuple of (H, W) then the outputs
        will be of shape `(N, H, W, ...)`.
    """
    points_packed = pointclouds.points_packed()
    cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    num_points_per_cloud = pointclouds.num_points_per_cloud()

    radius = _format_radius(radius, pointclouds)

    # In the case that H != W use the max image size to set the bin_size
    # to accommodate the num bins constraint in the coarse rasterizer.
    # If the ratio of H:W is large this might cause issues as the smaller
    # dimension will have fewer bins.
    # TODO: consider a better way of setting the bin size.
    im_size = parse_image_size(image_size)
    max_image_size = max(*im_size)

    if bin_size is None:
        bin_size = int(2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))

    if bin_size != 0:
        # There is a limit on the number of points per bin in the cuda kernel.
        points_per_bin = 1 + (max_image_size - 1) // bin_size
        if points_per_bin >= kMaxPointsPerBin:
            raise ValueError(
                "bin_size too small, number of points per bin must be less than %d; got %d"
                % (kMaxPointsPerBin, points_per_bin)
            )

    if max_points_per_bin is None:
        max_points_per_bin = int(max(10000, pointclouds._P / 5))

    # Function.apply cannot take keyword args, so we handle defaults in this
    # wrapper and call apply with positional args only
    return _RasterizePoints.apply(
        points_packed,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        im_size,
        radius,
        points_per_pixel,
        bin_size,
        max_points_per_bin,
    )


def _format_radius(
    radius: Union[float, List, Tuple, jt.Var], pointclouds: Pointclouds
) -> jt.Var:
    """
    Format the radius as a torch tensor of shape (P_packed,)
    where P_packed is the total number of points in the
    batch (i.e. pointclouds.points_packed().shape[0]).

    This will enable support for a different size radius
    for each point in the batch.

    Args:
        radius: can be a float, List, Tuple or tensor of
            shape (N, P_padded) where P_padded is the
            maximum number of points for each pointcloud
            in the batch.

    Returns:
        radius: torch.Tensor of shape (P_packed)
    """
    N, P_padded = pointclouds._N, pointclouds._P
    points_packed = pointclouds.points_packed()
    P_packed = points_packed.shape[0]
    if isinstance(radius, (list, tuple)):
        radius = jt.array(radius).type_as(points_packed)
    if isinstance(radius, jt.Var):
        if N == 1 and radius.ndim == 1:
            radius = radius[None, ...]
        if radius.shape != (N, P_padded):
            msg = "radius must be of shape (N, P): got %s"
            raise ValueError(msg % (repr(radius.shape)))
        else:
            padded_to_packed_idx = pointclouds.padded_to_packed_idx()
            radius = radius.view(-1)[padded_to_packed_idx]
    elif isinstance(radius, float):
        radius = jt.full((P_packed,), val=radius).type_as(points_packed)
    else:
        msg = "radius must be a float, list, tuple or tensor; got %s"
        raise ValueError(msg % type(radius))
    return radius


class _RasterizePoints(jt.Function):
    # @staticmethod
    def execute(
        self,
        points,  # (P, 3)
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size: Union[List[int], Tuple[int, int]] = (256, 256),
        radius: Union[float, jt.Var] = 0.01,
        points_per_pixel: int = 8,
        bin_size: int = 0,
        max_points_per_bin: int = 0,
    ):        
        # TODO: (WXZ) Any C tool for create jt.Var? I split the kernel functions and trigger them independently.
        # Coarse 1: To get bboxes, should_skip
        
        # logger.debug(f"points: {points.numpy()}")
        # logger.debug(f"radius: {radius.numpy()}")
        # logger.debug(f"WXZ TEST: bin_size: {bin_size}")
        
        P = points.shape[0]
        bboxes = jt.empty((4, P), dtype="float32")
        should_skip = jt.empty(P, dtype="bool")
        bboxes, should_skip = jt.code(inputs=[points, radius],
            outputs=[bboxes, should_skip], cuda_header='#include "rasterize_coarse.h"', cuda_src=f'''
                @alias(points, in0)
                @alias(radius, in1)
                @alias(bboxes, out0)
                @alias(should_skip, out1)

                const size_t blocks = 128;
                const size_t threads = 256;
                PointBoundingBoxKernel<<<blocks, threads>>>(
                    PackedVar32<float,2>(points),
                    PackedVar32<float,1>(radius),
                    {P},
                    PackedVar32<float,2>(bboxes),
                    PackedVar32<bool,1>(should_skip)
                );
            ''')
        bboxes.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        
        # logger.debug(f"bboxes: {bboxes.numpy()}")
        # logger.debug(f"should_skip: {should_skip.numpy()}")
        
        # Coarse 2: To get bin_points
        N_c2 = num_points_per_cloud.shape[0]
        H_c2 = image_size[0]
        W_c2 = image_size[1]
        E_c2 = bboxes.shape[1]
        M_c2 = max_points_per_bin
        num_bins_y_c2 = 1 + (H_c2 - 1) // bin_size
        num_bins_x_c2 = 1 + (W_c2 - 1) // bin_size
        elems_per_bin = jt.zeros((N_c2, num_bins_y_c2, num_bins_x_c2), dtype = "int32")
        bin_points = jt.full((N_c2, num_bins_y_c2, num_bins_x_c2, M_c2), -1, dtype = "int32")
        bin_points, elems_per_bin = jt.code(inputs=[bboxes, should_skip, cloud_to_packed_first_idx, num_points_per_cloud],
            outputs=[bin_points, elems_per_bin], cuda_header='#include "rasterize_coarse.h"', cuda_src=f'''   
                @alias(bboxes, in0)
                @alias(should_skip, in1)
                @alias(cloud_to_packed_first_idx, in2)
                @alias(num_points_per_cloud, in3)
                @alias(bin_points, out0)
                @alias(elems_per_bin, out1)
                
                const int chunk_size = 512;
                const size_t shared_size = {num_bins_y_c2} * {num_bins_x_c2} * chunk_size / 8;
                const size_t blocks = 64;
                const size_t threads = 512;
                
                RasterizeCoarseCudaKernel<<<blocks, threads, shared_size>>>(
                    PackedVar32<float,2>(bboxes),
                    PackedVar32<bool,1>(should_skip),
                    PackedVar64<int64_t,1>(cloud_to_packed_first_idx),
                    PackedVar64<int64_t,1>(num_points_per_cloud),
                    {N_c2},
                    {E_c2},
                    {H_c2},
                    {W_c2},
                    {bin_size},
                    chunk_size,
                    {M_c2},
                    PackedVar32<int32_t,3>(elems_per_bin),
                    PackedVar32<int32_t,4>(bin_points)
                );
            ''')
        bin_points.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        
        # logger.debug(f"bin_points: {bin_points.numpy()}")
        # logger.debug(f"elems_per_bin: {elems_per_bin.numpy()}")
        
        # Fine: To get idx, zbuf, dists
        N_f = bin_points.shape[0]
        BH_f = bin_points.shape[1]
        BW_f = bin_points.shape[2]
        M_f = bin_points.shape[3]
        K_f = points_per_pixel
        H_f = image_size[0]
        W_f = image_size[1]
        idx = jt.full((N_f, H_f, W_f, K_f), -1, dtype="int32")
        zbuf = jt.full((N_f, H_f, W_f, K_f), -1, dtype="float32")
        dists = jt.full((N_f, H_f, W_f, K_f), -1, dtype="float32")
        zbuf.requires_grad = True
        dists.requires_grad = True
        idx, zbuf, dists = jt.code(inputs=[points, bin_points, radius],
            outputs=[idx, zbuf, dists], cuda_header='#include "rasterize_points.h"', cuda_src=f'''     
                @alias(points, in0)
                @alias(bin_points, in1)
                @alias(radius, in2)
                @alias(idx, out0)
                @alias(zbuf, out1)
                @alias(dists, out2)

                const size_t blocks = 1024;
                const size_t threads = 64;
                RasterizePointsFineCudaKernel<<<blocks, threads>>>(
                    PackedVar32<float,2>(points),
                    PackedVar32<int32_t,4>(bin_points),
                    PackedVar32<float,1>(radius),
                    {bin_size},
                    {N_f},
                    {BH_f},
                    {BW_f},
                    {M_f},
                    {H_f},
                    {W_f},
                    {K_f},
                    PackedVar32<int32_t,4>(idx),
                    PackedVar32<float,4>(zbuf),
                    PackedVar32<float,4>(dists)
                );
            ''')
        idx.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        
        # logger.debug(f"idx: {idx.numpy()}")
        # logger.debug(f"zbuf: {zbuf.numpy()}")
        # logger.debug(f"dists: {dists.numpy()}")
        # logger.debug(f"idx_max: {jt.max(idx).numpy()}")
        # logger.debug(f"zbuf_max: {jt.max(zbuf).numpy()}")
        # logger.debug(f"dists_max: {jt.max(dists).numpy()}")
        
        # TODO: (WXZ) In our project, bin_size>0? Full version will be converted later
        # if ({bin_size} == 0) {
        # // Use the naive per-pixel implementation
        # return RasterizePointsNaive(
        #     points,
        #     cloud_to_packed_first_idx,
        #     num_points_per_cloud,
        #     image_size,
        #     radius,
        #     {points_per_pixel});
        # } else {
        # }
        self.points = points
        self.idx = idx
        # ctx.save_for_backward(points, idx)
        # ctx.mark_non_differentiable(idx)
        return idx, zbuf, dists

    # @staticmethod
    def grad(self, grad_idx, grad_zbuf, grad_dists):
        grad_cloud_to_packed_first_idx = None
        grad_num_points_per_cloud = None
        grad_image_size = None
        grad_radius = None
        grad_points_per_pixel = None
        grad_bin_size = None
        grad_max_points_per_bin = None
        # points, idx = ctx.saved_tensors
        points = self.points
        idx = self.idx
        
        # args = (points, idx, grad_zbuf, grad_dists)
        # grad_points = _C.rasterize_points_backward(*args)
        P = points.shape[0]
        N = idx.shape[0]
        H = idx.shape[1]
        W = idx.shape[2]
        K = idx.shape[3]
        grad_points = jt.zeros_like(points)
        grad_points = jt.code(inputs=[points, idx, grad_zbuf, grad_dists],
                outputs=[grad_points], cuda_header='#include "rasterize_points.h"', cuda_src=f'''
                    @alias(points, in0)
                    @alias(idx, in1)
                    @alias(grad_zbuf, in2)
                    @alias(grad_dists, in3)
                    @alias(grad_points, out0)

                    const size_t blocks = 1024;
                    const size_t threads = 64;

                    RasterizePointsBackwardCudaKernel<<<blocks, threads>>>(
                        PackedVar32<float,2>(points),
                        PackedVar32<int32_t,4>(idx),
                        {N},
                        {P},
                        {H},
                        {W},
                        {K},
                        PackedVar32<float,4>(grad_zbuf),
                        PackedVar32<float,4>(grad_dists),
                        PackedVar32<float,2>(grad_points)
                    );
                ''')
        grad_points.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        
        grads = (
            grad_points,
            grad_cloud_to_packed_first_idx,
            grad_num_points_per_cloud,
            grad_image_size,
            grad_radius,
            grad_points_per_pixel,
            grad_bin_size,
            grad_max_points_per_bin,
        )
        return grads
