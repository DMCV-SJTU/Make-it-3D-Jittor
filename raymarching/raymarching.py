import jittor as jt
from jittor import Function
import numpy as np
import time
from jittor import nn
from .global_header import proj_path
BACKEND = None

jt.flags.use_cuda = 1


class _near_far_from_aabb(Function):
    # @staticmethod
    # @custom_fwd(cast_inputs=jt.float32)
    def execute(self, rays_o, rays_d, aabb, min_near=0.2):
        ''' near_far_from_aabb, CUDA implementation
        Calculate rays' intersection time (near and far) with aabb
        Args:
            rays_o: float, [N, 3]
            rays_d: float, [N, 3]
            aabb: float, [6], (xmin, ymin, zmin, xmax, ymax, zmax)
            min_near: float, scalar
        Returns:
            nears: float, [N]
            fars: float, [N]
        '''
        rays_o = rays_o.cuda()
        rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # num rays
        # , device = rays_o.device
        nears = jt.empty(N, dtype=rays_o.dtype)
        fars = jt.empty(N, dtype=rays_o.dtype)

        # get_backend().near_far_from_aabb(rays_o, rays_d, aabb, N, min_near, nears, fars)
        nears, fars = jt.code(inputs=[rays_o, rays_d, aabb], outputs=[nears, fars],
            cuda_header='#include"raymarching.h"', cuda_src=f'''
            @alias(rays_o, in0)
            @alias(rays_d, in1)
            @alias(aabb, in2)
            @alias(nears, out0)
            @alias(fars, out1)
    
            static constexpr uint32_t N_THREAD = 128;
            static constexpr uint32_t N_in = {N};
    
            kernel_near_far_from_aabb << < div_round_up(N_in, N_THREAD), N_THREAD >> > (
                PackedVar32<float,2>(rays_o), PackedVar32<float,2>(rays_d), PackedVar32<float,1>(aabb),
                N_in, {min_near}, PackedVar32<float,1>(nears), PackedVar32<float,1>(fars)
            );
            ''')
        nears.compile_options = {
            f"FLAGS: -I{proj_path}": 1}

        # print(nears,fars)
        return nears, fars


near_far_from_aabb = _near_far_from_aabb.apply


class _morton3D(Function):
    def execute(self, coords):
        ''' morton3D, CUDA implementation
        Args:
            coords: [N, 3], int32, in [0, 128) (for some reason there is no uint32 tensor in torch...)
            TODO: check if the coord range is valid! (current 128 is safe)
        Returns:
            indices: [N], int32, in [0, 128^3)

        '''
        #if not coords.is_cuda: coords = coords.cuda()

        N = coords.shape[0]

        indices = jt.empty(N, dtype=jt.int32)

         # get_backend().morton3D(coords.int(), N, indices)
        (indices, ) = jt.code(inputs=[coords.int()],
                outputs=[indices],
                cuda_header='#include"raymarching.h"', cuda_src=f"""
                @alias(coords, in0)
                @alias(indices, out0)

                static constexpr uint32_t N_THREAD = 128;
                static constexpr uint32_t N_in = {N};

                kernel_morton3D<<<div_round_up(N_in, N_THREAD), N_THREAD>>>(
                    PackedVar32<int32_t,2>(coords), N_in, PackedVar32<int32_t,1>(indices)
                );

                """)
        indices.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        return indices

morton3D = _morton3D.apply


class _packbits(Function):
    # @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def execute(self, grid, thresh, bitfield=None):
        ''' packbits, CUDA implementation
        Pack up the density grid into a bit field to accelerate ray marching.
        Args:
            grid: float, [C, H * H * H], assume H % 2 == 0
            thresh: float, threshold
        Returns:
            bitfield: uint8, [C, H * H * H / 8]
        '''
        # if not grid.is_cuda:
        grid = grid.cuda()
        grid = grid.contiguous()

        C = grid.shape[0]
        H3 = grid.shape[1]
        N = C * H3 // 8

        if bitfield is None:
            bitfield = jt.empty(N, dtype=jt.uint8)

        # get_backend().packbits(grid, N, thresh, bitfield)

        (bitfield, ) = jt.code(inputs=[grid],
                outputs=[bitfield],
                cuda_header='#include"raymarching.h"', cuda_src=f"""
                @alias(grid, in0)
                @alias(bitfield, out0)

                static constexpr uint32_t N_THREAD = 128;
                static constexpr uint32_t N_in = {N};

                kernel_packbits<<<div_round_up(N_in, N_THREAD), N_THREAD>>>(
                    PackedVar32<float,2>(grid), N_in, {thresh}, PackedVar32<uint8_t,1>(bitfield)
                );

                """)
        bitfield.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        return bitfield


packbits = _packbits.apply


class _march_rays_train(Function):
    # @staticmethod
    # @custom_fwd(cast_inputs=jt.float32)
    def execute(self, rays_o, rays_d, bound, density_bitfield, C, H, nears, fars, step_counter=None, mean_count=-1,
                perturb=False, align=-1, force_all_rays=False, dt_gamma=0, max_steps=1024):
        ''' march rays to generate points (forward only)
        Args:
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            C: int
            H: int
            nears/fars: float, [N]
            step_counter: int32, (2), used to count the actual number of generated points.
            mean_count: int32, estimated mean steps to accelerate training. (but will randomly drop rays if the actual point count exceeded this threshold.)
            perturb: bool
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            force_all_rays: bool, ignore step_counter and mean_count, always calculate all rays. Useful if rendering the whole image, instead of some rays.
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [M, 3], all generated points' coords. (all rays concated, need to use `rays` to extract points belonging to each ray)
            dirs: float, [M, 3], all generated points' view dirs.
            deltas: float, [M, 2], all generated points' deltas. (first for RGB, second for Depth)
            rays: int32, [N, 3], all rays' (index, point_offset, point_count), e.g., xyzs[rays[i, 1]:rays[i, 2]] --> points belonging to rays[i, 0]
        '''

        # if not rays_o.is_cuda:
        rays_o = rays_o.cuda()
        # if not rays_d.is_cuda:
        rays_d = rays_d.cuda()
        # if not density_bitfield.is_cuda:
        density_bitfield = density_bitfield.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        density_bitfield = density_bitfield.contiguous()

        N = rays_o.shape[0]  # num rays
        M = N * max_steps  # init max points number in total

        # running average based on previous epoch (mimic `measured_batch_size_before_compaction` in instant-ngp)
        # It estimate the max points number to enable faster training, but will lead to random ignored rays if underestimated.
        if not force_all_rays and mean_count > 0:
            if align > 0:
                mean_count += align - mean_count % align
            M = mean_count

        xyzs = jt.zeros((M, 3), dtype=rays_o.dtype)
        dirs = jt.zeros((M, 3), dtype=rays_o.dtype)
        deltas = jt.zeros((M, 2), dtype=rays_o.dtype)
        rays = jt.empty((N, 3), dtype=jt.int32)  # id, offset, num_steps

        if step_counter is None:
            step_counter = jt.zeros(2, dtype=jt.int32)  # point counter, ray counter

        if perturb:
            noises = jt.rand(N, dtype=rays_o.dtype)
        else:
            noises = jt.zeros(N, dtype=rays_o.dtype)

        # get_backend().march_rays_train(rays_o, rays_d, density_bitfield, bound, dt_gamma, max_steps, N, C, H, M, nears,
        #                                fars, xyzs, dirs, deltas, rays, step_counter,
        #                                noises)  # m is the actually used points number
        xyzs, dirs, deltas, rays, step_counter = jt.code(inputs=[rays_o, rays_d, density_bitfield, nears, fars, noises],
                outputs=[xyzs, dirs, deltas, rays, step_counter],
                cuda_header='#include"raymarching.h"', cuda_src=f"""
                @alias(rays_o, in0)
                @alias(rays_d, in1)
                @alias(density_bitfield, in2)
                @alias(nears, in3)
                @alias(fars, in4)
                @alias(noises, in5)
                @alias(xyzs, out0)
                @alias(dirs, out1)
                @alias(deltas, out2)
                @alias(rays, out3)
                @alias(step_counter, out4)
                
                static constexpr uint32_t N_THREAD = 128;
                static constexpr uint32_t N_in = {N};

                kernel_march_rays_train << < div_round_up(N_in, N_THREAD), N_THREAD >> > (
                    PackedVar32<float,2>(rays_o), PackedVar32<float,2>(rays_d),
                    PackedVar32<uint8_t,1>(density_bitfield), {bound}, {dt_gamma}, {max_steps},
                    {N}, {C}, {H}, {M}, PackedVar32<float,1>(nears),
                    PackedVar32<float,1>(fars), PackedVar32<float,2>(xyzs), PackedVar32<float,2>(dirs),
                    PackedVar32<float,2>(deltas), PackedVar32<int32_t,2>(rays), 
                    PackedVar32<int32_t,1>(step_counter), PackedVar32<float,1>(noises)
                );
                """)
        xyzs.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        # print(step_counter, M)

        # only used at the first (few) epochs.
        if force_all_rays or mean_count <= 0:
            m = step_counter[0].item()  # D2H copy
            if align > 0:
                m += align - m % align
            xyzs = xyzs[:m]
            dirs = dirs[:m]
            deltas = deltas[:m]

            #jt.cuda.empty_cache()

        return xyzs, dirs, deltas, rays


march_rays_train = _march_rays_train.apply


class _composite_rays_train(Function):
    # @staticmethod
    # @custom_fwd(cast_inputs=jt.float32)
    def execute(self, sigmas, rgbs, deltas, rays, T_thresh=1e-4):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            sigmas: float, [M,]
            deltas: float, [M, 2]
            rays: int32, [N, 3]
        Returns:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N, ], the Depth
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        M = sigmas.shape[0]
        N = rays.shape[0]
        weights_sum = jt.empty(N, dtype=sigmas.dtype)
        depth = jt.empty(N, dtype=sigmas.dtype)
        image = jt.empty((N, 3), dtype=sigmas.dtype).float32()
        weights_sum.requires_grad=True
        depth.requires_grad = True
        image.requires_grad=True
        # get_backend().composite_rays_train_forward(sigmas, rgbs, deltas, rays, M, N, T_thresh, weights_sum, depth,
        #                                            image)
        weights_sum, depth, image = jt.code(inputs=[sigmas, rgbs, deltas, rays],
                outputs=[weights_sum, depth, image],
                cuda_header='#include"raymarching.h"', cuda_src=f"""
                @alias(sigmas, in0)
                @alias(rgbs, in1)
                @alias(deltas, in2)
                @alias(rays, in3)
                @alias(weights_sum, out0)
                @alias(depth, out1)
                @alias(image, out2)

                static constexpr uint32_t N_THREAD = 128;
                static constexpr uint32_t N_in = {N};

                kernel_composite_rays_train_forward<<<div_round_up(N_in, N_THREAD), N_THREAD>>>(
                    PackedVar32<float,1>(sigmas), PackedVar32<float,2>(rgbs), PackedVar32<float,2>(deltas),
                    PackedVar32<int32_t,2>(rays), {M}, {N}, {T_thresh}, PackedVar32<float,1>(weights_sum),
                    PackedVar32<float,1>(depth), PackedVar32<float,2>(image)
                );

                """)
        weights_sum.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        # self.save_for_backward(sigmas, rgbs, deltas, rays, weights_sum, depth, image)
        self.sigmas = sigmas
        self.rgbs = rgbs
        self.deltas = deltas
        self.rays = rays
        self.weights_sum = weights_sum
        self.depth = depth
        self.image = image
        self.dims = [M, N, T_thresh]

        return weights_sum, depth, image

    # @staticmethod
    # @custom_bwd
    def grad(self, grad_weights_sum, grad_depth, grad_image):
        # print("WXZ TEST: GRAD FOR COM---------------")
        # NOTE: grad_depth is not used now! It won't be propagated to sigmas.
        grad_weights_sum = grad_weights_sum.contiguous()
        grad_image = grad_image.contiguous()

        # sigmas, rgbs, deltas, rays, weights_sum, depth, image = self.saved_tensors
        sigmas = self.sigmas
        rgbs = self.rgbs
        deltas = self.deltas
        rays = self.rays
        weights_sum = self.weights_sum
        depth = self.depth
        image = self.image
        M, N, T_thresh = self.dims

        grad_sigmas = jt.zeros_like(sigmas)
        grad_rgbs = jt.zeros_like(rgbs)

        # get_backend().composite_rays_train_backward(grad_weights_sum, grad_image, sigmas, rgbs, deltas, rays,
        #                                             weights_sum, image, M, N, T_thresh, grad_sigmas, grad_rgbs)
        # print("------before cuda------")
        # print("ws:", grad_sigmas)

        grad_sigmas, grad_rgbs = jt.code(inputs=[grad_weights_sum, grad_image, sigmas, rgbs, deltas, rays, weights_sum, image],
                outputs=[grad_sigmas, grad_rgbs],
                cuda_header='#include"raymarching.h"', cuda_src=f"""
                @alias(grad_weights_sum, in0)
                @alias(grad_image, in1)
                @alias(sigmas, in2)
                @alias(rgbs, in3)
                @alias(deltas, in4)
                @alias(rays, in5)
                @alias(weights_sum, in6)
                @alias(image, in7)
                @alias(grad_sigmas, out0)
                @alias(grad_rgbs, out1)

                static constexpr uint32_t N_THREAD = 128;
                static constexpr uint32_t N_in = {N};

                kernel_composite_rays_train_backward<<<div_round_up(N_in, N_THREAD), N_THREAD>>>(
                    PackedVar32<float,1>(grad_weights_sum), PackedVar32<float,2>(grad_image), PackedVar32<float,1>(sigmas),
                    PackedVar32<float,2>(rgbs), PackedVar32<float,2>(deltas), PackedVar32<int32_t,2>(rays),
                    PackedVar32<float,1>(weights_sum), PackedVar32<float,2>(image), {M}, {N}, {T_thresh},
                    PackedVar32<float,1>(grad_sigmas), PackedVar32<float,2>(grad_rgbs)
                );

                """)
        grad_sigmas.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        # print("------after cuda------")
        # print("ws:", grad_sigmas)
        return grad_sigmas, grad_rgbs, None, None, None
# class _composite_rays_train(Function):
#     def execute(self, sigmas, rgbs, deltas, rays, T_thresh=0.4):
#         return sigmas,sigmas,sigmas
#     def grad(self, grad1,grad2,grad3):
#         print("grad")
#         print(grad1,grad2,grad3)
#         return jt.Var(0.),jt.Var(0.),jt.Var(0.),jt.Var(0.),jt.Var(0.)
composite_rays_train = _composite_rays_train.apply


class _march_rays(Function):
    # @staticmethod
    # @custom_fwd(cast_inputs=jt.float32)
    def execute(self, n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, density_bitfield, C, H, near, far,
                align=-1, perturb=False, dt_gamma=0, max_steps=1024):
        ''' march rays to generate points (forward only, for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [N], the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
            rays_t: float, [N], the alive rays' time, we only use the first n_alive.
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            C: int
            H: int
            nears/fars: float, [N]
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            perturb: bool/int, int > 0 is used as the random seed.
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [n_alive * n_step, 3], all generated points' coords
            dirs: float, [n_alive * n_step, 3], all generated points' view dirs.
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        '''

        # if not rays_o.is_cuda:
        rays_o = rays_o.cuda()
        # if not rays_d.is_cuda:
        rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        M = n_alive * n_step

        if align > 0:
            M += align - (M % align)

        xyzs = jt.zeros((M, 3), dtype=rays_o.dtype)
        dirs = jt.zeros((M, 3), dtype=rays_o.dtype)
        deltas = jt.zeros((M, 2), dtype=rays_o.dtype)  # 2 vals, one for rgb, one for depth

        if perturb:
            # jt.manual_seed(perturb) # test_gui uses spp index as seed
            noises = jt.rand(n_alive, dtype=rays_o.dtype)
        else:
            noises = jt.zeros(n_alive, dtype=rays_o.dtype)

        # get_backend().march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, dt_gamma, max_steps, C, H,
        #                          density_bitfield, near, far, xyzs, dirs, deltas, noises)

        xyzs, dirs, deltas = jt.code(inputs=[rays_alive, rays_t, rays_o, rays_d,
                        density_bitfield, near, far, noises],
                outputs=[xyzs, dirs, deltas],
                cuda_header='#include"raymarching.h"', cuda_src=f"""
                @alias(rays_alive, in0)
                @alias(rays_t, in1)
                @alias(rays_o, in2)
                @alias(rays_d, in3)
                @alias(density_bitfield, in4)
                @alias(near, in5)
                @alias(far, in6)
                @alias(noises, in7)
                @alias(xyzs, out0)
                @alias(dirs, out1)
                @alias(deltas, out2)

                static constexpr uint32_t N_THREAD = 128;
                static constexpr uint32_t N_in = {n_alive};

                kernel_march_rays<<<div_round_up(N_in, N_THREAD), N_THREAD>>>(
                    {n_alive}, {n_step}, 
                    PackedVar32<int32_t,1>(rays_alive), PackedVar32<float,1>(rays_t), 
                    PackedVar32<float,2>(rays_o), PackedVar32<float,2>(rays_d), {bound}, {dt_gamma}, 
                    {max_steps}, {C}, {H}, PackedVar32<uint8_t,1>(density_bitfield), PackedVar32<float,1>(near), 
                    PackedVar32<float,1>(far), PackedVar32<float,2>(xyzs), PackedVar32<float,2>(dirs), 
                    PackedVar32<float,2>(deltas), PackedVar32<float,1>(noises)
                );

                """)
        xyzs.compile_options = {
            f"FLAGS: -I{proj_path}": 1}

        return xyzs, dirs, deltas


march_rays = _march_rays.apply


class _composite_rays(Function):
    # @staticmethod
    # @custom_fwd(cast_inputs=jt.float32)  # need to cast sigmas & rgbs to float
    def execute(self, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, normals, deltas, weights_sum, depth, image,
                normal, T_thresh=1e-2):
        ''' composite rays' rgbs, according to the ray marching formula. (for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [n_alive], the alive rays' IDs in N (N >= n_alive)
            rays_t: float, [N], the alive rays' time
            sigmas: float, [n_alive * n_step,]
            rgbs: float, [n_alive * n_step, 3]
            normals: float, [n_alive * n_step, 3]
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        In-place Outputs:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N,], the depth value
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
            normal: float, [N, 3], the normal value
        '''
        sigmas = sigmas.float().contiguous()
        rgbs = rgbs.float().contiguous()
        normals = normals.float().contiguous()

        # get_backend().composite_rays(n_alive, n_step, T_thresh, rays_alive, rays_t, sigmas, rgbs, normals, deltas,
        #                              weights_sum, depth, image, normal)
        # print("-----cuda in -----")
        # print("ra:", rays_alive)

        weights_sum, depth, image, normal = jt.code(inputs=[rays_alive, rays_t, sigmas, rgbs, normals, deltas],
                outputs=[weights_sum, depth, image, normal],
                cuda_header='#include"raymarching.h"', cuda_src=f"""
                @alias(rays_alive, in0)
                @alias(rays_t, in1)
                @alias(sigmas, in2)
                @alias(rgbs, in3)
                @alias(normals, in4)
                @alias(deltas, in5)
                @alias(weights_sum, out0)
                @alias(depth, out1)
                @alias(image, out2)
                @alias(normal, out3)

                static constexpr uint32_t N_THREAD = 128;
                static constexpr uint32_t N_in = {n_alive};

                kernel_composite_rays<<<div_round_up(N_in, N_THREAD), N_THREAD>>>(
                    {n_alive}, {n_step}, {T_thresh}, PackedVar32<int32_t,1>(rays_alive), PackedVar32<float,1>(rays_t), 
                    PackedVar32<float,1>(sigmas), PackedVar32<float,2>(rgbs), PackedVar32<float,2>(normals), 
                    PackedVar32<float,2>(deltas), PackedVar32<float,1>(weights_sum), PackedVar32<float,1>(depth), 
                    PackedVar32<float,2>(image), PackedVar32<float,2>(normal)
                );

                """)
        weights_sum.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        # print("-----cuda out -----")
        # print("ra:", rays_alive)

        return tuple()


composite_rays = _composite_rays.apply


class _composite_sdf_rays(Function):
    # @staticmethod
    # @custom_fwd(cast_inputs=jt.float32)  # need to cast sigmas & rgbs to float
    def execute(self, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image,
                T_thresh=1e-2):
        ''' composite rays' rgbs, according to the ray marching formula. (for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [n_alive], the alive rays' IDs in N (N >= n_alive)
            rays_t: float, [N], the alive rays' time
            sigmas: float, [n_alive * n_step,]
            rgbs: float, [n_alive * n_step, 3]
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        In-place Outputs:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N,], the depth value
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        # get_backend().composite_sdf_rays(n_alive, n_step, T_thresh, rays_alive, rays_t, sigmas, rgbs, deltas,
        #                                  weights_sum, depth, image)

        weights_sum, depth, image = jt.code(inputs=[rays_alive, rays_t, sigmas, rgbs, deltas],
                outputs=[weights_sum, depth, image],
                cuda_header='#include"raymarching.h"', cuda_src=f"""
                @alias(rays_alive, in0)
                @alias(rays_t, in1)
                @alias(sigmas, in2)
                @alias(rgbs, in3)
                @alias(deltas, in4)
                @alias(weights_sum, out0)
                @alias(depth, out1)
                @alias(image, out2)

                static constexpr uint32_t N_THREAD = 128;

                kernel_composite_sdf_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(
                    {n_alive}, {n_step}, {T_thresh}, rays_alive.data_ptr<int>(), 
                    rays_t.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), 
                    rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), 
                    weights_sum.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), 
                    image.data_ptr<scalar_t>()
                );

                """)
        weights_sum.compile_options = {
            f"FLAGS: -I{proj_path}": 1}

        return tuple()


composite_sdf_rays = _composite_sdf_rays.apply
