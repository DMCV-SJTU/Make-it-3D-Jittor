import os
import jittor as jt
from jittor import Function
import numpy as np
import sys
from jnerf.utils.common import enlarge
from jnerf.ops.code_ops.global_vars import global_headers, proj_options
from math import exp, log, log2, pow, ceil
jt.flags.use_cuda = 1


class GridEncode(Function):
    def div_round_up(self, val, divisor, type):
        if type == int:
            return int((val + divisor - 1) / divisor)
        assert (type == int)

    def __init__(self, hash_func_header, aabb_scale=1, n_pos_dims=3, n_features_per_level=2, n_levels=16,
                 base_resolution=16, log2_hashmap_size=19, n_rays_per_batch=2048, MAX_STEP=1024, using_fp16=False):
        self.hash_func_header = hash_func_header
        desired_resolution = 2048.0
        m_per_level_scale = np.exp2(np.log2(2048 * aabb_scale / base_resolution) / (n_levels - 1))
        # Ori: m_per_level_scale = exp(log(desired_resolution * aabb_scale / base_resolution) / (n_levels - 1)) -WXZ
        # M3D-ver: per_level_scale = np.exp2(np.log2(2048 * self.bound(which is 1 -by WXZ) / 16) / (16 - 1)) -WXZ
        n_features = n_features_per_level * n_levels
        m_n_levels = self.div_round_up(n_features, n_features_per_level, int)
        offsets_table_host = [0 for x in range(33)]
        offset = 0
        for i in range(m_n_levels):
            scale = pow(2, (i * log2(m_per_level_scale))) * base_resolution - 1.0
            resolution = ceil(scale) + 1
            params_in_level = int(resolution) ** int(n_pos_dims)
            params_in_level = self.div_round_up(params_in_level, 8, int) * 8
            params_in_level = min(
                params_in_level, (1 << log2_hashmap_size))
            offsets_table_host[i] = offset
            offset += params_in_level
        offsets_table_host[m_n_levels] = offset
        m_n_params = offsets_table_host[m_n_levels] * n_features_per_level
        self.m_n_params = m_n_params
        self.m_hashmap_offsets_table = jt.empty([n_levels + 1], 'int32')
        for i in range(m_n_levels + 1):
            self.m_hashmap_offsets_table[i] = offsets_table_host[i]
        self.N_POS_DIMS = n_pos_dims
        self.N_FEATURES_PER_LEVEL = n_features_per_level
        self.m_n_features = n_features
        self.m_n_padded_output_dims = n_features
        self.m_n_levels = n_levels
        self.m_base_resolution = base_resolution
        self.m_per_level_scale = m_per_level_scale
        self.m_quantize_threshold = 0.0  # I think =0.0 is useless since the abs in kernel -WXZ
        self.m_max_level = 1000.0
        self.m_n_output_dims = n_features
        self.m_interpolation_type = 1  # Linear in tcnn -WXZ
        self.m_grid_type = 0  # Hash in tcnn -WXZ
        self.n_rays_per_batch = n_rays_per_batch  # Useless because of num_elements -WXZ
        self.MAX_STEP = MAX_STEP  # Useless because of num_elements -WXZ
        self.num_elements = self.n_rays_per_batch * self.MAX_STEP
        # input.size(0) in tcnn, but useless? Since it gets a new value for every fwd&bwd. -WXZ
        self.m_positions = jt.empty([self.num_elements * n_pos_dims * 2], 'float')
        # TODO: Without Check!!!!! -WXZ
        self.grad_type = 'float32'
        if using_fp16:
            self.grad_type = 'float16'
        self.m_encoded_positions = jt.empty(
            [self.num_elements * n_features], self.grad_type)
        # Ori: [self.num_elements * n_features * 2] but what is the bonus *2 -WXZ
        # TCNN-ver: workspace = allocate_workspace(synced_streams.get(0), num_elements * m_n_features * sizeof(T)) -WXZ
        self.m_grid_gradient = jt.empty([m_n_params], self.grad_type)
        # TCNN-ver: grid_gradient_tmp = allocate_workspace(stream, m_n_params * sizeof(grad_t)); -WXZ
        self.m_stochastic_interpolation = 0  # False in tcnn -WXZ
        header_path = os.path.join(os.path.dirname(__file__), 'op_header')
        proj_options[f"FLAGS: -I{header_path}"] = 1

    def execute(self, x, m_grid, is_grad):
        self.num_elements = x.shape[0]
        assert (m_grid.dtype == self.grad_type)
        assert (self.m_encoded_positions.dtype == self.grad_type)

        output = jt.empty([self.num_elements, 32], self.grad_type)
        if is_grad:
            m_grid.requires_grad = True
            output.requires_grad = True
        else:
            output.requires_grad = False
        output, self.m_positions, self.m_encoded_positions = jt.code([self.m_hashmap_offsets_table, x, m_grid],
                                                                     [output, self.m_positions,
                                                                      self.m_encoded_positions],
                                                                     cuda_header=self.hash_func_header + '#include "HashEncode.h"',
                                                                     cuda_src=f"""
        #define grad_t in2_type
        @alias(m_positions, out1)
        @alias(m_encoded_positions, out2)
        @alias(hashmap_offsets_table,in0)
        cudaStream_t stream=0;
        const unsigned int num_elements=in1_shape0;
        if(num_elements==0){{
            return ;
        }}
        const int m_n_padded_output_dims={self.m_n_padded_output_dims};
		const int N_POS_DIMS={self.N_POS_DIMS};
        const int N_FEATURES_PER_LEVEL={self.N_FEATURES_PER_LEVEL};
        const int m_n_features={self.m_n_features};
        const dim3 threads = {{ 64, N_POS_DIMS, 1}};
		const uint32_t blocks = div_round_up(num_elements, threads.x);
        extract_position<float,N_POS_DIMS><<<blocks, threads, 0, stream>>>(
			num_elements,
		{{in1_p,in1_shape1}},
			m_positions_p
		);
        static constexpr uint32_t N_THREADS_HASHGRID = 512;
		const dim3 blocks_hashgrid = {{ div_round_up(num_elements, N_THREADS_HASHGRID), {self.m_n_levels}, 1 }};
        grad_t* m_grid=(grad_t*)in2_p;
        float* dy_dx=nullptr;
        grad_t* m_encoded_pos=(grad_t*)m_encoded_positions_p;
        const float bbb = {self.m_per_level_scale};
		kernel_grid<grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
			num_elements,
            m_n_features,
			(const uint32_t *)hashmap_offsets_table_p,
			{self.m_base_resolution},
			std::log2(bbb),
			{self.m_quantize_threshold},
			{self.m_max_level},
			{self.m_interpolation_type},
			{self.m_grid_type},
            m_grid,
			(float*)m_positions_p,
			m_encoded_pos,
			dy_dx
		);      

        const dim3 threads_transpose = {{ {self.m_n_levels} * N_FEATURES_PER_LEVEL, 8, 1 }};
		const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);

        PitchedPtr<grad_t> outputs{{ out0_p,out0_shape1 }};

	    transpose_encoded_position<grad_t><<<blocks_transpose, threads_transpose, 0, stream>>>(
			num_elements,
			m_encoded_positions_p,
			PitchedPtr<grad_t>{{outputs}}
		);	
""")
        output.compile_options = proj_options
        self.m_positions = self.m_positions.detach()
        self.m_encoded_positions = self.m_encoded_positions.detach()
        # print("output shape", output.shape)
        return output

    def grad(self, grad_x):
        # print("grad")
        self.m_grid_gradient, self.m_encoded_positions = \
            jt.code([self.m_positions, self.m_hashmap_offsets_table, grad_x],
                    [self.m_grid_gradient, self.m_encoded_positions],
                    cuda_header=self.hash_func_header + '#include"HashEncode.h"', cuda_src=f"""
        #define grad_t in2_type
        @alias(m_positions, in0)
        @alias(m_encoded_positions, out1)
        @alias(hashmap_offsets_table,in1)
        @alias(m_grid_gradient,out0)
        const unsigned int num_elements=in2_shape0;
        if(num_elements==0){{
            return ;
        }}
        const int m_n_padded_output_dims={self.m_n_padded_output_dims};
        const unsigned int N_FEATURES_PER_LEVEL={self.N_FEATURES_PER_LEVEL};                 
        cudaStream_t stream=0;
	    const dim3 threads_transpose ={{  {self.m_n_levels} * N_FEATURES_PER_LEVEL, 8, 1}};
        PitchedPtr<grad_t> dL_dy{{ in2_p,in2_shape1 }};
        cudaMemsetAsync(out0_p, 0, out0->size);    
        const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);  
        transpose_gradients<grad_t><<<blocks_transpose, threads_transpose, 0, stream>>>(
				num_elements,
				(grad_t *)m_encoded_positions_p,
				PitchedPtr<const grad_t>{{dL_dy}});
                
	    grad_t* grid_gradient=(grad_t*)m_grid_gradient_p;
        static constexpr uint32_t N_THREADS_HASHGRID = 256;
        static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u,N_FEATURES_PER_LEVEL);
        const dim3 blocks_hashgrid = {{div_round_up(num_elements *N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD, N_THREADS_HASHGRID), {self.m_n_levels}, 1}};
        const float bbb = {self.m_per_level_scale};
        kernel_grid_backward<grad_t, grad_t, {self.N_POS_DIMS}, N_FEATURES_PER_LEVEL, N_FEATURES_PER_THREAD><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
				num_elements,
				{self.m_n_features},
			    (const uint32_t *)hashmap_offsets_table_p,
			    {self.m_base_resolution},
				std::log2(bbb),
				{self.m_max_level},
				{self.m_stochastic_interpolation},
			    {self.m_interpolation_type},
		    	{self.m_grid_type},
				grid_gradient,
                (float*)m_positions_p,												 
				(const grad_t *)m_encoded_positions_p
			);
                  
        """)

        self.m_grid_gradient.compile_options = proj_options
        self.m_positions = self.m_positions.detach()
        self.m_encoded_positions = self.m_encoded_positions.detach()

        return None, self.m_grid_gradient, None