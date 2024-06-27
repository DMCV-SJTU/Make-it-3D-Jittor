import jittor as jt
from jittor import Function

from .global_header import proj_path

jt.flags.use_cuda = 1

class _CompositeAlphaPoints(Function):        
    def execute(self, points_idx, alphas, features):
        """_summary_
        features: Packed Tensor of shape (C, P) giving the features of each point.
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].
        """
        features = features.float()
        alphas = alphas.float()
        points_idx = points_idx.int64()
        
        N = points_idx.shape[0]  # num rays

        pt_cld = jt.zeros((N,features.shape[0],points_idx.shape[2],points_idx.shape[3]),
                          dtype='float')
        pt_cld.requires_grad = True

        (pt_cld,) = jt.code(inputs=[features, alphas, points_idx], outputs=[pt_cld],
            cuda_header='#include "alpha_composite.h"',cuda_src=f'''
            @alias(features, in0)
            @alias(alphas, in1)
            @alias(pointsidx, in2)
            @alias(pt_cld, out0)

            static constexpr int64_t batch_size = {N};

            const dim3 threadsPerBlock(64);
            const dim3 numBlocks(batch_size, 1024 / batch_size + 1);

            alphaCompositeCudaForwardKernel<<<numBlocks, threadsPerBlock>>>(
                PackedVar32<float,2>(features),
                PackedVar32<float,4>(alphas), 
                PackedVar32<int64_t,4>(pointsidx),
                PackedVar32<float,4>(pt_cld)
            );
            ''')
        pt_cld.compile_options = {
            f"FLAGS: -I{proj_path}": 1}

        # save for grad
        self.features = features
        self.alphas = alphas
        self.point_idx = points_idx
        
        return pt_cld


    def grad(self, grad_output):
        '''
        grad_feature:[]
        grad_alphas:[]
        '''
        features = self.features   #[19,1294080]
        alphas = self.alphas       #[1,8,800,800]
        points_idx = self.point_idx  #[1,8,800,800]
        N = points_idx.shape[0]  # num rays
        grad_features = jt.zeros_like(features)
        grad_alphas = jt.zeros_like(alphas)
        

        (grad_features, grad_alphas) = jt.code(
            inputs=[grad_output, features, alphas, points_idx],
            outputs=[grad_features, grad_alphas],  # syh: 首先,这个缺少")"应该不是头文件里面的问题,因为我们去掉后还有这个报错
            cuda_header='#include "alpha_composite.h"',
            cuda_src=f''' 
            @alias(grad_outputs, in0)
            @alias(features, in1)
            @alias(alphas, in2)
            @alias(pointsidx, in3)
            @alias(grad_features, out0)
            @alias(grad_alphas, out1)
            
            static constexpr int64_t batch_size = {N}; 
            
            const dim3 threadsPerBlock(64);
            const dim3 numBlocks(batch_size, 1024 / batch_size + 1);
            
            alphaCompositeCudaBackwardKernel<<<numBlocks, threadsPerBlock>>>(                
                PackedVar32<float,4>(grad_outputs),
                PackedVar32<float,2>(features),
                PackedVar32<float,4>(alphas), 
                PackedVar32<int64_t,4>(pointsidx),
                PackedVar32<float,2>(grad_features),
                PackedVar32<float,4>(grad_alphas)
            );
        '''
        )
        grad_features.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        
        #print("grad_features_shape", grad_features.shape)  #[19,1294080]
        #print("grad_alphas_shape", grad_alphas.shape)      #[1,8,800,800]
        
        return None, grad_alphas, grad_features


alpha_composite = _CompositeAlphaPoints.apply

