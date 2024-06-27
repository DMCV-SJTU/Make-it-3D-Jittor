from alpha_composite import alpha_composite,_CompositeAlphaPoints
import jittor as jt

jt.flags.use_cuda = 1

feature = jt.rand(3,6)
alphas = jt.rand(4,32,800,800,dtype=jt.float32)
point_idx = jt.rand(4,32,800,800,dtype=jt.int32)
feature = feature.detach()
alphas = alphas.detach()
optimizer = jt.nn.SGD([feature, alphas], lr=1e-1)

for jj in range(100):
    optimizer.zero_grad()
    pt_cld = alpha_composite(feature, alphas, point_idx)
    loss = pt_cld.mean()
    print(loss)
    optimizer.backward(loss)
    optimizer.step()
    print(loss)


