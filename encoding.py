import jittor as jt
from jittor import init
from jittor import nn

'''
import jittor as jt
from jittor import nn
class LinearModel(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(10, 2)
    def execute(self, x):
        x = self.linear(x)
        return x
net = LinearModel()
x = jt.random((10, 10))
out = net(x)
print(out.shape)
[10,2,]
'''


class FreqEncoder_jt(nn.Module):

    def __init__(self, input_dim, max_freq_log2, N_freqs, log_sampling=True, include_input=True, periodic_fns=(jt.sin, jt.cos)):
        super().__init__()
        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim
        self.output_dim += ((self.input_dim * N_freqs) * len(self.periodic_fns))
        if log_sampling:
            self.freq_bands = (2 ** jt.linspace(0, max_freq_log2, N_freqs))
        else:
            self.freq_bands = jt.linspace((2 ** 0), (2 ** max_freq_log2), N_freqs)

    def execute(self, input, **kwargs):
        out = []
        if self.include_input:
            out.append(input)
        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn((input * freq)))
        out = jt.contrib.concat(out, dim=(- 1))
        return out


def get_encoder(encoding, input_dim=3, multires=6, degree=4, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False, interpolation='linear', **kwargs):
    if (encoding == 'None'):
        return ((lambda x, **kwargs: x), input_dim)
    elif (encoding == 'frequency_jt'):
        encoder = FreqEncoder_jt(input_dim=input_dim, max_freq_log2=(multires - 1), N_freqs=multires, log_sampling=True)
    elif (encoding == 'frequency'):
        from freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)
    elif (encoding == 'sphere_harmonics'):
        from shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)
    elif (encoding == 'hashgrid'):
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners, interpolation=interpolation)
    elif (encoding == 'tiledgrid'):
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners)
    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')
    return (encoder, encoder.output_dim)
