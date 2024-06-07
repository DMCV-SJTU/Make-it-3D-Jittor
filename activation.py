import jittor as jt
from jittor import Function

'''
import jittor as jt
from jittor import Function
class MyFunc(Function):
    def execute(self, x, y):
        self.x = x
        self.y = y
        return x*y, x/y
    def grad(self, grad0, grad1):
        return grad0 * self.y, grad1 * self.x

a = jt.array(3.0)
b = jt.array(4.0)
func = MyFunc.apply
c,d = func(a, b)
da, db = jt.grad(c+d*3, [a, b])
print(da,db)
jt.Var([4.], dtype=float32) jt.Var([9.], dtype=float32)
'''


class _trunc_exp(Function):
    def execute(self, x):
        self.x = x
        return jt.exp(x)

    def grad(self, g):
        x = self.x
        return g * jt.exp(jt.clamp(x,max_v=15))

trunc_exp = _trunc_exp.apply