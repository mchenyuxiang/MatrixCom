import pycuda.autoinit
import pycuda.driver as cuda
import numpy

from pycuda.compiler import SourceModule

mod = SourceModule(""" 
__global__  void multiply_them(float *dest, float *a, float *b) 
{ 
  const int i = threadIdx.x; 
  dest[i] = a[i] * b[i]; 
} 
""")

def cuda_add():
    multiply_them = mod.get_function("multiply_them")

    a = numpy.random.randn(400).astype(numpy.float32)
    b = numpy.random.randn(400).astype(numpy.float32)

    dest = numpy.zeros_like(a)
    multiply_them(
        cuda.Out(dest), cuda.In(a), cuda.In(b),
        block=(400, 1, 1), grid=(1, 1))
    return dest

if __name__ == "__main__":
    dest = cuda_add()
    print(dest)
# print(dest-a*b)
