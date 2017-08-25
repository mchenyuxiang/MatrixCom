import pycuda.autoinit
import pycuda.driver as cuda
import numpy

from pycuda.compiler import SourceModule

mod = SourceModule(""" 
__global__  void multiply_them(float *dest, float *a, float *b,float c) 
{ 
  const int i = threadIdx.x; 
  a[i] = a[i] * b[i]; 
} 
""")

def cuda_add():
    multiply_them = mod.get_function("multiply_them")

    a = numpy.random.randn(10).astype(numpy.float32)
    b = numpy.random.randn(10).astype(numpy.float32)
    print(a)

    dest = numpy.zeros_like(a)
    c = 2
    multiply_them(
        cuda.Out(dest), cuda.InOut(a), cuda.In(b), numpy.float64(c),
        block=(10, 1, 1), grid=(1, 1))
    return a

if __name__ == "__main__":
    dest = cuda_add()
    print(dest)
# print(dest-a*b)
