from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='iou3d',
    ext_modules=[
        CUDAExtension('iou3d_cuda', [
            'src/iou3d.cpp',
            'src/iou3d_kernel.cu',
        ],
        extra_compile_args=['-D_GLIBCXX_USE_CXX11_ABI=1', '-std=c++11',{'nvcc': ['-O2']}])],
    cmdclass={'build_ext': BuildExtension}) 
