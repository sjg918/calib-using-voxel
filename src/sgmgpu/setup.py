from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='semi_global_matching',
    ext_modules=[
        CUDAExtension('semi_global_matching_cuda', [
            'src/semi_global_matching.cpp',
            'src/semi_global_matching_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
