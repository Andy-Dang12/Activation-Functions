from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


CUDAExt_kwargs = {
    'extra_compile_args':{
        'cxx': [],
        'nvcc': ['--expt-extended-lambda']
    },
    'include_dirs':['external']
}

mish_extension = CUDAExtension(
    name ='activation/mish_cuda._C',
    sources = [
        'csrc/mish/mish_cuda.cpp',
        'csrc/mish/mish_cpu.cpp',
        'csrc/mish/mish_kernel.cu'
    ],
    **CUDAExt_kwargs
)

swish_extension = CUDAExtension(
    name='activation/swish._C',
    sources=[
        'csrc/swish/swish.cpp',
        'csrc/swish/swish.cu'
    ],
    **CUDAExt_kwargs
)

setup(
    name='activation',
    version='0.0.3',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    install_requires=['torch>=1.2'],
    ext_modules=[
        # mish_extension,
        swish_extension,
    ],
    cmdclass={
        'build_ext': BuildExtension
})
