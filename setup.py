from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="index_sve",
    version="0.1.0",
    packages=["index_sve"],
    ext_modules=[
        CppExtension(
            "index_sve._hpc_index",
            ["index_sve/op.cpp", "index_sve/normalize.cpp", "index_sve/linearize.cpp", "index_sve/gather.cpp"],
            extra_compile_args=["-march=armv8-a+sve2", "-msve-vector-bits=512", "-O3", "-std=c++17", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
