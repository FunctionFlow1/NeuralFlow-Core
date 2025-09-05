from setuptools import setup, find_packages

setup(
    name='neuralflow-core',
    version='0.1.0',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    install_requires=[
        'numpy',
    ],
    # For C++ extension, you would typically use CMake or a custom build step
    # and link it here. For simplicity, we'll assume the C++ part is compiled
    # separately or integrated via a binding library like pybind11.
    # ext_modules=[CMakeExtension('neuralflow_core_cpp')],
    # cmdclass={'build_ext': CMakeBuild},
    author='Alexander J. Sterling',
    author_email='MarlonG1996@protonmail.com',
    description='A high-performance, lightweight deep learning framework with C++ backend and Python API.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/FunctionFlow1/NeuralFlow-Core',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
)
