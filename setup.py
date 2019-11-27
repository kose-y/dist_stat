from setuptools import setup

setup(name="dist_stat", 
    version="0.1",
    description="Statistical HPC applications using PyTorch",
    author="Seyoon Ko",
    author_email="syko0507@snu.ac.kr",
    url="https://github.com/kose-y/dist_stat",
    packages=['dist_stat'],
    install_requires=['numpy>=1.13','scipy>=1.0.0'],
    extras_require={
        'torch': ['torch>=0.4.0'],
    },
    keywords=['high-performance computing', 'multi-GPU', 'cloud computing', 'distributed computing'],
    license="MIT"
)
