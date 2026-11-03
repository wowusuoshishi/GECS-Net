"""Setup script for GECS-Net."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gecs-net",
    version="1.0.0",
    author="Shuyuan Lin, Mengtin Lo, Haosheng Chen, Yanjie Liang, Qiangqiang Wu",
    description="GECS-Net: Two-View Correspondence Learning via Geometric Embedding and Cross-Stage Consensus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wowusuoshishi/GECS-Net",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6.0",
        "numpy>=2.0.0",
        "opencv-contrib-python>=4.11.0",
        "h5py>=3.13.0",
        "matplotlib>=3.10.0",
        "tensorboardX>=2.6.0",
        "tqdm>=4.67.0",
        "six>=1.17.0",
    ],
)
