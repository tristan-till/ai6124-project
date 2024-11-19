from setuptools import setup, find_packages

setup(
    name="ai6124_project",  # Replace with your package name
    version="0.1.0",  # Semantic versioning: major.minor.patch
    author="Goyal Hitesh & Tristan Till",  # Your name
    author_email="TR0001LL@e.ntu.edu.sg",  # Your email
    description="A project for AI6124 - GRU-LSTM-Attention AMO-GenFIS", 
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown",  
    url="https://github.com/tristan-till/ai6124-project.git",
    packages=find_packages(),
    install_requires=[
        "fuzzylab==0.13",
        "h5py==3.12.1",        
        "pillow==11.0.0",
        "PyWavelets==1.7.0",
        "scikit-learn==1.5.2",
        "ta==0.11.0",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "torchvision==0.20.1",
        "yfinance==0.2.48",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version
)
