from setuptools import setup, find_packages

setup(
    name="note-scribe",
    version="1.0.0",
    description="End-to-End Handwritten Notes to Searchable Digital Notes with OCR, Text Cleaning, and Extractive Summarization",
    author="Khushi Yadav",
    author_email="ky19tech@gmail.com",
    url="https://github.com/khushiyadav35/handwritten-notes-to-searchable-notes",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.35.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-image>=0.22.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.0",
        "pytest>=8.0.0",
    ],
)
