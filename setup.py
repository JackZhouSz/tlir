from setuptools import setup, find_packages

setup(
    name="tlir",
    version="0.1.0",
    packages=find_packages(),
    description="TLIR: Trainable Light and Image Rendering - A framework for differentiable volumetric rendering",
    python_requires=">=3.8",
    install_requires=[
        'numpy',
        'matplotlib',
        # Note: mitsuba and drjit should be installed separately via conda
    ],
    extras_require={
        'dev': [
            'pytest',
            'jupyter',
            'ipywidgets',
        ]
    },
)
