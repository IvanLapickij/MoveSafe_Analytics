# setup.py
from setuptools import setup, find_packages

setup(
    name="msa",
    version="1.0",
    description="Move Safe Analytics - Sports safety video analysis with pose and player detection",
    author="Ivan Lapickij",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'PyQt5==5.15.9',
        'matplotlib==3.8.0',
        'opencv-python==4.8.0.76',
        'ultralytics==8.0.20',
        'supervision==0.14.0',
        'numpy==1.24.3',
        'roboflow==1.0.6',
        'pytest==7.4.0'
    ],
    entry_points={
        'gui_scripts': [
            'msa = main:main',
        ]
    },
    python_requires='>=3.8',
)
