from setuptools import setup, find_packages

setup(
        name='arcface-pytorch',
        version='0.1dev',
        packages=find_packages(),
        license='Creative Commons Attribution-Noncommercial-Share Alike license',
        install_requires=[
            'tqdm>=4.31',
            'numpy>=1.16',
            'visdom>=0.1',
            'torchvision>=0.4',
            'graphviz>=0.10',
            'matplotlib>=3',
            'Pillow>=6',
            'scikit_learn>=0',
            'PyYAML>=5',
        ]
)
