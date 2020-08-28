from setuptools import setup, find_packages

setup(
    name='laserbeam',
    version='1.0.0',
    url='https://github.com/bdortamarques/laserbeam',
    author='Bruno Dorta Marques',
    author_email='brunodortamarques@gmail.com',
    description='laser beam DL package',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)