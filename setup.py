from setuptools import setup, find_packages

setup(
    name='torchtools',
    packages=find_packages(),
    description='PyTorch useful tools',
    version='0.1.9',
    url='https://github.com/pabloppp/pytorch-tools',
    author='Pablo Pernías',
    author_email='pablo@pernias.com',
    keywords=['pip', 'pytorch', 'tools', 'RAdam', 'quantization'],
    zip_safe=False,
    install_requires=[
        'torch==1.*',
        'torchvision',
        'numpy==1.*'
    ]
)
