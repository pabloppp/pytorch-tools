from setuptools import setup, find_packages

setup(
    name='torchtools',
    packages=find_packages(),
    description='PyTorch useful tools',
    version='0.2.19',
    url='https://github.com/pabloppp/pytorch-tools',
    author='Pablo Pernías',
    author_email='pablo@pernias.com',
    keywords=['pip', 'pytorch', 'tools', 'RAdam', 'Lookahead', 'RALamb', 'quantization'],
    zip_safe=False,
    install_requires=[
        'torch~=1.6',
        'torchvision',
        'numpy==1.*',
        'ninja==1.*'
    ],
    package_data={
        'stylegan2.tools': ['torchtools/nn/stylegan2/*'],
    },
    include_package_data=True,
)
