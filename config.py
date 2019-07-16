from setuptools import setup, find_packages

setup(
    name='toch_tools',
    packages=find_packages(),
    description='PyTorch useful tools',
    version='0.1.1',
    url='https://github.com/pabloppp/pytorch-tools',
    author='Pablo Pernías',
    author_email='pablo@pernias.com',
    keywords=['pip','pytorch','tools'],
	zip_safe=False,
	install_requires=[
		'torch==1.1.0',
		'torchvision==0.3.0',
		'numpy==1.16.4'
	]
)