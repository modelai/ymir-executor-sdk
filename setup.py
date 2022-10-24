from setuptools import find_packages, setup

__version__ = '1.3.1.1024'

requirements = []
with open('requirements.txt') as f:
    for line in f.read().splitlines():
        requirements.append(line)

setup(
    name='ymir-exc',
    version=__version__,
    python_requires=">=3.6",
    install_requires=requirements,
    author_email="wang.jiaxin@intellif.com",
    description="ymir executor SDK: SDK for develop ymir training, mining and infer docker images",
    url="https://github.com/modelai/ymir-executor-sdk.git",
    packages=find_packages(exclude=["*tests*"]),
    include_package_data=True,
)
