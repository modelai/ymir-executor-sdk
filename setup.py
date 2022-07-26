from setuptools import setup, find_packages


__version__ = '1.2.1.0706'

requirements = []
with open('requirements.txt') as f:
    for line in f.read().splitlines():
        requirements.append(line)

setup(
    name='ymir-exc',
    version=__version__,
    python_requires=">=2.7",
    install_requires=requirements,
    author_email="contact.viesc@gmail.com",
    description="ymir executor SDK: SDK for develop ymir training, mining and infer docker images",
    url="https://github.com/IndustryEssentials/ymir",
    packages=find_packages(exclude=["*tests*"]),
    include_package_data=True,
)