from setuptools import find_packages, setup

setup(
    name='scportrait_ot',
    version='1.0.0',
    author='Alessandro Palma, Sophia Mädler, Nikals Schmacke',
    url='',
    packages=['scportrait_ot/']+find_packages(),
    zip_safe=False,
    include_package_data=True
)