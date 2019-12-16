import setuptools
from IteratedLearning.version import version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'IteratedLearning',
    packages = setuptools.find_packages(),
    version = version,  # Ideally should be same as your GitHub release tag version
    description = 'Iterated Learning framework based on PyTorch',
    long_description = long_description,
    long_description_content_type="text/markdown",
    author = 'Shangmin Guo',
    author_email = 'sg955@cam.ac.uk',
    url = 'https://github.com/Shawn-Guo-CN/IteratedLearning',
    keywords = [
        'NLP',
        'Iterated Learning',
        'MARL'
    ],
    classifiers = [
        'Programming Language :: Python :: 3.7',
    ],
    license='GNU GENERAL PUBLIC LICENSE v3',
    python_requires='>=3.7',
    install_requires = [
        "torch==1.3.1",
        "torchvision==0.4.2",
    ],
)