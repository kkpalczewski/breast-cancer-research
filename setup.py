from setuptools import setup, find_packages

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession

def load_requirements(fname):
    reqs = parse_requirements(fname, session=PipSession())
    return [str(ir.req) for ir in reqs]

setup(
    name="unet_utils",
    version="0.1",
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt")
)