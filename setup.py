import setuptools
from pathlib import Path
readme_path = Path(__file__).parent.joinpath("README.md")
long_description = readme_path.read_text()


EXTRAS_REQUIRE = {
    'fmu': ['FMPy>=0.3.6'],
    'scipy': ['scipy>=1.5.2'],
    'plot': ['matplotlib'],
    'clonemap': ['clonemapy @ git+https://github.com/RWTH-ACS/clonemapy@develop#egg=clonemapy'],
    'orjson': ['orjson>=3.9.5'],
    'fuzz': ['rapidfuzz>=1.7.0'],
    'mqtt': ['paho-mqtt>=1.6.1']
}
FULL_REQUIRES = []
for OPTIONAL_REQUIRES in EXTRAS_REQUIRE.values():
    FULL_REQUIRES.extend(OPTIONAL_REQUIRES)
EXTRAS_REQUIRE.update({'full': FULL_REQUIRES})

INSTALL_REQUIRES = [
    'numpy>=1.17.4',
    'pandas>=1.1.0',
    'simpy>=4.0.1',
    'pydantic>=2.0.0',
    'attrs>=22.2.0',
]

with open(Path(__file__).parent.joinpath("agentlib", "__init__.py"), "r") as file:
    for line in file.readlines():
        if line.startswith("__version__"):
            VERSION = line.replace("__version__", "").split("=")[1].strip().replace("'", "").replace('"', '')

setuptools.setup(
    name="agentlib",
    version=VERSION,
    long_description=long_description,
    author="Associates of the AGENT project",
    author_email="AGENT.Projekt@eonerc.rwth-aachen.de",
    description="Framework for development and execution "
                "of agents for control and simulation of "
                "energy systems.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    extras_require=EXTRAS_REQUIRE,
    install_requires=INSTALL_REQUIRES
)
