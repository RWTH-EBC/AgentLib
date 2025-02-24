<img src="./docs/images/logos/agentlib_logo/logo.svg" alt="drawing" height="150"/>


[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Coverage](https://rwth-ebc.github.io/AgentLib/main/coverage/badge.svg)](https://rwth-ebc.github.io/AgentLib/main/coverage/)
[![pylint](https://rwth-ebc.github.io/AgentLib/main/pylint/pylint.svg)](https://rwth-ebc.github.io/AgentLib/main/pylint/pylint.html)
[![documentation](https://rwth-ebc.github.io/AgentLib/main/docs/doc.svg)](https://rwth-ebc.github.io/AgentLib/main/docs/index.html)


This is a collection of base classes for the implementation of agents in Python.
The original purpose of this library is the execution of distributed simulations and controllers for energy systems, however it is not limited to any specific field.
You can find the motivation, core principles and several exemplary applications in the associated [journal paper](https://doi.org/10.1016/j.apenergy.2025.125496) and in the [documentation](https://rwth-ebc.github.io/AgentLib/main/docs/index.html).

To get started, please check out our examples.

### Installation

To install the agentlib with minimal dependencies, run:

```
pip install agentlib
```

To install with full dependencies (recommended), run:
```
pip install agentlib[full]
```

If you want to work on the agentlib, first clone it, go its directory and then install it in editable mode:

```
pip install -e .[full]
```

## Optional Dependencies
AgentLib has a number of optional dependencies, ranging from additional features to performance improvements:
 
 - **fmu**: Support simulation of FMU models (https://fmi-standard.org/).
 - **scipy**: Support simulation of linear state space models, based on scipy.
 - **mqtt**: Support communication between agents through the mqtt protocol.
 - **plot**: Installs matplotlib, allows to plot the result of examples.
 - **orjson**: Faster json library, improves performance when using network communicators.
 - **fuzz**: Improves error messages when providing wrong configurations.

**clonemap**: Support the execution of agents and their communication through [clonemap](https://github.com/sogno-platform/clonemap). As clonemapy is not available through PYPI, please install it from source, or through the AgentLib's ``requirements.txt`` .

## Plugins
AgentLib supports extension, especially in the form of additional modules through plugins.
Official Plugins available are:
  - **[AgentLib_MPC](https://github.com/RWTH-EBC/AgentLib-MPC)**: Provides modules for model predictive control.
  - **[AgentLib_FIWARE](https://github.com/RWTH-EBC/AgentLib-FIWARE)**: Provides communicators for the IoT Platform FIWARE.

## Referencing the AgentLib

To cite the AgentLib, please use the following paper:

> Eser, Steffen and Storek, Thomas and Wüllhorst, Fabian and Dähling, Stefan and Gall, Jan and Stoffel, Phillip and Müller, Dirk, A Modular Python Framework for Rapid Development of Advanced Control Algorithms for Energy Systems. Available at SSRN: [https://doi.org/10.1016/j.apenergy.2025.1254969](https://www.sciencedirect.com/science/article/pii/S0306261925002260)

## Copyright and license

This project is licensed under the BSD 3 Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We gratefully acknowledge the financial support by Federal Ministry \\ for Economic Affairs and Climate Action (BMWK), promotional references 03ET1495A and 03EN1006A.

<img src="./docs/images/logos/BMWK_logo.png" alt="BMWK" width="200"/>
