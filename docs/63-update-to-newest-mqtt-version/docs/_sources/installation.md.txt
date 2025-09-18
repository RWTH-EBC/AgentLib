# Installation

The AgentLib is on PyPI. 
To use it, simply install it with pip:

```
pip install agentlib
```

The basic version comes with minimal dependencies.
To install with full dependencies (recommended), run:
```
pip install agentlib[full]
```

If you want to work on the AgentLib, first clone it from [github](https://github.com/RWTH-EBC/AgentLib), go its directory and then install it in editable mode:

```
pip install -e .[full]
```

If later you want to use uninstalled modules, an `OptionalDependencyError` will tell you how to install this specific module.

## Optional dependencies

To install optional dependencies, install the AgentLib with the key, i.e. to install with FMU support, do:
```
pip install agentlib[fmu]
```


| Key      | Installs                                           | Used for                                                                  |
|----------|----------------------------------------------------|---------------------------------------------------------------------------|
| fmu      | FMPy>=0.3.6                                        | FMU model                                                                 |
| scipy    | scipy>=1.5.2                                       | Statespace model                                                          |
| plot     | matplotlib                                         | Plots in examples                                                         |
| mqtt     | paho-mqtt>=1.6.1                                   | MQTT communicator module                                                  |
| fuzz     | rapidfuzz>=1.7.0                                   | Improves error messages when misspelling module or model type identifiers |
| orjson   | orjson>=3.9.5                                      | Improves performance of json (de)serialization in communicators like mqtt |
| clonemap | [clonemapy](https://github.com/RWTH-ACS/clonemapy) | Utility to execute agents in clonemap                                     |