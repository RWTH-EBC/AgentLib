# List of standard modules and models

Here, we list all the modules and model types, that are shipped with the AgentLib core.
The _identifier_ specifies the string that should be used in type to use the module or model (see example at the Reference [bottom of this page](type_example))

## Communicators
link to simulator Reference [simulator](simulator)
link to simulator Reference {doc}`Simulator <simulator_page>`

| Identifier                | Extra dependencies | Description                                                                                                                                                                          | Reference                                                       |
|---------------------------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| local                     | -                  | Exchanges messages between agents executed within the same Python process. Usable with realtime and instant execution. Specify, which agents you subscribe to.                       | [local](local_comm)                                             |
| local_broadcast           | -                  | Exchanges messages between agents executed within the same Python Process, realtime or instant execution. Messages are sent to all other agents with a local_broadcast communicator. | [local_broadcast](local_broadcast)                              |
| multiprocessing_broadcast | -                  | Exchanges messages between all agents within a network that have this communicator, and are configured with the same ipv4 and port.                                                  | [multiprocessing_broadcast](simulatormultiprocessing_broadcast) |
| mqtt                      | mqtt (paho-mqtt)   | Exchanges messages through MQTT. Requires the address of an MQTT broker, and you need to specify, which agents you subscribe to.                                                     | [mqtt](mqtt_comm)                                               |
| clonemap                  | clonemap           | Exchange messages through cloneMAP (with MQTT). Used, when MAS is executed via cloneMAP.                                                                                             | [clonemap](clonemap)                                            |



## Controllers
| Identifier | Required dependencies | Description                     | Reference              |
|------------|-----------------------|---------------------------------|------------------------|
| pid        | -                     | A simple PID controller.        | [pid](pid)             |
| bangbang   | -                     | A simple hysteresis controller. | [bangbang](bangbang)   |


## Miscellaneous
| Identifier  | Required dependencies | Description                                                                                                     | Reference                  |
|-------------|-----------------------|-----------------------------------------------------------------------------------------------------------------|----------------------------|
| simulator   | -                     | A module to simulate models. Models can be FMU, state-space or custom defined, or come from plugins.            | [simulator](simulator)     |
| agentlogger | -                     | Logs the AgentVariables that are exchanged within an agent, effectively keeping a log of the state of an agent. | [agent_logger](agent_logger) |
| trysensor   | -                     | Sends weather data of the Test Reference Year 2015.                                                             | [try_sensor](try_sensor)   |

## Models

| Identifier | Extra dependencies | Description                              | Reference                  |
|------------|--------------------|------------------------------------------|----------------------------|
| statespace | scipy              | Linear state space model based on scipy. | [fmu_model](fmu_model)     |
| fmu        | fmu (fmpy)         | Used to simulate FMUs.                   | [scipy_model](scipy_model) |


(type_example)=

**Example**

When using modules, they are identified by the ```"type"``` keyword in the configuration, which should include an identifier from this table, or a custom injection (combination of file and class name). 
For example, to run an agent with a simulator and an FMU model, the config should look like this:

````json
{
  "id": "myAgent",
  "modules": [
    {
      "type": "simulator",
      "model": {
        "type": "fmu",
        "path": "path/toFMU.fmu"
      },
      
      "...": "remaining simulator config goes here"
    }
  }
````