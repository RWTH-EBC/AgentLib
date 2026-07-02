PingPong
--------

To demonstrate the construction of a multi-agent-system (MAS) and the 
communication 
between the agents, a series of PingPong examples is available, 
corresponding to different communicators. 

### Running PingPong Local Broadcast
The first multi-agent system we're looking at is the pingpong local 
broadcast example, which is located at
``examples/multi-agent-systems/pingpong/pingpong_local_broadcast.py``. The 
local broadcast is the simplest communicator and can be used for a 
multi-agent-system running in a single process.
To set up a multi-agent-system by hand, the following steps are required:
1. Specify the environment configuration
2. Specify the config of all agents
3. Create the agents
4. Run the simulation by calling the run() method of the environment.

Let's go through the example from top to bottom. First, look at the imports.

```python
import logging
import pingpong_module
from agentlib.core import Environment, Agent
```

We import logging, the pingpong_module from the same directory and 
Environment and Agent from the agentlib.core. The logging module is only 
required to define the log level, while Environment and Agent are the 
components we use to create a local Multi-Agent-System. Finally, we import 
the pingpong_module, so we can conveniently access its filepath later.

.. note::
    In general, you don't need to import the modules you are using, it is 
    sufficient to know their file path. Here, we only import it so we can 
    access its file path conveniently using the ``__file__`` attribute.

After setting the log_level, we set up the environment config.
A config can be a path to a .json file, or a python dictionary. 

```python
env_config = {"rt": True, "factor": 1}
```

In this case, we set "rt" to true, meaning the
simulation is performed in Realtime. The factor modifies the speed at which
time passes in Realtime simulations. A factor smaller than one means the
simulation goes faster, a factor larger than 1 means the simulation goes
slower.


Now we setup the config of our first agent. An agent config always consists
of an "id" and a list of "modules".

```python
agent_config1 = {"id": "FirstAgent", 
 "modules":
     [
         {"module_id": "Ag1Com",
          "type": "local_broadcast"},
{"module_id": "Ping",
 "type": {"file": pingpong_module.__file__,
          "class_name": "PingPong"},
 "start": True},
```
The id is a string and has to be unique within the multi-agent-system. Next, we specify a list of modules. 
A module config consist at least of a "module_id" and a "type".
The first module we specify is a communicator.
The first method to define a module type is to
choose a standard module from the AgentLib
modules. To learn the identifier of module type,
look in the ``__init__.py`` files in the 'modules'
subpackages of the AgentLib. Here, we choose
"local_broadcast" from the communicator package as
our communicator.
Our second module is the pingpong module. For this module, we will use custom injection to specify its type, as it is not from the standard AgentLib modules. We also specify a parameter.

When using custom injection, the type of a module
is not specified by a string, but a dictionary,
consisting of "file" and "class_name". This way,
you can load any python class that inherits from
the standard module into the agent config.
Here, the expression ``pingpong_module.__file__``
generates the file path to the pingpong_module we
imported and "PingPong" is the name of the module
class.

The pingpong module takes one parameter.
Parameters are specified by a list of
dictionaries. To specify a parameter, one MUST
include its name and should also provide a value.
Here, we set the ``start`` parameter of the first
agent's pingpong module to ``True``, so it knows it
has to start the match.

The second agent's config looks similar to the first one. Here, we skip 
setting the ``start`` parameter, as it defaults to ``False``. After setting 
all the configs, we can start the actual script. 

```python
if __name__ == '__main__':
    env = Environment(config=env_config)
    agent1 = Agent(config=agent_config1, env=env)
    agent2 = Agent(config=agent_config2, env=env)
    env.run(until=None)
```
First, we create the environment and the agents. The environment takes its 
config as a single argument, an agent takes its config and an environment as 
arguments.
Finally, we run the simulation. Depending on the log_level, you can now see 
the agents messaging each other with 'ping' and 'pong' messages.
Since we set ``until`` to ``None``, the script will never stop running unless we 
externally stop the process, e.g. via KeyboardInterrupt.


### The pingpong_module
Now that we have seen how to run a MAS, let's look at what is happening 
inside the agents and how to implement your own functionality. The pingpong 
module is located at
``examples/multi-agent-systems/pingpong/pingpong_module.py``.

As usual, the file begins with the module-level imports.

```python
import logging
import time
import sys
from pydantic import Field
from agentlib.core import BaseModule, BaseModuleConfig
from agentlib.core.datamodels import AgentVariable, Causality

logger = logging.getLogger(__name__)
```
``logging``, ``time`` and ``sys`` are used for the functionality of the 
module, which we will look at later. From pydantic we import Field. Pydantic 
is used in _AgentLib_ to specify module configurations and validate them. 
Finally, from the _AgentLib_'s core we import ``BaseModule``, which we need to inherit 
from, ``BaseModuleConfig`` for the configuration and ``AgentVariable``, 
``AgentVariable`` and ``Causality``.
``AgentVariable`` is the base class for all quantities that need to be 
communicated throughout a MAS. In a sense, they can simply be interpreted as 
message objects. ``AgentVariable`` is, as the name implies, a message that 
should be sent to other agents. We will look at ``Causality`` later.

```python
class PingPongConfig(BaseModuleConfig):
    start: bool = Field(
        default=False,
        description="Indicates if the agent should start communication"
    )
    initial_wait: float = Field(
        default=0,
        description="Wait the given amount of seconds before starting."
    )

class PingPong(BaseModule):

    config: PingPongConfig
```
To create a module, we need to declare a class which inherits from ``BaseModule``. 
Every module has a Settings inner class, which inherits from the BaseModule 
Settings. The attributes specified in this class are the module config. 
Usually, these attributes can be any Python object.\

Let's see how we can access our config property. In the case of properties directly 
defined in the config, they can also be directly accessed through ``self.config.my_property``.
If the property is an AgentVariable, it can be retrieved with its current value 
throguh the ``self.get(<<name>>)`` method of the base class with a 
name corresponding to a variable that was defined in the config.

Next, there are two abstract methods we need to overwrite:
- ``process()``
- ``register_callbacks()``

The process method is a generator method which is called by the Environment. 
Using yield statements, the control over the simulation flow is given back 
to the event manager of the environment. However in this example, the only 
function of the process is to start the pingpong game.

```python
    def process(self):
        if self.config.start:
            self.logger.debug("Waiting %s s before starting",
                              self.config.initial_wait)
            yield self.env.timeout(self.config.initial_wait)
            self.logger.debug("Sending first message: %s", self.id)
            self.agent.data_broker.send_variable(
                AgentVariable(name=self.id,
                              value=self.id,
                              source=self.source,
                              shared=True))
        yield self.env.event()
```
If the agent starts, i. e. ``start == True``, an AgentVariable is 
set in the data_broker, with name and value corresponding to the own module id.
The agent will wait with sending the message for a specified amount of time. 
In this toy example, that is necessary in the examples with other communicators, so the message is not sent before the other ping pong agent can listen.
The data_broker is the place through which all variables of an Agent are communicated, 
and 
serves to connect the modules with each other, most importantly the 
communicator of the agent. Without going into further detail at this point, 
the above statement will send out an AgentVariable, which can be received by 
other agents. \
This is, where callbacks come into play. Callbacks are functions, that are 
executed, when variables are written to the data_broker. Since we are playing 
Pingpong, we want to answer when we receive a message.
Let's look at the second method we 
need to overwrite, ``register_callbacks()``.

```python
def register_callbacks(self):
    if self.id == "Ping":
        alias = "Pong"
    else:
        alias = "Ping"
    self.agent.data_broker.register_callback(
        alias=alias, source=None, callback=self._callback
    )
```
To register a callback, we need to call the ``register_callback()`` method 
of the data_broker and pass it 3 arguments. The first two arguments, alias and 
source specify, for which variables the callback should be 
executed. The last argument is the function that should be executed on callback.
With the specification above, we execute the callback on variables with any 
source and either _Ping_ or _Pong_ as alias. Thus, the module with _Ping_ as its 
alias will listen to the message with alias _Pong_ and vice-versa. 

Finally, there is the function that is executed on callback. A callback 
function always takes an AgentVariable as its single argument. This function 
logs the received variable to the console, waits a second and then also 
sends a variable.

```python
def _callback(self, variable: AgentVariable):
    logger.info("%s received: %s",
                self.agent.id, variable.json())
    sys.stdout.flush()
    time.sleep(1)
    self.agent.data_broker.send_variable(
        AgentVariable(name=self.id,
                    value=self.id,
                    source=self.source))
```

