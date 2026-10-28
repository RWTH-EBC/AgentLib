
## How to start a MAS on cloneMAP

In the following it will be described step-by-step how to execute a MAS on [cloneMAP](https://www.fein-aachen.org/projects/clonemap/).

### Start cloneMAP

There are two options for starting cloneMAP: locally and on a Kubernetes cluster.
Please refer to the [administration guide](https://github.com/RWTH-ACS/clonemap/blob/develop/docs/administration_guide.md) of cloneMAP for both options.

### Implement agent behavior and start MAS

For demonstration of this part the pingpong example in `examples/multi-agent-systems/pingpong` is used.

#### Step 1 Behavior implementation

The agent behavior in the AgentLib is composed of the individual behavior of modules.
There are no special requirements for the agent behavior implementation if you want to use cloneMAP for MAS execution.
You can either use the standard modules provided by the AgentLib or implement your own modules.

#### Step 2 Building the Docker image

If you only use the AgentLib standard modules, you can simply use the docker image provided in the repository's registry with the tag `clonemap`.
If you have implemented your own modules, you have to build a new image, which contains these modules.
Here is an example how a corresponding Dockerfile could look like:

```Docker
FROM registry.git.rwth-aachen.de/ebc/ebc_all/github_ci/agentlib:latest

COPY my_module.py my_module.py
CMD ["python", "-u", "agentlib/modules/communicator/clonemap.py"]
```

You start from the standard AgentLib image, add your custom module and set the entry point to the clonemap communicator.

#### Step 3 Scenario creation

Besides specifying the agent behavior you also have to define the MAS configuration in order to start a MAS application. The MAS configuration consists of a list of agents you want to execute as well as individual configuration information for each agent.
Take a look at the [API specification of the AMS](https://github.com/RWTH-ACS/clonemap/blob/develop/api/ams/openapi.yaml) for a description of the MAS configuration.
In the configuration you also have to specify the docker image to be used for the agencies.
This corresponds to the image you created in the previous step.

Here is an example configuration which starts an agent using the standard image.
This agent only has a clonemap communicator as module.
This is mandator to start agents with clonemap.

```json
{
    "config":{
        "name":"test",
        "agentsperagency":1,
        "mqtt":{
            "active":true
        },
        "df":{
            "active":false
        },
        "logger":{
            "active":true,
            "msg":true,
            "app":true,
            "debug":true
        }
    },
    "imagegroups":[
        {
            "config": {
                "image":"agentlibtest"
            },
            "agents":[
                {
                    "nodeid":0,
                    "name":"FirstAgent",
                    "type":"test",
                    "custom":"{\"id\":\"FirstAgent\", \"modules\":[{\"module_id\":\"AgCom\", \"type\":\"clonemap\", \"subtopics\":[\"/agentlib/SecondAgent/#\"]}]}"
                }
            ]
        }
    ],
    "graph":{
        "node":null,
        "edge":null
    }
}
```

In the first field, general configuration of the MAS can be adjusted.
One important field is `agentsperagency` which specifies the maximum number of agents that are executed in one agency container.
Within the logger field you can decide what to do with log messages from agents.
If `logger.active` is set to false, all logs will be added to stdout of the docker container and can be viewed with the `docker logs` command.
If `logger.active` is set to true, the cloneMAP logger is used.
That means that all logs are sent to the logger and can be viewed in the cloneMAP WebUI or downloaded in json format.
The fields `msg`, `app` and `debug` control which logs are saved.

The `imagegroups` field expects a list of image groups.
An image group contains the image to be used (`config.image`) and a list of agents to create (`agents`).
Each agent consists of a set of fields.
The important one is `custom` which is used to provide the configuration to the agent as specified by the agentlib, i.e, the configuration of all modules that make up the agent.
The `custom` field expects a string.
Since we use json as format for the agent configuration, make sure to escape quotation marks.

For a new MAS the provided example can be used as template.
The list of groups, the names of the images and the list of agents per image group have to be adjusted.

#### Step 4 MAS execution

In order to execute a MAS you have to post the previously created scenario file to the AMS.
Look at the [api specification](https://github.com/RWTH-ACS/clonemap/blob/develop/api/ams/openapi.yaml) for the correct path and method to be used.
Subsequently the AMS will start all agencies which then will start the single agents.
Depending on the size the creation of a MAS might take a few seconds.

```bash
curl -X "POST" -d @clonemap_config.yaml <ip-address>:30009/api/clonemap/mas
```

The AMS is made available to the outside world via NodePort on port 30009.
Replace `<ip-address>` with the IP address of one of your Kubernetes machines or localhost if you execute cloneMAP locally.
The AMS should answer with http code 201.
It starts the agency containers which automatically execute the agents.

To terminate the MAS run

```bash
curl -X DELETE <ip-address>:30009/api/clonemap/mas/0
```

Alternatively you can also use the WebUI for starting and terminating agents.
