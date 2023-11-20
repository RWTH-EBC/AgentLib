FROM registry.git-ce.rwth-aachen.de/ebc/projects/ebc_acs0017_bmwi_agent/agents_python/agentlib:latest

COPY examples/multi-agent-systems/pingpong/pingpong_module.py pingpong_module.py
COPY agentlib/modules/communicator/clonemap.py clonemap.py
CMD ["python", "-u", "clonemap.py"]
