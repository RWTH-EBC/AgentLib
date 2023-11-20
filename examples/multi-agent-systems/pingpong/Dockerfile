FROM registry.git.rwth-aachen.de/ebc/ebc_all/github_ci/agentlib:latest

COPY examples/multi-agent-systems/pingpong/pingpong_module.py pingpong_module.py
COPY agentlib/modules/communicator/clonemap.py clonemap.py
CMD ["python", "-u", "clonemap.py"]
