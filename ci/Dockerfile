FROM python:3.9-slim

#RUN apt add git
RUN apt-get update -y && apt-get install git -y
# Set workdir
WORKDIR /agency
# Insall the agentlib
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY agentlib agentlib
COPY README.md README.md
RUN pip install --upgrade pip && \
    pip install -e .[full]
RUN pip install git+https://github.com/sogno-platform/clonemapy

ENV PYTHONPATH=.

CMD ["python", "-u", "agentlib/modules/communicator/clonemap.py"]
