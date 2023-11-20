FROM python:3.8-slim

#RUN apt add git
RUN apt-get update -y && apt-get install git -y
# Set workdir
WORKDIR /agency
# Insall the agentlib
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY setup.py setup.py
COPY agentlib agentlib
RUN pip install -e .
ENV PYTHONPATH=.

CMD ["python", "-u", "agentlib/modules/communicator/clonemap.py"]
