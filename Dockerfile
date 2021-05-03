FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
MAINTAINER data-science@shoprunner.com

USER root

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get install -y libglib2.0-0 \
    && apt-get clean

COPY requirements.txt /octopod/
WORKDIR /octopod

RUN python -m pip install -U --no-cache-dir pip
RUN pip install -r requirements.txt

COPY setup.py README.md LICENSE requirements.txt /octopod/
COPY octopod/_version.py /octopod/octopod/

RUN pip install -e .
RUN pip list > /python_environment.txt

COPY . /octopod

CMD /bin/bash
