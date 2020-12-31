FROM nvcr.io/nvidia/pytorch:20.02-py3
MAINTAINER data-science@shoprunner.com

USER root

RUN apt-get update \
    && apt install -y libgl1-mesa-glx \
    && apt-get clean

COPY requirements.txt /octopod/
WORKDIR /octopod

RUN python -m pip install -U --no-cache-dir pip
RUN pip install -r requirements.txt

COPY setup.py README.md LICENSE.txt requirements.txt /octopod/
COPY octopod/_version.py /octopod/octopod/

RUN pip install -e .
RUN pip list > /python_environment.txt

COPY . /octopod

CMD /bin/bash
