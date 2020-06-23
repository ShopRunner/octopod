FROM nvcr.io/nvidia/pytorch:20.02-py3
MAINTAINER data-science@shoprunner.com
USER root

RUN apt-get update \
    && apt-get clean

COPY requirements.txt /tonks/
WORKDIR /tonks

RUN python -m pip install -U --no-cache-dir pip
RUN pip install -r requirements.txt
RUN pip list > /python_environment.txt

COPY . /tonks

CMD /bin/bash
