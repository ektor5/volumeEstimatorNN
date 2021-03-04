FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y parallel

RUN pip install sklearn
RUN pip install numpy

RUN mkdir /ext
COPY ./volEstNN.py /

WORKDIR /ext
CMD python /volEstNN.py
