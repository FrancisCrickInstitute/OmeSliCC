FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade && apt-get clean all

RUN apt-get -y install build-essential
#RUN apt-get install -y libsm6 libxext6 libxrender-dev
# For: libGL.so.1
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y openssl libssl-dev libbz2-dev
#RUN apt-get install -y python3 python3-pip
#RUN pip3 install --upgrade pip
RUN apt-get install -y python3-pip
RUN apt-get install -y openjdk-8-jdk

COPY requirements.txt requirements.txt
# Javabridge workaround: install numpy first:
RUN pip install numpy
RUN pip install -r requirements.txt

COPY . .

WORKDIR /omeslicc

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3", "/run.py"]
