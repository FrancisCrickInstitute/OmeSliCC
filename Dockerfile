FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade && apt-get clean all

RUN apt-get -y install build-essential
#RUN apt-get install -y libsm6 libxext6 libxrender-dev
# For: libGL.so.1
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y openssl libssl-dev libbz2-dev

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

WORKDIR /omeslicc

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3", "run.py"]
