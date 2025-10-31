FROM tensorflow/tensorflow:2.18.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y parallel

RUN mkdir /app
WORKDIR /app
COPY ./requirements.txt /app

RUN pip install -r requirements.txt
COPY . /app

CMD sh launch.sh