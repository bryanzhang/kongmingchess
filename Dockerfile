FROM tensorflow/tensorflow:1.15.0-py3

ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y bc python3 python3-pip git vim
RUN pip3 install --upgrade pip
RUN pip3 install numpy gym stable-baselines3
