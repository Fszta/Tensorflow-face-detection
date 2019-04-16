FROM ubuntu:16.04

FROM python:3

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip install -r requirements.txt

COPY . /

CMD [ "python", "./face_detection_webcam.py"]
