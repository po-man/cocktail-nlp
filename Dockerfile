FROM python:3.7

WORKDIR /root

ARG BUILD_DIR

COPY ${BUILD_DIR}/requirements.txt /root/requirements.txt

RUN pip install -r requirements.txt
