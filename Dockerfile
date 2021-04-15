FROM python:3.9.1
WORKDIR /webfish
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV WEBFISH_CREDS=a