FROM python:3.9.4-slim-buster
WORKDIR /webfish
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ["env", "WEBFISH_CREDS=ec2-readcredentials", "WEBFISH_HOST=0.0.0.0", "python", "index.py"]