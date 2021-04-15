FROM python:3.9.4-slim-buster
WORKDIR /webfish
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ["bash"]