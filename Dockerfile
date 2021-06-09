FROM python:3.9.4-slim-buster
WORKDIR /webfish
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY hpc-wasabi-usercredentials .
ENTRYPOINT ["env", "WEBFISH_CREDS=hpc-wasabi-usercredentials", "WEBFISH_HOST=0.0.0.0", "python", "index.py"]

HEALTHCHECK --interval=5m --timeout=3s --start-period=1m \
    CMD python -c "import urllib.request; \
    urllib.request.urlopen('http://localhost:8050')" || exit 1
