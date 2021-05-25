#!/bin/bash

# Some useful commands to start Webfish in a Docker container
# builds the Docker image from the Dockerfile
docker build -t lincdog/sandbox .

# Starts a container running the image with these options:
# --name webfish_container_1: gives the container a nice name so we can access it later
# -d: runs in detached mode, i.e. returns from the command to our local terminal
#   (Use -i instead of -d to remain interactive and execute commands on the container.)
# -p 8050:8050: forwards local port 8050 to port 8050 on the container, allowing us
#   to tunnel to this port to access the webfish app in the browser. This would
#   be whatever you want $WEBFISH_PORT to be.
docker run --name webfish_container_1 -d -p 8050:8050 lincdog/sandbox

# Copies our AWS credentials file to the /webfish directory (where everything is installed)
# on the container.
docker cp path/to/aws/credentials webfish_container_1:/webfish

