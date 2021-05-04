#!/bin/bash

# Some useful commands to start Webfish in a Docker container
# builds the Docker image from the Dockerfile
docker build -t test/webfish .

# Starts a container running the image with these options:
# --name webfish_container_1: gives the container a nice name so we can access it later
# -t: opens a terminal in the container, which causes it to persist (i.e. wait for input)
# -d: runs in detached mode, i.e. returns from the command to our local terminal
#   (Use -i instead of -d to remain interactive and execute commands on the container.)
# --entrypoint bash: starts running command bash instead of the default entrypoint command
#   which is python from the base image
# --init: starts an initd process as PID 1 like a "normal" linux machine would.
#   not sure if this is necessary, but it may help with some services in the future.
# -p 8050:8050: forwards local port 8050 to port 8050 on the container, allowing us
#   to tunnel to this port to access the webfish app in the browser. This would
#   be whatever you want $WEBFISH_PORT to be.
docker run --name webfish_container_1 -td --entrypoint bash --init -p 8050:8050 test/webfish

# Copies our AWS credentials file to the /webfish directory (where everything is installed)
# on the container.
docker cp path/to/aws/credentials webfish_container_1:/webfish

# Finally, run webfish using 0.0.0.0 as the host and specifying the location of the
# credentials file. Options -d, -i and -t can be used just like docker run to execute
# in detached or interactive (terminal) mode.
docker exec -d env WEBFISH_CREDS=./credentialsfile WEBFISH_HOST=0.0.0.0 python index.py