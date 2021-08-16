#!/bin/bash
export CRON_HOME="/home/lombelet/cron"
export WF_HOME=${CRON_HOME}/webfish

${CRON_HOME}/cron/bin/python ${WF_HOME}/scripts/upload_datasets.py "$@" >> ${CRON_HOME}/cron.log 2>&1

