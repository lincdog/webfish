#!/bin/bash
CRON_HOME="/home/lombelet/cron"
export WF_HOME=${CRON_HOME}/webfish

${CRON_HOME}/cron/bin/python ${WF_HOME}/scripts/upload_logging.py \
    >>${CRON_HOME}/logging.log 2>&1
${CRON_HOME}/cron/bin/python ${WF_HOME}/scripts/download_json_analyses.py \
    --allow-duplicates >>${CRON_HOME}/download.log 2>&1
