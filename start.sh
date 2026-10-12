#!/bin/bash

PORT=7860


echo "check if $PORT is using..."
PID=$(lsof -t -i :$PORT)
if [ -n "$PID" ]; then
    echo "find $PID using $PORT£¬kill it..."
    kill -9 $PID
    if [ $? -eq 0 ]; then
        echo "$PID is killed"
    else
        echo "failed to kill $PID"
        exit 1
    fi
else
    echo "$PORT is free"
fi


if [ "$1" = "--daemon" ]; then
    nohup gradio app.py > log.txt 2>&1 &
    echo "launched service. Check logs in log.txt"
else
    echo "please use following cmd to start the service: ./start.sh --daemon"
fi