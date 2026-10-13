#!/bin/bash

PORT=7860


echo "check if $PORT is using..."
PID=$(lsof -t -i :$PORT)
if [ -n "$PID" ]; then
    echo "find $PID using $PORT£¬kill it..."
    kill -9 $PID
    if [ $? -eq 0 ]; then
        echo "$PID is killed"
		sleep 2
        if lsof -t -i :$PORT > /dev/null; then
            echo "port $PORT still in use, force kill again"
            PID2=$(lsof -t -i :$PORT)
            kill -9 $PID2
            sleep 1
		fi
    else
        echo "failed to kill $PID"
        exit 1
    fi
else
    echo "$PORT is free"
fi

if lsof -t -i :$PORT > /dev/null; then
    echo "ERROR: port $PORT is still occupied, abort"
    exit 1
fi

if [ "$1" = "--daemon" ]; then
    nohup gradio app.py > log.txt 2>&1 &
    NEW_PID=$!
    sleep 3
    if kill -0 $NEW_PID 2>/dev/null; then
        echo "launched service (PID $NEW_PID). Check logs in log.txt: tail -f log.txt"
    else
        echo "ERROR: service failed to start. Check log.txt: tail -f log.txt"
        tail -20 log.txt
        exit 1
    fi
else
    echo "please use following cmd to start the service: ./start.sh --daemon"
fi