#!/bin/bash
python3.8 subscriber.py "$@" &
while true; do sleep 1; done