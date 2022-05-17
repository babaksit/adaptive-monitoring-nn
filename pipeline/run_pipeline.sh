#!/bin/bash
python3.8 main.py "$@" &
while true; do sleep 1; done