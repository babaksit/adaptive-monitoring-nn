#!/bin/bash
python3.8 csv_exporter.py "$@" &
while true; do sleep 1; done