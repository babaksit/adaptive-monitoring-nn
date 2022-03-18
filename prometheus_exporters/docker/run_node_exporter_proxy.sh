#!/bin/bash
python3.8 node_exporter_proxy.py "$@" &
while true; do sleep 1; done