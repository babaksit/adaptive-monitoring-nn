#!/bin/bash
#Mica2Dot
mkdir -p mica2dot
wget -O mica2dot.txt.gz http://db.csail.mit.edu/labdata/data.txt.gz
gunzip -c mica2dot.txt.gz > data/mica2dot.txt
