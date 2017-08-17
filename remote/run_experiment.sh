#!/bin/bash

cd evaluate-kmeans-smote
nohup python3 remote.py _experiment > ~/experiment.out &
exit
