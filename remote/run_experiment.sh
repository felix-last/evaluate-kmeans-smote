#!/bin/bash

cd evaluate-kmeans-smote
nohup python3 remote.py _run_experiment &
exit
