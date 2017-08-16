#!/bin/bash

nohup sh -c 'cd evaluate-kmeans-smote && python3 imbalanced_benchmark.py && cd remote && python3 remote_experiment_finished.py && echo experiment_done' &
exit
