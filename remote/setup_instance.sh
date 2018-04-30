#!/bin/bash

# could do $ ssh -t instance@host.com << EOF
# but ain't gotta do this here, just pipe:
# cat run_remote_experiment.sh | ssh -t user@host.com

echo ''
echo '> sudo apt-get update'
sudo apt-get update

echo ''
echo '> sudo apt-get --yes install python3-tk'
sudo apt-get --yes install python3-tk
echo ''
echo '> sudo apt-get --yes install htop'
sudo apt-get --yes install htop

echo ''
echo '> git clone -b current_experiment https://github.com/felix-last/evaluate-kmeans-smote'
git clone -b current_experiment https://github.com/felix-last/evaluate-kmeans-smote

echo ''
echo '> curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py'
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
echo ''
echo '> sudo python3 get-pip.py'
sudo python3 get-pip.py
echo ''
echo '> rm get-pip.py'
rm get-pip.py

echo ''
echo '> cd evaluate-kmeans-smote'
cd evaluate-kmeans-smote
echo ''
echo '> sudo pip3 install -r requirements.txt'
sudo pip3 install -r requirements.txt
echo ''
echo '> cd ..'
cd ..

echo ''
echo '> mkdir results'
mkdir results
echo ''
echo '> mkdir datasets'
mkdir datasets
