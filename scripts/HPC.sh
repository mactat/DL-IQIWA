#!/bin/bash
git reset --hard HEAD
git pull
chmod +x ../model/push_to_arti.sh  
pip3 install --user -r ../requirements.txt
sxm2sh
