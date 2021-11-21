#!/bin/bash

# 1 -> folder name in arti and valid model name!
# 2 -> path to model file
# 3 -> path to log file 

# push model itself
curl -umaciektatarski@gmail.com:APAntTipeLPUsdB53fm4YkcoMtA -T ./$2 "https://dlmodels.jfrog.io/artifactory/iqiwa-models/$1/$2.pth"

# push logs
curl -umaciektatarski@gmail.com:APAntTipeLPUsdB53fm4YkcoMtA -T ./$3 "https://dlmodels.jfrog.io/artifactory/iqiwa-models/$1/$2.log" || echo "logs not found"

# push model file
curl -umaciektatarski@gmail.com:APAntTipeLPUsdB53fm4YkcoMtA -T ../scripts/$3.py "https://dlmodels.jfrog.io/artifactory/iqiwa-models/$1/$2.py" || echo "logs not found"
# push example phots

# to be done