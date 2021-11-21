#!/bin/bash

# 1 -> folder name in arti and valid model name!
# 2 -> path to model file
# 3 -> path to log file 

# push model itself
curl -umaciektatarski@gmail.com:APAntTipeLPUsdB53fm4YkcoMtA -T ./$2 "https://dlmodels.jfrog.io/artifactory/iqiwa-models/$1/$1.pth"

# push logs
curl -umaciektatarski@gmail.com:APAntTipeLPUsdB53fm4YkcoMtA -T ./$3 "https://dlmodels.jfrog.io/artifactory/iqiwa-models/$1/$1.log" || echo "logs not found"

# push example phots
# to be done