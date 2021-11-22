#!/bin/bash
# example usage ./get_images.sh "model_res_enh_autoencoder/model_res_enh_autoencoder_22112021_09_10 2" 
curl -L -umaciektatarski@gmail.com:APAntTipeLPUsdB53fm4YkcoMtA -O "https://dlmodels.jfrog.io/artifactory/iqiwa-models/$1/model.pth"
curl -L -umaciektatarski@gmail.com:APAntTipeLPUsdB53fm4YkcoMtA -O "https://dlmodels.jfrog.io/artifactory/iqiwa-models/$1/model.py"
python3 generate_results.py --modeldef=model --modelparams=model.pth --out=image
