pip3 install kaggle || pip3 install --user aggle
mkdir -p ~/.kaggle
echo $1
cp $1 ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list || ~/.local/bin/kaggle datasets list

kaggle competitions download -c dogs-vs-cats || ~/.local/bin/kaggle competitions download -c dogs-vs-cats 
mkdir -p data
unzip -qq -n dogs-vs-cats.zip -d ./
unzip -qq -n ./train.zip -d  data
unzip -qq -n ./test1.zip -d  data
rm -f dogs-vs-cats.zip test1.zip train.zip sampleSubmission.csv