mkdir data

# download LjSpeech
echo "Downloading LjSpeech..."
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1/wavs data/wavs
rm -r LJSpeech-1.1
rm LJSpeech-1.1.tar.bz2

# remove extra data
rm mel.tar.gz

# download train.txt
echo "Downloading train.txt..."
gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt data/text.txt

echo "Download has been finished."