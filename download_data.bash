curl https://nlp.stanford.edu/~zijwang/talkdown/talkdown.tar.gz -o talkdown.tar.gz
mkdir -p data
tar xzf talkdown.tar.gz -C data/
rm talkdown.tar.gz
echo 'Done!'
