#!/usr/bin/env bash
# -*- coding:utf-8 -*-


# download data
git clone https://hub.nuaa.cf/Ebiquity/CASIE.git || exit 1
cp -r CASIE/data raw_data
# ignore file with some error
for file_id in 999 10001 10002
do
  rm raw_data/source/${file_id}.txt raw_data/annotation/${file_id}.json
done


# download corenlp
mkdir corenlp
# Length: 504773765 (481M) [application/zip]
wget -P corenlp/backup http://nlp.stanford.edu/software/stanford-corenlp-4.1.0.zip
unzip -d corenlp corenlp/backup/stanford-corenlp-4.1.0.zip
# Length: 670717962 (640M) [application/java-archive]
wget -P corenlp/models http://nlp.stanford.edu/software/stanford-corenlp-4.1.0-models-english.jar


# run
python generate_content.py raw_data/annotation raw_data/content
export MEMORY_LIMIT=1g
export STANFORD_CORENLP_PATH=./corenlp/stanford-corenlp-4.1.0
export STANFORD_CORENLP_MODEL_PATH=./corenlp/models
export THREAD_NUM=8
java -mx${MEMORY_LIMIT} \
    -cp "${STANFORD_CORENLP_PATH}/*:${STANFORD_CORENLP_PATH}/lib/*:${STANFORD_CORENLP_PATH}/liblocal/*:${STANFORD_CORENLP_MODEL_PATH}/*" \
    edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,cleanxml,ssplit \
    -outputFormat json \
    -threads ${THREAD_NUM} \
    -outputDirectory raw_data/corenlp \
    -file raw_data/content
python check_stanford_corenlp.py
python extract_doc_json.py
python split_data.py
tree data/