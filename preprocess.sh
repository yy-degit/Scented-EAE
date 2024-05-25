export dataset=ace
export need_ace_preprocess=false
export need_aceplus_preprocess=false
export need_casie_preprocess=false
export need_ere_preprocess=false

if [ ${dataset} == "ace" ] && [ ${need_ace_preprocess} == true ]
then
    # ACE dataset needs to be preprocessed
    ace_dir=data/dataset/ace
    ace_original_dir=${ace_dir}/data/English
    ace_target_dir=${ace_dir}/target

    # copy useful files to target directory 
    mkdir -p ${ace_target_dir}
    ls ${ace_original_dir} |
    while read subdir
    do
        this_dir=${ace_original_dir}/${subdir}/timex2norm
        cp ${this_dir}/*apf.xml ${ace_target_dir}
        cp ${this_dir}/*.sgm ${ace_target_dir}
    done

    # analyze original files to json format
    ace_data_dir=${ace_target_dir}
    ace_output_dir=data/raw_data/ace
    ace_split_dir=${ace_dir}/event-split
    mkdir -p ${ace_output_dir}
    python preprocess/ace/parse_ace_event.py \
        --data_dir=${ace_data_dir} \
        --output_dir=${ace_output_dir} \
        --split_dir=${ace_split_dir}
fi

if [ ${dataset} == "aceplus" ] && [ ${need_aceplus_preprocess} == true ]
then
    aceplus_dir=data/dataset/ace
    aceplus_data_dir=${aceplus_dir}/data
    aceplus_output_dir=data/raw_data/aceplus
    aceplus_split_dir=${aceplus_dir}/event-plus-split
    bert=plm/bert-large-cased
    bert_cache_dir=None
    language=english
    mkdir -p ${aceplus_output_dir}
    python preprocess/aceplus/parse_aceplus_event.py \
        --input=${aceplus_data_dir} \
        --output=${aceplus_output_dir} \
        --split=${aceplus_split_dir} \
        --bert=${bert} \
        --bert_cache_dir=${bert_cache_dir} \
        --lang=${language}
fi

if [ ${dataset} == "casie" ] && [ ${need_casie_preprocess} == true ]
then
    # bash preprocess/casie/process.sh  !!! please run this file in its own folder
    # cp -r preprocess/casie/data data/dataset/casie/target  !!! and then copy 
    casie_data_dir=data/dataset/casie/target
    casie_output_dir=data/raw_data/casie
    mkdir -p ${casie_output_dir}
    python preprocess/casie/parse_casie_event.py \
        --data_dir=${casie_data_dir} \
        --output_dir=${casie_output_dir}
fi

if [ ${dataset} == "ere" ] && [ ${need_ere_preprocess} == true ]
then
    ere_dir=data/dataset/ere
    ere_data_dir=${ere_dir}/data
    ere_output_dir=data/raw_data/ere
    ere_split_dir=${ere_dir}/split
    bert=plm/bert-large-cased
    bert_cache_dir=None
    language=english
    mkdir -p ${ere_output_dir}
    python preprocess/ere/parse_ere_event.py \
        --input=${ere_data_dir} \
        --output=${ere_output_dir} \
        --split=${ere_split_dir} \
        --bert=${bert} \
        --bert_cache_dir=${bert_cache_dir} \
        --lang=${language}
fi


export raw_data_dir=data/raw_data
export train_file=train.json
export valid_file=valid.json
export test_file=test.json
export entity_file=data/prompt/${dataset}_entity_map.json
export schema_file=schema.json
export output_dir=data/new_data


python preprocess/data_preprocessing.py \
    --dataset=${dataset} \
    --raw_data_dir=${raw_data_dir} \
    --train_file=${train_file} \
    --valid_file=${valid_file} \
    --test_file=${test_file} \
    --entity_file=${entity_file} \
    --schema_file=${schema_file} \
    --output_dir=${output_dir}


cp data/prompt/${dataset}_template.json ${output_dir}/${dataset}/template.json