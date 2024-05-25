#!/usr/bin/env bash
# -*- coding:utf-8 -*-
EXP_ID=$(date +%F-%H-%M-$RANDOM)
export CUDA_VISIBLE_DEVICES=0
export task=ALL
export data_dir=data/aceplus
export train_file=train.json
export valid_file=test.json
export test_file=test.json
export schema_file=schema.json
export template_file=template.json
export encoder_max_seq_length=220
export sample_ratio=1.0
export seed=42
export bart_learning_rate=4e-5
export learning_rate=4e-5
export warmup_ratio=0.1
export batch_size=16
export epochs=1
export logging_steps=10
export gradient_accumulation_steps=1
export max_grad_norm=5.0
export weight_decay=0.01
export dropout=0.1
export max_threshold=0.9
export min_threshold=0.0
export threshold_intervals=18
export max_entity_num=28
export model_name_or_path=plm/bart-base
export output_dir=result/${task}/${EXP_ID}
export output_model_path=${output_dir}/best_checkpoint
export load_model_path=${output_model_path}
export logging_file=${output_dir}/finetune.log


mkdir -p ${output_dir}
cp run.sh ${output_dir}/params.sh


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py \
    --do_train --do_eval --do_predict \
    --task=${task} \
    --train_file=${data_dir}/${train_file} \
    --valid_file=${data_dir}/${valid_file} \
    --test_file=${data_dir}/${test_file} \
    --schema_file=${data_dir}/${schema_file} \
    --template_file=${data_dir}/${template_file} \
    --encoder_max_seq_length=${encoder_max_seq_length} \
    --sample_ratio=${sample_ratio} \
    --seed=${seed} \
    --bart_learning_rate=${bart_learning_rate} \
    --learning_rate=${learning_rate} \
    --warmup_ratio=${warmup_ratio} \
    --train_batch_size=${batch_size} \
    --eval_batch_size=$((batch_size*4)) \
    --drop_last \
    --epochs=${epochs} \
    --logging_steps=${logging_steps} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --max_grad_norm=${max_grad_norm} \
    --weight_decay=${weight_decay} \
    --dropout=${dropout} \
    --max_threshold=${max_threshold} \
    --min_threshold=${min_threshold} \
    --threshold_intervals=${threshold_intervals} \
    --max_entity_num=${max_entity_num} \
    --model_name_or_path=${model_name_or_path} \
    --output_dir=${output_dir} \
    --output_model_path=${output_model_path} \
    --load_model_path=${load_model_path} \
    >${logging_file}