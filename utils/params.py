import argparse
import random
import torch
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser()
    
    # data training args
    parser.add_argument("--task", type=str, default="ALL", choices=["ALL", "NER", "EAE"], help="Name of task.")
    parser.add_argument("--data_dir", type=str, default="data/new_data/ace", help="Directory of data.")
    parser.add_argument("--train_file", type=str, default="data/new_data/ace/train.json", help="Path to training set.")
    parser.add_argument("--valid_file", type=str, default="data/new_data/ace/valid.json", help="Path to validation set.")
    parser.add_argument("--test_file", type=str, default="data/new_data/ace/test.json", help="Path to test set.")
    parser.add_argument("--schema_file", type=str, default="data/new_data/ace/schema.json", help="Path to schema.")
    parser.add_argument("--template_file", type=str, default="data/new_data/ace/template.json", help="Path to template.")
    
    parser.add_argument("--do_train", action="store_true", help="Whether to do training or not.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to do validation or not.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to do testing or not.")

    parser.add_argument("--encoder_max_seq_length", type=int, default=256, help="Maximum of sequence length for encoder.")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Only select some samples from datasets.")
    parser.add_argument("--seed", type=int, default=48, help="Random seed.")
    parser.add_argument("--bart_learning_rate", type=float, default=2e-5, help="Learning rate for bart.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for others.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warm up value.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size of training.")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size of validation or test.")
    parser.add_argument("--drop_last", action="store_true", help="Whether to drop last batch of dataset.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Specific steps to do evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Specific steps to update gradient.")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Parameters to reduce gradient")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay value.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    
    parser.add_argument("--max_entity_num", type=int, default=22, help="Threshold to decide the result of binary classification.")
    parser.add_argument("--max_threshold", type=float, default=0.5, help="Maximum of threshold to decide the result of binary classification.")
    parser.add_argument("--min_threshold", type=float, default=0.0, help="Minimum of threshold to decide the result of binary classification.")
    parser.add_argument("--threshold_intervals", type=int, default=5, help="Num of threshold intervals.")
    
    # model args
    parser.add_argument("--model_name_or_path", type=str, default="plm/bart-base", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir", type=str, default='result', help="Directory of result.")
    parser.add_argument("--output_model_path", type=str, default='result/best_checkpoint', help="Path to model stored.")
    parser.add_argument("--load_model_path", type=str, default='result/best_checkpoint', help="Path to load model stored.")


    args = parser.parse_args()
    
    args.thresholds = get_thresholds(args.max_threshold, args.min_threshold, args.threshold_intervals)
    set_seed(args.seed)
    
    return args


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_thresholds(max_threshold, min_threshold, threshold_intervals):
    assert threshold_intervals>0, "Invalid threshold_intervals!"
    interval_length = (max_threshold-min_threshold)/threshold_intervals
    thresholds = [min_threshold+(i)*interval_length for i in range(threshold_intervals,0,-1)]
    thresholds[0] = max_threshold
    return thresholds


def add_config_params(args, config): 
    config.task = args.task
    config.bio_type2id = args.schema["bio_type2id"]
    config.entity_type2id = args.schema["entity_type2id"]
    config.dropout = args.dropout
    config.max_entity_num = args.max_entity_num
    config.entity_bio_map = args.schema["entity_bio_map"]
    
    return config