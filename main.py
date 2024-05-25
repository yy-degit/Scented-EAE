# -*- encoding:utf -*-
from transformers import *
from utils.params import parse_args, add_config_params
import logging
import sys
import torch
from models.model import ScentedEAE
from trainer.framework import Framework
from utils.dataloader import get_schema_dict, get_template_dict, data_loader
import warnings


warnings.filterwarnings('ignore')
MODEL_CALSSES = {'bart': (BartConfig, ScentedEAE, BartTokenizerFast)}


def main():
    # 传参+设置随机数种子
    args = parse_args()
    args.schema = get_schema_dict(schema_file=args.schema_file)
    args.template = get_template_dict(template_file=args.template_file, schema=args.schema)
    
    
    # 设置日志    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)


    # 设置运行在GPU或CPU上
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f'There are {torch.cuda.device_count()} GPU(s) available, We will use the GPU: {torch.cuda.get_device_name(0)}.')
    else:
        logger.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    args.device = device
    
    
    # 加载模型
    args.plm_type="bart"
    config_class, model_class, tokenizer_class = MODEL_CALSSES[args.plm_type]
    load_model_path = args.model_name_or_path if args.do_train else args.load_model_path
    config = config_class.from_pretrained(load_model_path)
    config = add_config_params(args, config)
    eaemodel = model_class.from_pretrained(load_model_path, from_tf=bool('.ckpt' in load_model_path), config=config)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, add_special_tokens=True)
    
    
    # 加载数据
    train_loader = data_loader(mission="train", args=args, tokenizer=tokenizer, shuffle=True)
    valid_loader = data_loader(mission="eval", args=args, tokenizer=tokenizer)
    test_loader = data_loader(mission="test", args=args, tokenizer=tokenizer)
    
    
    # 创建模型运行框架
    framework = Framework(args, eaemodel)


    # 模型训练
    if args.do_train:
        framework.train(train_loader, valid_loader)
        best_model = model_class.from_pretrained(args.output_model_path, from_tf=bool('.ckpt' in load_model_path), config=config)
        framework = Framework(args, best_model)
    
    
    # 模型评估
    if args.do_eval:
        framework.evaluate(valid_loader, output_result=True)
        
        
    # 模型测试
    if args.do_predict:
        framework.predict(test_loader)
        
        

if __name__ == "__main__":
    main()