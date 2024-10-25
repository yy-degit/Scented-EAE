# Scented-EAE
Scented-EAE: Stage-Customized Entity Type Embedding for Event Argument Extraction - [[Paper](https://aclanthology.org/2024.findings-acl.309.pdf)] (ACL Findings)


## Data Preprocessing
1. Construct the subdirectory structure of the `data` fold as follows:

```
-- data
  -- dataset
    -- ace
      -- event-split
      -- event-plus-split
    -- ere
      -- split
  -- new_data
  -- prompt
    -- different template/map.json
  -- raw_data
```

2. Download the [[ACE05](https://catalog.ldc.upenn.edu/LDC2006T06)] / [[ERE](https://catalog.ldc.upenn.edu/LDC2023T04)] raw datasets.

3. After extracting the above dataset archives, place the contents of the subdirectories into `data/dataset/ace` and `data/dataset/ere` respectively.

4. Obtain the ACE05 / ACE05+ / ERE data for training and evaluation.

```
# ACE05
conda deactivate
conda create --name eae-preprocess python=3.7
conda activate eae-preprocess
pip install -r preprocess/ace/requirements.txt
bash preprocess.sh  # set dataset=ace, need_ace_preprocess=true

# ACE05+

# ERE
```


6. a 



1. 运行preprocess.sh得到模型可读入数据
2. 运行run.sh训练评估测试
3. scripts里面是每个模型的训练参数
