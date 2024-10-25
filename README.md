# Scented-EAE
Scented-EAE: Stage-Customized Entity Type Embedding for Event Argument Extraction - [Paper Address](https://aclanthology.org/2024.findings-acl.309.pdf) (Findings of ACL 2024)


## Data Preprocessing
1. Construct the subdirectory structure of the `data` fold as follows:

```
-- data
  -- dataset
    -- ace
      -- files we provided
    -- ere
      -- files we provided
  -- new_data
  -- prompt
    -- files we provided
  -- raw_data
```



1. 运行preprocess.sh得到模型可读入数据
2. 运行run.sh训练评估测试
3. scripts里面是每个模型的训练参数
