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
pip install -r preprocess/ace/requirements.txt # need read to download something manually
bash preprocess.sh  # set dataset=ace, need_ace_preprocess=true

# ACE05+
pip install transformers==3.0.2 (It seems using transformers 3.1+ may cause some model loading issue)
pip install tqdm
python
>> import nltk 
>> nltk.download('punkt')
>> exit()
# if downloading nltk fails, you can download [punkt.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip), unzip it and move to `/root/nltk_data/tokenizers/punkt`
bash preprocess.sh  # set dataset=aceplus, need_aceplus_preprocess=true

# ERE
bash preprocess.sh  # set dataset=ere, need_ere_preprocess=true
```

- Check `/preprocess/ace(or aceplus/ere)` for more preprocessing details.

## Training and Evaluation
1. Configure runtime environment
```
pip install -r requirements.txt
```

2. train and evaluate
```
bash run.sh
```

- Check `/scripts` for more training configuration.

## Reference
```
@inproceedings{yang2024scented,
  title={Scented-EAE: Stage-Customized Entity Type Embedding for Event Argument Extraction},
  author={Yang, Yu and Guo, Jinyu and Shuang, Kai and Mao, Chenrui},
  booktitle={Findings of the Association for Computational Linguistics ACL 2024},
  pages={5222--5235},
  year={2024}
}
```

