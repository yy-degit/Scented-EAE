# OneIE
Forked from *OneIE: A Joint Neural Model for Information Extraction with Global Features*

OneIE v0.4.8

# Requirements

Python 3.7
Python packages
- PyTorch 1.0+ (Install the CPU version if you use this tool on a machine without GPUs)
- transformers 3.0.2 (It seems using transformers 3.1+ may cause some model loading issue)
- tqdm
- lxml
- nltk

## Pre-processing

### ACE2005 to OneIE format
The `prepreocessing/process_ace.py` script converts raw ACE2005 datasets to the
format used by OneIE. Example:

```
python preprocessing/process_ace.py -i <INPUT_DIR>/LDC2006T06/data -o <OUTPUT_DIR>
  -s resource/splits/ACE05-E -b bert-large-cased -c <BERT_CACHE_DIR> -l english
```

Arguments:
- -i, --input: Path to the input directory (`data` folder in your LDC2006T06
  package).
- -o, --output: Path to the output directory.
- -b, --bert: Bert model name.
- -c, --bert_cache_dir: Path to the BERT cache directory.
- -s, --split: Path to the split directory. We provide document id lists for all
  datasets used in our paper in `resource/splits`.
- -l, --lang: Language (options: english, chinese).