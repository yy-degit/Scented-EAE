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

### ERE to OneIE format
The `prepreocessing/process_ere.py` script converts raw ERE datasets (LDC2015E29,
LDC2015E68, LDC2015E78, LDC2015E107) to the format used by OneIE. 

```
python preprocessing/process_ere.py -i <INPUT_DIR>/data -o <OUTPUT_DIR>
  -b bert-large-cased -c <BERT_CACHE_DIR> -l english -d normal
```

Arguments:
- -i, --input: Path to the input directory (`data` folder in your ERE package).
- -o, --output: Path to the output directory.
- -b, --bert: Bert model name.
- -c, --bert_cache_dir: Path to the BERT cache directory.
- -d, --dataset: Dataset type: normal, r2v2, parallel, or spanish.
- -l, --lang: Language (options: english, spanish).

This script only supports:
- LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V1
- LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2
- LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2
- LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2
- LDC2015E107_DEFT_Rich_ERE_Spanish_Annotation_V2