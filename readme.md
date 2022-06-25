# STECL

This repo contains the data and code for our paper [STECL: A Multi-level Contrastive Learning Framework for Sentiment Triplet Extraction].


## Requirements

Pls note that some packages (such as transformers) are under highly active development, so we highly recommend you to install the specified version of the following packages:
- transformers==4.0.0
- sentencepiece==0.1.91
- pytorch_lightning==0.8.1
- nlpaug==1.1.7
- pandas==1.1.5
- editdistance==0.6.0
- nltk==1.1.7
pip install -r requirements.txt

## Quick Start

- Set up the environment as described in the above section
- Download the pre-trained T5-base model (you can also use larger versions for better performance depending on the availability of the computation resource), put it under the folder `T5-base`.
  - You can also skip this step and the pre-trained model would be automatically downloaded to the cache in the next step
- Run command `sh run.sh`, which runs the `ASTE` task on the `rest14` dataset.



## Detailed Usage
We conduct experiments on two STE tasks with four datasets in the paper, you can change the parameters in `run.sh` to try them:
```
python main.py --task aste \
            --dataset rest14 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --element all
            --train_batch_size 16 \
            --gradient_accumulation_steps 2 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 30 \
            --cl=True \
            --T=0.07 
```
- `$task` refers to one of the STE task in [`aste`, `acsd`] 
- `$dataset` refers to one of the four datasets in [`laptop14`, `rest14`, `rest15`, `rest16`]
- `$cl` refers to use contrastive learning to optimize
More details can be found in the paper and the help info in the `main.py`.

