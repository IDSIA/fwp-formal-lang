## Formal Language Recognition with Recurrent and Self-Referential Linear Transformers

This is the official code repository for the paper:

[Practical Computational Power of Linear Transformers and Their Recurrent and Self-Referential Extensions (EMNLP 2023, short paper)]() (Link coming soon)

This repository was originally forked from [IDSIA/recurrent-fwp
/algorithmic](https://github.com/IDSIA/recurrent-fwp/tree/master/algorithmic), and contains code from [IDSIA/modern-srwm](https://github.com/IDSIA/modern-srwm) (SRWM implementation) and [IDSIA/rtrl-elstm](https://github.com/IDSIA/rtrl-elstm) (e-LSTM implementation).

[`fast_transformers`](https://github.com/IDSIA/fwp-formal-lang/tree/main/fast_transformers) directory contains code taken and modified from [idiap/fast-transformers](https://github.com/idiap/fast-transformers/tree/master/fast_transformers/causal_product) (a specific license file is included).

## Requirements
* PyTorch. We used PyTorch `2.0.1+cu117` with Python 3.9 or 3.11
* Ninja to compile custom CUDA kernels (`pip install ninja`)
* Optionally: wandb for monitoring jobs (you can enable it by adding the `--use_wandb` flag) (We did not use/need this option for our experiments)

## Data
We obtained the datasets from [satwik77/Transformer-Formal-Languages](https://github.com/satwik77/Transformer-Formal-Languages) (for all datasets except `reset Dyck-1` we downloaded their pre-generated datasets; for `reset Dyck-1` we generated a dataset using their code).

The exact set of datasets we used can also be downloaded from this [Google Drive link](https://drive.google.com/file/d/1eyNGFJpw4lJEbq5HAs4SseaVAg151apR/view?usp=sharing).

## Training
A generic script to train a model is provided below.

- `TORCH_EXTENSIONS_DIR` is where the compiled custom CUDA kernel will be stored (this way, you will not need to re-compile them every time).
- the script below assumes the dataset to be available under `data/${task}`. The expected files are: `train_50.src`,  `train_50.tgt`, `valid_50.src`,  `valid_50.tgt`, `test_50.src`, and `test_50.tgt` (`*.src`/`*.tgt` files containing the input/output sequences; "test" corresponds to "valid2"; see examples in our files provided above, and see `main.py` if further details are needed).

`modeltype` specifies the model type by its ID. 
The IDs of the models used in the paper are as follows:
* `0`: LSTM
* `1`: (regular) Transformer
* `2`: DeltaNet
* `7`: Recurrent Delta
* `8`: Linear Transformer
* `10`: e-LSTM (called "Quasi-LSTM" in the code)
* `12`: SRWM

The corresponding CUDA implementations can be found in the following directories:
* `fast_transformers`: vanilla Linear Transformer
* `fast_weight`: DeltaNet
* `rec_update_fwm_tanh`: Recurrent Delta
* `self_ref_v0`: SRWM

When using our data files, we used `task` to specify the language to learn, which has to be consistent with the directory name:
* `parity`
* `aa-star`
* `abab-star`
* `counter-2`
* `counter-3`
* `shuffle-2`
* `dyck-1`
* `reset_dyck`

Otherwise, the task is implicitly specified through specification of the data directory `--data_dir` (see below)

`--level` flag can be ignored/removed. We initially thought about having various "levels" for each tasks, but in the end, we did not implement such an option, and we always have `level=50`.
Nevertheless, we should not set it to other values than 50, as the dataset file names have to contain the `level` value (see file names above).

Other flags/parameters should be self-explanatory (see hyper-parameter tables in the appendix of the paper).

```
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export TORCH_EXTENSIONS_DIR="arbitrary_path/torch_extensions/formal_lang"

task=
modeltype=
lay=
size=
head=
ff=
batch=
lr=

DATA_DIR='data/'${task}

python main.py \
  --data_dir ${DATA_DIR} \
  --level 50 \
  --model_type ${modeltype} \
  --num_layer ${lay} \
  --hidden_size ${size} \
  --n_head ${head} \
  --ff_factor ${ff} \
  --dropout 0.0 \
  --batch_size ${batch} \
  --learning_rate ${lr} \
  --seed 1 \
  --grad_cummulate 1 \
  --num_epoch 300 \
  --report_every 50 \
  --project_name "2023--formal-lang" \
  --remove_pos_enc
```

**NB:**
* In training logs, `valid` corresponds to the validation set with the same distribution as the training set ("Bin0" in the paper)
while `valid2` is the validation set with longer sequences ("Bin1" in the paper).
* A training run will stop when either 100% accuracy is achieved on the valid2 dataset or the maximum number of training epoch `num_epoch` is reached.
* In logs, `no-op acc` should be ignored (this used to be originally relevant for other tasks used in [IDSIA/recurrent-fwp
/algorithmic](https://github.com/IDSIA/recurrent-fwp/tree/master/algorithmic))
* Our final results are reported by consistently testing a single seed (`seed=1`) for each configuration described in the appendix.
However, it is not impossible that certain good configurations still "fail" for some seeds; if some of the good configurations we report happen to fail in your setting (i.e., not achieving 100% accuracy on valid2 using the script above), we recommend trying other seeds.

## BibTex
```
@inproceedings{irie2023practical,
      title={Practical Computational Power of Linear Transformers and Their Recurrent and Self-Referential Extensions}, 
      author={Kazuki Irie and R\'obert Csord\'as and J\"urgen Schmidhuber},
      booktitle={Proc. Conf. on Empirical Methods in Natural Language Processing (EMNLP)},
      address={Sentosa, Singapore},
      year={2023}
}
```
