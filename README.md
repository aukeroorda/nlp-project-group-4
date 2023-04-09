# 1. nlp-project-group-4

## 1.1. Table of Contents
- [1. nlp-project-group-4](#1-nlp-project-group-4)
  - [1.1. Table of Contents](#11-table-of-contents)
  - [1.2. General](#12-general)
  - [1.3. The process](#13-the-process)
  - [1.4. Overview of the repository](#14-overview-of-the-repository)

## 1.2. General
This repository contains the code for an NLP project focussing on `morphology`. Specifically, `plural inflection` in `German` and `Turkish`. The project was made as part of a course at the [University of Groningen](https://www.rug.nl/).

We make use of a `ByT5` character level model taken from [Huggingface](https://huggingface.co/) to gauge its inflectional capabilities.

The necessary `requirements.txt` can be found in the root folder.

## 1.3. The process
For both languages (separately) we do the following:

1. Compare a `pre-trained ByT5 model finetuned on language data` with a `ByT5 trained on language data from scratch`.

   - Compare the learning curves of the models
   - Look into the types of errors

2. Analyse the `finetuned ByT5 models` using `feature attribution methods` to see whether there are any patterns regarding the importance of input characters for the output characters.

   - For feature attribution we make use of the [`Inseq`](https://github.com/inseq-team/inseq) library.

## 1.4. Overview of the repository
```bash
 ./
├──  byt5.ipynb
├──  byt5_learning_curves_finetuning.ipynb
├──  byt5_learning_curves_scratch.ipynb
├──  byt5_model.py
├──  create_plots.ipynb
├──  data/
│   ├──  deu.dev
│   ├──  deu.gold
│   ├──  deu.test
│   ├──  deu_100.train
│   ├──  deu_200.train
│   ├──  deu_300.train
│   ├──  deu_400.train
│   ├──  deu_500.train
│   ├──  deu_600.train
│   ├──  tur.dev
│   ├──  tur.gold
│   ├──  tur.test
│   ├──  tur_large.train
│   └──  tur_small.train
├──  inseq_analysis.ipynb
├──  plot_utils.py
├──  README.md
```

The repository directory structure consists of a `root` directory with a `data/` directory.

Within the root we find the fundamental `$CODE.py` files for the larger codebase used within the `$ANALYSIS.ipynb`. The `$ANALYSIS.ipynb` files contain the base analyses for the (1) model comparison (byt5_\*) and (2) Inseq analyses (inseq_\*).

The `data/` directory contains the datasets that are used for training (`data/*.train`) and generation (`data/*.gold`) for the analyses.