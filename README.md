# PyTorch implementation of Robust Fill

Original Paper: https://arxiv.org/pdf/1703.07469.pdf

## TODO

- GPU
- Beam search

## Instructions

Set up environment:

```
conda env create -f environment.yml
source activate robust_fill
```

Train neural net:

```
python train.py
```

For testing purposes:

```
python train.py --dry
```

Run unit tests:

```
python -m unittest
```

Lint:

```
flake8
```
