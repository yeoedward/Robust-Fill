# PyTorch implementation of Robust Fill

Original Paper: https://arxiv.org/pdf/1703.07469.pdf

## TODO

- Embedding of characters and programs
- Add attention
- Add support for multiple I/O examples
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
python nn.py
```

Run unit tests:

```
python -m unittest
```

Lint:

```
flake8
```
