# PyTorch implementation of Robust Fill

Original Paper: https://arxiv.org/pdf/1703.07469.pdf

## Instructions

Set up environment:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Train neural net:

```
python train.py --mode full
```

For testing purposes:

```
python train.py --mode easy
```

Run profiler:

```
python train.py --mode profile
```

Run unit tests:

```
python -m unittest
```

Lint:

```
flake8
```
