# SSF
Code for "Interpretable Subgraph Feature Extraction for Hyperlink Prediction." ICDM 2023


## Required Packages

The following environment has been tested.
```
pytorch == 1.9.0
torch_geometric == 2.0.1
numpy == 1.21.5
scipy == 1.9.3
scikit-learn == 1.1.3
argparse == 1.5.2
```



## Configuration

### Datasets

ARB datasets --- under a supervised classification setting

Reaction datasets --- under a positive unlabeled setting

We use 20% of training data to form the validation set.


### Setting Alpha

Two intermediate states $alpha = 0$ and $alpha = 1$ are selected by default.

### Setting MLP
```
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--batch_size', type=int, default = 32)
```


## Quick Start (to reproduce the results in Table III.)
```
python main_arb_5fold.py --data=contact-high-school --epoch_num=500 --walk_len=6 --num_hops=3
python main_arb_5fold.py --data=contact-primary-school --epoch_num=1000 --walk_len=8 --num_hops=2
python main_arb_5fold.py --data=email-Enron --epoch_num=1000 --walk_len=9 --num_hops=2
python main_arb_5fold.py --data=email-Eu --epoch_num=1000 --walk_len=8 --num_hops=2
python main_arb_5fold.py --data=DAWN --epoch_num=1000 --walk_len=3 --num_hops=2

python main_reaction_5fold.py --data=iAB_RBC_283 --epoch_num=300 --walk_len=5 --num_hops=2
python main_reaction_5fold.py --data=iAF692 --epoch_num=300 --walk_len=6 --num_hops=3
python main_reaction_5fold.py --data=iHN637 --epoch_num=500 --walk_len=6 --num_hops=3
python main_reaction_5fold.py --data=iAF1260b --epoch_num=1500 --walk_len=7 --num_hops=2
python main_reaction_5fold.py --data=iJO1366 --epoch_num=1500 --walk_len=8 --num_hops=3
```

## Note

The feature extraction results will be saved in the 📁 walk_profile.

For the subgraph feature extraction procedure, if the extract subgraph is small, using cpu is faster, and vice versa. (See the function obtain_walk_profile in utils.py and utils_reaction.py)







