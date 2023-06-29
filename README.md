# SSF
Code for "Interpretable Subgraph Feature Extraction for Hyperlink Prediction."

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


### Quick Start
```
python main_arb_5fold.py --data=contact-high-school --epoch_num=500 --walk_len=6 --num_hops=2
python main_arb_5fold.py --data=contact-primary-school --epoch_num=1000 --walk_len=6 --num_hops=2
python main_arb_5fold.py --data=email-Enron --epoch_num=1000 --walk_len=6 --num_hops=2
python main_arb_5fold.py --data=email-Eu --epoch_num=1000 --walk_len=6 --num_hops=2
python main_arb_5fold.py --data=email-Eu --epoch_num=1000 --walk_len=3 --num_hops=2

python main_reaction_5fold.py --data=iAB_RBC_283 --epoch_num=300 --walk_len=6 --num_hops=2
python main_reaction_5fold.py --data=iAF692 --epoch_num=300 --walk_len=6 --num_hops=2
python main_reaction_5fold.py --data=iHN637 --epoch_num=500 --walk_len=6 --num_hops=2
python main_reaction_5fold.py --data=iAF1260b --epoch_num=1500 --walk_len=6 --num_hops=2
python main_reaction_5fold.py --data=iJO1366 --epoch_num=1500 --walk_len=6 --num_hops=2
```
