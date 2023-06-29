# SSF
Code for "Interpretable Subgraph Feature Extraction for Hyperlink Prediction."


## Required Packages

The following environment was tested.
```
pytorch == 1.9.0
torch_geometric == 2.0.1
numpy == 1.21.5
scipy == 1.9.3
sklearn 
argparse 
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


## Quick Start
```
python main_arb_5fold.py --data=contact-high-school --epoch_num=500 --walk_len=6 --num_hops=2
python main_arb_5fold.py --data=contact-primary-school --epoch_num=1000 --walk_len=6 --num_hops=2
python main_arb_5fold.py --data=email-Enron --epoch_num=1000 --walk_len=6 --num_hops=2
python main_arb_5fold.py --data=email-Eu --epoch_num=1000 --walk_len=6 --num_hops=2
python main_arb_5fold.py --data=DAWN --epoch_num=1000 --walk_len=3 --num_hops=2

python main_reaction_5fold.py --data=iAB_RBC_283 --epoch_num=300 --walk_len=6 --num_hops=2
python main_reaction_5fold.py --data=iAF692 --epoch_num=300 --walk_len=6 --num_hops=2
python main_reaction_5fold.py --data=iHN637 --epoch_num=500 --walk_len=6 --num_hops=2
python main_reaction_5fold.py --data=iAF1260b --epoch_num=1500 --walk_len=6 --num_hops=2
python main_reaction_5fold.py --data=iJO1366 --epoch_num=1500 --walk_len=6 --num_hops=2
```

## Note

The feature extraction results will be saved in the 📁 walk_profile.

For the subgraph feature extraction procedure, if the extract subgraph is small, using cpu is faster, and vice versa. (See the function obtain_walk_profile in utils.py and utils_reaction.py)

In the 📂 other, we give code for adaptively analysis, ablation studies, and sensitive analysis. In general, we first fix the number of hops, set a larger walk_length (i.e., $\tau_c=10$), set a larger $\alpha$-set (i.e., from 1 to 0，with 0.1 intervals), then we can get a feature representation with a longer dimension. Further, the feature extraction results can be reused (by slicing indices).







