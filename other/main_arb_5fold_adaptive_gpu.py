import torch
import numpy as np
import argparse
import os.path
from utils_mu_gpu import prepare_data
from model import MLP
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import set_start_method

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2none(v):
    if v.lower()=='none':
        return None
    else:
        return str(v)
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



parser = argparse.ArgumentParser(description='Link Prediction with Walk-Pooling')
#Dataset


# small = ['iAB_RBC_283', 'iAF692', 'iAF1260b', 'iHN637', 'iIT341', 'iJO1366']




parser.add_argument('--data', type=str,  default='email-Enron', help='graph name')
parser.add_argument('--data_split',type=str, default='0',
                    help='If use-splitted is true, choose one of splitted data')

#training/validation/test divison and ratio
parser.add_argument('--val_ratio', type=float, default=0.2,
                    help='ratio of  the validation set')
#Drnl feature in the SEAL paper

#Model and Training
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--walk_len', type=int, default=6, help='cutoff in the length of walks')
parser.add_argument('--num_hops', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--embedding_dim', type=int, default= 16,
                    help='Dimension of the initial node representation, default: 32)')
parser.add_argument('--batch_size', type=int, default = 32)
parser.add_argument('--epoch_num', type=int, default= 1000)

parser.add_argument('--log', type=str, default=None,
                    help='log by tensorboard, default is None')

args = parser.parse_args()


def RunExp(args):
    print('<<Begin generating training data>>')
    train_data, val_data, test_data, train_lb, val_lb, test_lb = prepare_data(args)
    print('<<Complete generating training data>>')

    print ("-"*42+'Model and Training'+"-"*45)
    print ("{:<13}|{:<13}|{:<13}|{:<8}|{:<13}|{:<15}"\
        .format('Learning Rate','Weight Decay','Batch Size','Epoch',\
            'Walk Length','Hidden Channels'))
    print ("-"*105)

    print ("{:<13}|{:<13}|{:<13}|{:<8}|{:<13}|{:<15}"\
        .format(args.lr,args.weight_decay, str(args.batch_size),\
            args.epoch_num,args.walk_len, args.hidden_channels))
    print ("-"*105)


    walk_len = args.walk_len
    hidden_channels=args.hidden_channels
    lr=args.lr
    weight_decay=args.weight_decay

    torch.cuda.empty_cache()

    args.num_features = args.embedding_dim


    torch.cuda.empty_cache()
    print("Dimention of features after concatenation:",args.num_features)
    set_random_seed(args.seed)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # torch.cuda.manual_seed_all(1)

    classifier = MLP(train_data.size(1), args.hidden_channels).to(device)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    train_lb = train_lb.to(device)
    val_lb = val_lb.to(device)
    test_lb = test_lb.to(device)

    def train(data, lbs):
        classifier.train()
        torch.cuda.empty_cache()
        out = classifier(data)
        loss = criterion(out.view(-1), lbs.view(-1).to(torch.float))
        optimizer_classifier.zero_grad()
        loss.backward()
        optimizer_classifier.step()
        loss_epoch = loss.item()
        return loss_epoch

    def test(data, lbs):
        classifier.eval()
        with torch.no_grad():
            out = classifier(data)
            loss = criterion(out.view(-1), lbs.view(-1).to(torch.float))
            scores = out.cpu().clone().detach()
            tpred = scores.flatten()
            num_predictions = int(sum(lbs))
            cut = np.partition(tpred, -num_predictions)[-num_predictions]
            tpred = torch.Tensor(tpred)
            pred = torch.where(tpred >= cut, torch.ones_like(tpred), torch.zeros_like(tpred)).view(-1)

            # pred = torch.sigmoid(scores)
            # pred = torch.where(pred>0.6, torch.ones_like(pred), torch.zeros_like(pred)).view(-1)
            f1 = f1_score(np.array(lbs.cpu().tolist()), np.array(pred.cpu().tolist()))
            auc = roc_auc_score(np.array(lbs.cpu().tolist()), scores)
            ap = average_precision_score(np.array(lbs.cpu().tolist()), scores)
            return f1, auc, ap, loss.item()


    best_from_val = 0
    best_val_f1 = 0

    for epoch in range(0, args.epoch_num):
        train_loss = train(train_data, train_lb)
        frac = sum(train_lb)/len(train_lb)
        val_f1, val_auc, val_ap, val_loss = test(val_data, val_lb)
        test_f1, test_auc, test_ap, test_loss = test(test_data, test_lb)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_from_val = test_f1
            best_auc = test_auc
            best_ap = test_ap
        if epoch % 50 ==0:
            print(f'Epoch: {epoch:03d}, Loss : {train_loss:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}, Test f1: {test_f1:.4f}, Picked f1: {best_from_val:.4f}')
    return best_from_val,best_auc


if __name__ == '__main__':

    f1_result = []
    auc_result = []
    # range(5)
    for data_split in range(5):
        args.data_split = str(data_split)
        f1,auc = RunExp(args)
        argsDict = args.__dict__
        tf1 = round(f1,3)
        auc1 = round(auc,3)

        f1_result.append(tf1)
        auc_result.append(auc1)

    mean_f1 = np.average(f1_result)
    mean_f1 = round(mean_f1, 3)
    std_f1 = np.std(f1_result)
    std_f1 = round(std_f1, 3)

    mean_auc = np.average(auc_result)
    mean_auc = round(mean_auc, 3)
    std_auc = np.std(auc_result)
    std_auc = round(std_auc, 3)

    f1_result = list(map(str, f1_result))
    auc_result = list(map(str, auc_result))
    file_out = args.data + '_' + str(mean_f1) + '_' + str(std_f1) + '_' + str(mean_auc) + '_' + str(std_auc) + '.txt'

    with open(file_out, 'w') as f:
        f.write("mean and std of F1 and AUC: ")
        f.write(str(mean_f1)+'('+str(std_f1)+')')
        f.write('\n')
        f.write(str(mean_auc)+'('+str(std_auc)+')')
        f.write('\n')
        f.write("F1 results: ")
        f.write('\n'.join(f1_result))
        f.write('\n')
        f.write("AUC results: ")
        f.write('\n'.join(auc_result))
        f.write('\n')

        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    f.close()