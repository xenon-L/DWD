import torch
import torch as th
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss, ClassificationReport
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import logging
from torch.optim import lr_scheduler
from model import BertGCN
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--bert_init', type=str, default='roberta-base',
                    choices=['roberta-base', 'electra-base-discriminator', 'bert-base-uncased', 'xlnet-base-cased'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--dataset', default='ohsumed', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])
parser.add_argument('--checkpoint_dir', default=None,
                    help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=200,
                    help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
parser.add_argument('--heads', type=int, default=8, help='the number of attentionn heads for gat')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gcn_lr', type=float, default=1e-3)
parser.add_argument('--bert_lr', type=float, default=1e-5)
parser.add_argument('--temperature', type=float, default=0.02)
parser.add_argument('--keep_ratio', type=float, default=0.2)
parser.add_argument('--alpha', type=float, default=1.0)

args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr
temperature = args.temperature
keep_ratio = args.keep_ratio
alpha = args.alpha

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
gpu = th.device('cuda:0')

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))
# Model

# Data Preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)


adj_f = adj.copy()
adj_fc = adj.copy()

if not sp.issparse(adj_fc):
    adj_fc = sp.csr_matrix(adj_fc)

vocab_size = adj_fc.shape[0] - test_size

adj_f_1 = adj_fc[train_size:vocab_size, :train_size]  # (train_size, train_size)
adj_f_2 = adj_fc[train_size:vocab_size, vocab_size:]  # (test_size, train_size)

adj_f_3 = adj_fc[:train_size, train_size:vocab_size]  # (train_size, test_size)
adj_f_4 = adj_fc[vocab_size::, train_size:vocab_size]  # (test_size, test_size)

adj_1 = sp.hstack([adj_f_1, adj_f_2])
adj_2 = sp.vstack([adj_f_3, adj_f_4])

adj_result = adj_2 @ adj_1

def pagerank_based_sparsify(adj_matrix, keep_ratio=0.1, alpha=0.85):
    n = adj_matrix.shape[0]
    adj_matrix = adj_matrix.astype(float)
    out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1
    adj_matrix = sp.diags(1.0 / out_degree) @ adj_matrix
    pr = np.ones(n) / n
    for _ in range(100):
        pr = alpha * (adj_matrix.T @ pr) + (1 - alpha) / n
    sorted_indices = np.argsort(pr)[-int(n * keep_ratio):]
    adj_coo = adj_matrix.tocoo()
    mask_row = np.isin(adj_coo.row, sorted_indices)
    mask_col = np.isin(adj_coo.col, sorted_indices)
    mask = mask_row & mask_col
    new_adj = sp.coo_matrix((adj_coo.data[mask], (adj_coo.row[mask], adj_coo.col[mask])),
                            shape=adj_matrix.shape)
    return new_adj.tocsr()

# 应用示例
adj_sampled = pagerank_based_sparsify(adj_result, keep_ratio)

# adj_sampled = adj_result

adj_f[:train_size, :train_size] = adj_sampled[:train_size, :train_size]
adj_f[vocab_size:, :train_size] = adj_sampled[train_size:, :train_size]
adj_f[:train_size, vocab_size:] = adj_sampled[:train_size, train_size:]
adj_f[vocab_size:, vocab_size:] = adj_sampled[train_size:, train_size:]

adj_f[train_size:vocab_size, :train_size] = 0
adj_f[train_size:vocab_size, vocab_size:] = 0
adj_f[:train_size, train_size:vocab_size] = 0
adj_f[vocab_size:, train_size:vocab_size] = 0

'''
adj: n*n sparse adjacency matrix
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
'''

# compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# instantiate model according to class number
if gcn_model == 'gcn':
    model = BertGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout)

if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])

# load documents and compute input encodings
corpse_file = './data/corpus/' + dataset + '_shuffle.txt'
with open(corpse_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')


def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    #     print(input.keys())
    return input.input_ids, input.attention_mask


#
input_ids, attention_mask = encode_input(text, model.tokenizer)

seq_length = input_ids.shape[1]
position_ids = th.arange(seq_length, dtype=th.long).expand(input_ids.shape)

input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat(
    [attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])
position_ids = th.cat(
    [position_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), position_ids[-nb_test:]])

y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

doc_mask = train_mask + val_mask + test_mask

adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'], g.ndata['position_ids'] = \
    input_ids, attention_mask, position_ids
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

adj_norm_f = normalize_adj(adj_f + sp.eye(adj_f.shape[0]))
g_f = dgl.from_scipy(adj_norm_f.astype('float32'), eweight_name='edge_weight')
g_f.ndata['input_ids'], g_f.ndata['attention_mask'], g_f.ndata['position_ids'] = \
    input_ids, attention_mask, position_ids
g_f.ndata['label'], g_f.ndata['train'], g_f.ndata['val'], g_f.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g_f.ndata['label_train'] = th.LongTensor(y_train)
g_f.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

logger.info('graph information:')
logger.info(str(g))

logger.info('graph_fine information:')
logger.info(str(g_f))

# create index loader
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node - nb_test, nb_node, dtype=th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0, epsilon=1e-8, neg_pos_ratio=4):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        if batch_size == 0:
            return torch.tensor(0.0, device=z_i.device)

        z = torch.cat((z_i, z_j), dim=0)

        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / (self.temperature + self.epsilon)

        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat((sim_ij, sim_ji), dim=0)
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        negatives_all = sim_matrix[mask].view(2 * batch_size, -1)

        num_negatives = positives.shape[0] * self.neg_pos_ratio
        negatives = []
        for i in range(2 * batch_size):
            neg_indices = torch.multinomial(torch.ones(negatives_all.shape[1]), num_negatives // (2 * batch_size),
                                            replacement=True)
            negatives.append(negatives_all[i, neg_indices])
        negatives = torch.stack(negatives)
        logits = torch.cat((positives.unsqueeze(1), negatives), dim=1)

        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)

        loss_weights = torch.ones(logits.shape[1], device=z_i.device)
        loss_weights[0] = self.neg_pos_ratio
        loss = F.cross_entropy(logits, labels, weight=loss_weights)
        return loss

def generate_positive_pairs(g, train_mask, perturbation_rate=0.1):
    device = g.device
    g1 = dgl.graph(([], []), num_nodes=g.number_of_nodes(), device=device)
    g1.ndata.update({k: v.clone().to(device) for k, v in g.ndata.items()})
    g1.add_edges(g.edges()[0].to(device), g.edges()[1].to(device),
                 data={'edge_weight': g.edata['edge_weight'].clone().to(device)})

    train_mask = train_mask.to(device)

    num_edges = g.num_edges()
    num_perturb = int(num_edges * perturbation_rate)
    edge_indices = torch.randperm(num_edges, device=device)[:num_perturb]
    g1.remove_edges(edge_indices)

    new_src = torch.randint(0, g.number_of_nodes(), (num_perturb,), device=device)
    new_dst = torch.randint(0, g.number_of_nodes(), (num_perturb,), device=device)
    g1.add_edges(new_src, new_dst)

    mask_nodes = torch.randperm(g.number_of_nodes(), device=device)[:int(0.1 * g.number_of_nodes())]
    mask_nodes = mask_nodes[train_mask[mask_nodes]]
    if len(mask_nodes) > 0:
        mask_feats = torch.randint(0, g.ndata['cls_feats'].shape[1], (len(mask_nodes),), device=device)
        mean_values = torch.mean(g.ndata['cls_feats'], dim=0, keepdim=True)
        g1.ndata['cls_feats'][mask_nodes, mask_feats] = mean_values[:, mask_feats]

    permute_nodes = torch.randperm(g.number_of_nodes(), device=device)
    permute_nodes = permute_nodes[train_mask[permute_nodes]]
    original_features = g1.ndata['cls_feats'][permute_nodes].clone()
    g1.ndata['cls_feats'][permute_nodes] = original_features[torch.randperm(len(permute_nodes))]

    return g1


# Training
def update_feature():
    global model, g, doc_mask

    dataloader = Data.DataLoader(
        Data.TensorDataset(
            g.ndata['input_ids'][doc_mask],
            g.ndata['position_ids'][doc_mask],
            g.ndata['attention_mask'][doc_mask]
        ),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, position_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)[0][
                     :, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g


optimizer = th.optim.Adam([
    {'params': model.bert_model.parameters(), 'lr': bert_lr},
    {'params': model.classifier.parameters(), 'lr': bert_lr},
    # {'params': model.gin.parameters(), 'lr': gcn_lr},
    {'params': model.gcn.parameters(), 'lr': gcn_lr},
], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

def train_step(engine, batch):
    # global model, g, g_f, optimizer, model_l, temperature, alpha
    # global model, g, g_f, optimizer, model_l, temperature
    global model, g, g_f, optimizer, temperature
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx,) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    train_mask_g1 = g.ndata['train'].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)

    contrastive_loss_fn = ContrastiveLoss(temperature)

    g1 = generate_positive_pairs(g, train_mask_g1)
    # g1 = dgl.add_self_loop(g1)

    g1 = g1.to(gpu)
    g_f = g_f.to(gpu)

    # --------- Prevent contrastive loss from triggering backward issues ---------
    with th.no_grad():
        z_y = y_pred.detach()
        z_j = model(g_f, idx)[train_mask].detach()
        z_i = model(g1, idx)[train_mask].detach()
        contrastive_loss_cf = contrastive_loss_fn(z_y, z_j)
        contrastive_loss_at = contrastive_loss_fn(z_y, z_i)

    # z_j = model_l(g_f, idx)[train_mask]
    # z_i = model_l(g1, idx)[train_mask]
    #
    # # print(y_pred.requires_grad)
    #
    # # loss = F.nll_loss(y_pred, y_true)
    # contrastive_loss_cf = contrastive_loss_fn(y_pred, z_j)
    # contrastive_loss_at = contrastive_loss_fn(y_pred, z_i)
    # # contrastive_loss = contrastive_loss_fn(z_i, z_j)

    # total_loss = loss + alpha * contrastive_loss_cf + (1-alpha) * contrastive_loss_at
    total_loss = loss + contrastive_loss_cf + contrastive_loss_at
    # total_loss = loss + contrastive_loss_cf
    # total_loss = loss + contrastive_loss_at

    # print(y_pred.requires_grad)
    # print(z_j.requires_grad)
    # print(z_i.requires_grad)

    total_loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = total_loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1

    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx,) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true


evaluator = Engine(test_step)
metrics = {
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss()),
    'report': ClassificationReport(output_dict=True)
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll, test_report = metrics["acc"], metrics["nll"], metrics["report"]
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    # logger.info(test_report)
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                # 'gcn': model.gin.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc


log_training_results.best_val_acc = 0
g = update_feature()
trainer.run(idx_loader, max_epochs=nb_epochs)


