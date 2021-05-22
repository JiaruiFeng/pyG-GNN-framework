"""
Jiarui Feng
The file for evaluating a graph classifier
"""

from models import GraphClassifier,make_gnn_layer,GNN
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch_geometric.data import DataListLoader
from torch.utils.data import DataLoader
import utils
from args import get_test_args
from collections import OrderedDict
from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch_geometric.nn import DataParallel


def get_model(log,args):

    gnn_layer=make_gnn_layer(gnn_type=args.GNN_type,
                             emb_dim=args.hidden_size,
                             aggr=args.aggr,
                             eps=args.eps,
                             train_eps=args.train_eps,
                             head=args.num_head,
                             negative_slope=args.negative_slope,
                             num_edge_type=args.num_edge_type)

    GNN_model=GNN(input_dim=args.input_size,
                  output_dim=args.hidden_size,
                  num_layer=args.num_GNN,
                  gnn_layer=gnn_layer,
                  JK=args.JK,
                  norm_type=args.norm_type,
                  edge_drop_prob=args.edge_drop_prob,
                  drop_prob=args.drop_prob)

    model=GraphClassifier(embedding_model=GNN_model,
                          emb_dim=args.hidden_size,
                          pooling_method=args.pooling_method,
                          num_tasks=args.num_tasks,
                          sort_pooling_ratio=args.sort_pooling_ratio)

    #If use multiple gpu, torch geometric model must use DataParallel class
    if args.parallel:
        model = DataParallel(model, args.gpu_ids)
    else:
        model=nn.DataParallel(model,args.gpu_ids)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = utils.load_model(model, args.load_path+"_graph_classifier.pth.tar", args.gpu_ids)

    else:
        step = 0

    return model,step



def main(args):
    # Set up logging and devices
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, type="evaluate")
    log = utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = utils.get_available_devices()
    #whether to use multiple gpu for training
    args.parallel=len(args.gpu_ids)>1
    args.batch_size *= max(1, len(args.gpu_ids))
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')


    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get your model
    log.info('Building model...')
    model,step=get_model(log,args)
    model.to(device)
    model.eval()


    # Get data loader
    log.info('Building dataset...')
    train_expression, train_y, val_expression, val_y, test_expression, test_y, \
    A, gene_feature, gene_list=utils.make_ad_dataset(args.ad_data_path,args.group,args.val_ratio,
                                                     args.test_ratio,args.seed)

    test_dataset = utils.LoadTrainDataset(test_expression,test_y,A,gene_feature,gene_list)

    #if use multiple gpu, torch geometric model must use DataListLoader
    if args.parallel:
        test_loader = DataListLoader(test_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.num_workers)

    else:
        test_loader = DataLoader(test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      collate_fn=utils.collate_fn)




    # Train
    log.info('Evaluating...')
    loss_meter = utils.AverageMeter()
    metrics_meter=utils.MetricsMeter()

    with torch.no_grad(), \
        tqdm(total=len(test_loader.dataset)) as progress_bar:
        for batch_data in test_loader:
            # Setup for forward

            if args.parallel:
                batch_size = len(batch_data)
            else:
                batch_data = batch_data.to(device)
                batch_size = batch_data.num_graphs
            predict = model(batch_data)
            predict = torch.log_softmax(predict, dim=-1)
            if args.parallel:
                batch_y = torch.cat([data.y.view(-1) for data in batch_data]).to(predict.device).long()
            else:
                batch_y = batch_data.y.long()
            # loss re-weighting in each batch
            weight_vector = torch.zeros([args.num_tasks], device=device)
            for i in range(args.num_tasks):
                n_samplei = torch.sum(batch_y == i).item()
                if n_samplei > 0:
                    weight_vector[i] = batch_size / (n_samplei * args.num_tasks)
            loss = F.nll_loss(predict, batch_y, weight_vector)
            # update meter
            loss_meter.update(loss.item(), batch_size)
            metrics_meter.update(predict.exp()[:, 1].cpu(), batch_y.cpu())
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(Loss=loss_meter.avg)

        metrics_result = metrics_meter.return_metrics()

        results_list = [
            ('Loss', loss_meter.avg),
            ('Accuracy', metrics_result["Accuracy"]),
            ('Recall', metrics_result["Recall"]),
            ('Precision', metrics_result["Precision"]),
            ('Specificity', metrics_result["Specificity"]),
            ('F1', metrics_result["F1"]),
            ("AUC", metrics_result["AUC"])
        ]
        results = OrderedDict(results_list)
        # Log to console
        results_str = ', '.join(f'{k}: {v:05.5f}' for k, v in results.items())
        log.info(f'Dev {results_str}')



if __name__ == '__main__':
    main(get_test_args())


