"""
Jiarui Feng
Arguments definition file, used to save and modify all the arguments in the project.

"""

import argparse
#set name of your project
PROJECT_NAME="GNN_training"


def get_train_args():
    """Get arguments needed in training the model."""
    parser = argparse.ArgumentParser(f'Train a model on {PROJECT_NAME}')

    add_common_args(parser)
    add_train_test_args(parser)
    add_train_args(parser)
    #Add other arguments if needed

    args = parser.parse_args()

    if args.metric_name == 'Loss':
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ("Accuracy","Recall","AUC"):
        # Best checkpoint is the one that maximizes Accuracy or recall
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args


def get_test_args():
    """Get arguments needed in testing the model or prediction."""
    parser = argparse.ArgumentParser(f'Test a trained model on {PROJECT_NAME}')

    add_common_args(parser)
    add_train_test_args(parser)
    add_test_args(parser)

    # Require mdoel load_path for testing
    args = parser.parse_args()

    if not args.load_path:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args



def add_common_args(parser):
    """Add arguments common in all files, typically file directory"""

    parser.add_argument('--ad_data_path',
                    type=str,
                    default='../data/processed_data.npz')

    parser.add_argument('--mSigDB_path',
                        type=str,
                        default='../data/gene_feature/mSigDB.json')




def add_train_test_args(parser):
    """Add arguments common to training and testing"""
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of zeroing an activation in dropout layers.')

    parser.add_argument('--edge_drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of drop out edge in each GNN layer.')

    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')


    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')

    parser.add_argument('--group',
                        type=str,
                        default='male',
                        help='group of ad scRNA data.')

    parser.add_argument('--val_ratio',
                        type=float,
                        default=.2,
                        help='ratio of validation data')

    parser.add_argument('--test_ratio',
                        type=float,
                        default=.1,
                        help='ratio of test data')


    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')

    parser.add_argument('--input_size',
                        type=int,
                        default=51,
                        help='Number of features in gene input.')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help='hidden size of model.')

    parser.add_argument('--GNN_type',
                        type=str,
                        default='GCN',
                        choices=("GCN","GAT","GIN","GraphSAGE"),
                        help='The type of GNN layer.')

    parser.add_argument('--num_GNN',
                        type=int,
                        default=3,
                        help='Number of GNN layer in model')

    parser.add_argument('--num_head',
                        type=int,
                        default=8,
                        help='Number of head in GAT layer.')

    parser.add_argument('--num_edge_type',
                        type=int,
                        default=0,
                        help='Number of type of edge feature. 0 indicate no edge feature')

    parser.add_argument('--aggr',
                        type=str,
                        default="add",
                        help='aggregation method in GNN layer, only works in GraphSAGE')


    parser.add_argument('--eps',
                        type=float,
                        default=0.,
                        help='epsilon in GIN layer.')

    parser.add_argument("--train_eps",
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether the eps in GIN layer is trainable')

    parser.add_argument('--negative_slope',
                        type=float,
                        default=0.2,
                        help='The negative slope in GAT leky relu activation function.')


    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')


    parser.add_argument('--JK',
                        type=str,
                        default="sum",
                        choices=("concat",  "last", "max", "sum","attention"),
                        help="Jump knowledge method in GNN model"
                        )

    parser.add_argument('--pooling_method',
                        type=str,
                        default="attention",
                        choices=("sum","mean","max","attention","set2set"),
                        help="Pooling method in graph classification")

    parser.add_argument('--norm_type',
                        type=str,
                        default="Batch",
                        choices=("Batch","Layer","Instance","GraphSize","Pair"),
                        help="normalization method in model")

    parser.add_argument('--num_tasks',
                        type=int,
                        default=2,
                        help="Number of type in classification")



def add_train_args(parser):
    parser.add_argument('--eval_steps',
                        type=int,
                        default=10000,
                        help='Number of steps between successive evaluations.')

    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate for gene gnn.')

    parser.add_argument('--l2_wd',
                        type=float,
                        default=3e-7,
                        help='L2 weight decay.')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=25,
                        help='Number of epochs for which to train. Negative means forever.')

    parser.add_argument('--metric_name',
                        type=str,
                        default='AUC',
                        choices=( 'Loss','AUC'),
                        help='Name of dev metric to determine best checkpoint.')

    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=10,
                        help='Maximum number of checkpoints to keep on disk.')

    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')

    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')

def add_test_args(parser):
    return

