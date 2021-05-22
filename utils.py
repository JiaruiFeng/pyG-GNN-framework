"""
Jiarui Feng
This file contains util functions
"""

import networkx as nx
import ujson as json
import numpy as np
import torch
import os
import queue
import shutil
import logging
import tqdm
from sklearn.metrics import roc_curve,auc
import torch.utils.data as data
from torch_geometric.data import Data,Batch
from sklearn.model_selection import train_test_split

def reindex_nx_graph(G,ordered_node_list):
    """reindex the nodes in nx graph according to given ordering
    Args:
    G: nx graph object
    ordered_node_list: a list served as node ordering, the first is 0
    """
    ordered_node_dict=dict(zip(ordered_node_list,range(len(ordered_node_list))))
    return nx.relabel_nodes(G,ordered_node_dict)


def save_json(filename, obj, message=None,ascii=True):
    """Save data in JSON format
    Args:
        filename(str):name of save directory(including file name)
        obj(object):data you want to save
        message(str):anything you want to print
    """
    if message is not None:
        print(f"Saving {message}...")
    with open(filename, "w") as fh:
        json.dump(obj, fh,ensure_ascii=ascii)

def nx_to_graph_data(G,x,y):
    """convert nx graph to torch geometric Data object
    Args
    G（networkx): nx graph
    x(numpy.array): node feature for each node given index
    """
    edge_list=G.edges
    edge_index = torch.from_numpy(np.array(edge_list).T).to(torch.long)

    edge_attr=list(nx.get_edge_attributes(G,"edge_type").values())
    if len(edge_attr)>0:
        edge_attr = torch.from_numpy(np.array(edge_attr)).to(torch.long)
        return Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y)
    else:
        return Data(x=x,edge_index=edge_index,edge_attr=None,y=y)


def make_ad_dataset(ad_data_path,group,val_ratio,test_ratio,seed=234):
    """Make ad dataset
    Args:
        ad_data_path(str): file path of ad data
        group(str): which group used for training or testing, support male or female
        val_ratio(float):ratio of validation dataset
        test_ratio(float):ratio of test dataset
        seed(int): random seed
    """
    ad_data=np.load(ad_data_path,allow_pickle=True)

    if group=="female":
        positive_data=ad_data["ex_early_female"]
        negative_data=ad_data["ex_normal_female"]
    elif group=="male":
        positive_data=ad_data["ex_early_male"]
        negative_data=ad_data["ex_normal_male"]
    else:
        raise ValueError("invaild scRNA data group")

    expression_data=np.concatenate([positive_data,negative_data],axis=0)
    positive_y=np.ones([positive_data.shape[0]])
    negative_y=np.zeros([negative_data.shape[0]])
    y=np.concatenate([positive_y,negative_y],axis=0)
    train_expression,remain_expression,train_y,remain_y=\
    train_test_split(expression_data,y,test_size=(val_ratio+test_ratio),stratify=y,random_state=seed)

    val_expression,test_expression,val_y,test_y=\
    train_test_split(remain_expression,remain_y,test_size=test_ratio/(val_ratio+test_ratio),
                     stratify=remain_y,random_state=seed)
    A=ad_data["A"].tolist()
    gene_list=ad_data["gene_list"]
    gene_feature=ad_data["gene_feature"]
    return train_expression,train_y,val_expression,val_y,test_expression,test_y,\
            A,gene_feature,gene_list



class LoadTrainDataset(data.Dataset):
    """load ad dataset for training
    Args:
        expression(numpy.array):scRNA expression data
        y(numpy.array): prediction label
        A(numpy.array): adajacency matrix
        gene_feature(numpy.array): gene feature matrix
        gene_list(list): gene name list
        device(list): the device used in training
    """
    def __init__(self,expression,y,A,gene_feature,gene_list):
        super(LoadTrainDataset, self).__init__()
        self.gene_list=gene_list
        expression=torch.from_numpy(expression).float()
        self.length=expression.size(0)
        self.y=torch.from_numpy(y).long()
        G=nx.from_scipy_sparse_matrix(A)
        gene_feature=torch.from_numpy(gene_feature).float()
        data_list=[]
        for idx in range(expression.size(0)):
            expression_data=expression[idx].unsqueeze(-1)
            x=torch.cat([gene_feature,expression_data],dim=-1)
            data_list.append(nx_to_graph_data(G,x,self.y[idx].view(-1)))

        self.data_list=Batch.from_data_list(data_list)

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        return self.data_list[idx]


def collate_fn(examples):
    data_list=Batch.from_data_list(examples)
    return (data_list)




class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model_dict, metric_val, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """

        checkpoint_path = os.path.join(self.save_dir,f'step_{step}')
        for name,model in model_dict.items():
            ckpt_dict = {
                'model_name': model.__class__.__name__,
                'model_state': model.cpu().state_dict(),
                'step': step
            }

            model.to(device)
            torch.save(ckpt_dict, f"{checkpoint_path}{name}.pth.tar")
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best')
            for name in model_dict.keys():
                shutil.copy(f"{checkpoint_path}{name}.pth.tar", f"{best_path}{name}.pth.tar")

            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                for name in model_dict.keys():
                    os.remove(f"{worst_ckpt}{name}.pth.tar")
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


def get_available_devices() -> object:
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def get_save_dir(base_dir, name, type, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = type
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.

    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.

    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor



class MetricsMeter:
    """Keep track of model performance.
    """
    def __init__(self,threshold=0.5):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN= 0
        self.threshold=0.5
        self.prediction=np.array([1])
        self.label=np.array([1])

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, input,target ):
        """Update meter with new result

        Args:
            input (torch.tensor, Batch_size*1): predicted probability tensor.
            target (torch.tensor, Batch_size*1): ground true, 1 represent positive

        """
        predict=(input>self.threshold).int()
        self.TP+=(target[torch.where(predict==1)]==1).sum().item()
        self.FP += (target[torch.where(predict==1)]==0).sum().item()
        self.TN += (target[torch.where(predict==0)]==0).sum().item()
        self.FN += (target[torch.where(predict==0)]==1).sum().item()
        input=input.view(-1).numpy()
        target=target.view(-1).numpy()
        self.prediction=np.concatenate([self.prediction,input],axis=-1)
        self.label=np.concatenate([self.label,target],axis=-1)


    def return_metrics(self):
        recall=self.TP/(self.TP+self.FN+1e-30)
        precision=self.TP/(self.TP+self.FP+1e-30)
        specificity=self.TN/(self.TN+self.FP+1e-30)
        accuracy=(self.TP+self.TN)/(self.TP+self.FP+self.TN+self.FN+1e-30)
        F1=self.TP/(self.TP+0.5*(self.FP+self.FN)+1e-30)
        fpr,tpr,thresholds=roc_curve(self.label[1:],self.prediction[1:])
        AUC=auc(fpr,tpr)
        metrics_result = {'Accuracy': accuracy,
                          "Recall": recall,
                          "Precision": precision,
                          "Specificity": specificity,
                          "F1":F1,
                          "AUC":AUC,
                          "fpr":fpr,
                          "tpr":tpr,
                          "thresholds":thresholds
                          }
        return metrics_result

