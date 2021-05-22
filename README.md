# pyG-GNN-framework
An integrated framework for training and evaluating GNN model with torch_geometric. The framework provide code for fast build several famous GNN model for node classification and graph classification, training, evaluating, saving the model and other utilities function. The framework support GPU training and GPU parallel with DataParallel class.



### Implemented Model

1. [GCN](https://arxiv.org/abs/1609.02907) with edge attributes. Set `GNN_type=GCN`

2. [GraphSAGE](https://arxiv.org/abs/1706.02216) with edge attributes. Set `GNN_type=GraphSAGE`

3. [GAT](https://arxiv.org/abs/1710.10903) with edge attributes. Set `GNN_type=GAT`

4. [GIN](https://arxiv.org/abs/1810.00826) with edge attributes. Set `GNN_type=GIN`
5. [JumpingKnowledge](https://arxiv.org/abs/1806.03536). Set `JK`,supported methods are `sum,concat,last,max,sum,attention`
6. [DropEdge](https://openreview.net/forum?id=Hkx1qkrKPr). Set `edge_drop_prob`
7. Graph pooling methods. Set `pooling_method`, supported methods are `sum,mean,max,attention,set2set`
8. Normalization methods. Set `norm_type`, supported methods are `Batch,Layer,Instance,GraphSize,Pair`.

You can also use or implement your GNN layer with [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html).

### Data

You need to provide your own dataset loading function by revising `utils.LoadTrainDataset` class and the data path `data_path`. Notice that your class must support [DataListLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html) if you want GPU parallel training. 

### Training

run `python train_graph_classifier.py -n=graph_classifier`  for graph classification, run `python train_node_classifier -n=node_classifier` for node classification. set the number of type in classification by `num_tasks`. **Node classification and classification with tasks larger than 2 are not yet implemented.**

### Evaluating

run `python evaluate_graph_classifier.py -n=graph_classifier --load_path=/save/train/graph_classifier-01/best` for evaluating graph classification model. You can modify the load path to load curresponding model. 





