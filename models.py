"""
Jiarui Feng
file contains the model used for training the classifier
"""
import torch
import torch
import torch.nn as nn
from layers import GINLayer,GCNLayer,GATLayer,GraphSAGELayer,clones
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set,global_sort_pool
from torch_geometric.nn import BatchNorm,LayerNorm,InstanceNorm,PairNorm,GraphSizeNorm
from torch_geometric.utils import dropout_adj
class GNN(nn.Module):
    """A generalized GNN framework
    Args:
        input_dim(int): the size of input feature
        output_dim(int): the size of output feature
        num_layer(int): the number of GNN layer
        gnn_layer(nn.Module): gnn layer used in GNN model
        JK(str):method of jumping knowledge, last,concat,max or sum
        norm_type(str): method of normalization, batch or layer
        drop_prob (float): dropout rate
    """
    def __init__(self,input_dim,output_dim,num_layer,gnn_layer,JK="last",norm_type="batch",
                        edge_drop_prob=0.1,drop_prob=0.1):
        super(GNN, self).__init__()
        self.num_layer=num_layer
        self.output_dim=output_dim
        self.dropout=nn.Dropout(drop_prob)
        self.edge_drop_prob=edge_drop_prob
        self.JK=JK

        if self.JK=="attention":
            self.attention_lstm=nn.LSTM(output_dim,self.num_layer,1,batch_first=True,bidirectional=True,dropout=0.)
            for layer_p in self.attention_lstm._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.xavier_uniform_(self.attention_lstm.__getattr__(p))

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.init_proj=nn.Linear(input_dim,output_dim)
        nn.init.xavier_uniform_(self.init_proj.weight.data)

        #gnn layer list
        self.gnns=clones(gnn_layer,num_layer)
        #norm list
        if norm_type=="Batch":
            self.norms=clones(BatchNorm(output_dim),num_layer)
        elif norm_type=="Layer":
            self.norms=clones(LayerNorm(output_dim),num_layer)
        elif norm_type=="Instance":
            self.norms=clones(InstanceNorm(output_dim),num_layer)
        elif norm_type=="GraphSize":
            self.norms=clones(GraphSizeNorm(),num_layer)
        elif norm_type=="Pair":
            self.norms=clones(PairNorm(),num_layer)
        else:
            raise ValueError("Not supported norm method")

    def forward(self,*argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        #initial projection
        x=self.init_proj(x)

        #forward in gnn layer
        h_list=[x]
        for l in range(self.num_layer):
            edge_index,edge_attr=dropout_adj(edge_index,edge_attr,p=self.edge_drop_prob)
            h=self.gnns[l](h_list[l],edge_index,edge_attr)
            h=self.norms[l](h)
            #if not the last gnn layer, add dropout layer
            if l!=self.num_layer-1:
                h=self.dropout(h)

            h_list.append(h)


        #JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim = -1),kernel_size=self.num_layer+1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK=="attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list=torch.cat(h_list, dim = 0).transpose(0,1) # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score,_=self.attention_lstm(h_list) # N * num_layer * 2
            attention_score=torch.softmax(torch.sum(attention_score,dim=-1),dim=1).unsqueeze(-1) #N * num_layer  * 1
            node_representation=torch.sum(h_list*attention_score,dim=1)


        return node_representation


def make_gnn_layer(gnn_type,emb_dim,aggr="add",eps=0.,train_eps=False,head=None,negative_slope=0.2,num_edge_type=0):
    """function to construct gnn layer
    Args:
        gnn_type(str):
    """
    if gnn_type=="GCN":
        return GCNLayer(emb_dim,emb_dim,"add",num_edge_type)
    elif gnn_type=="GIN":
        return GINLayer(emb_dim,eps,train_eps,"add",num_edge_type)
    elif gnn_type=="GraphSAGE":
        return GraphSAGELayer(emb_dim,emb_dim,aggr,num_edge_type)
    elif gnn_type=="GAT":
        return GATLayer(emb_dim,emb_dim,head,negative_slope,"add",num_edge_type)
    else:
        raise ValueError("Not supported GNN type")

class NodeClassifier(nn.Module):
    """Node classifier
    Args:
        embedding_model(nn.Module):model used for learning node embedding
        emb_dim(int): the size of output feature in embedding model
        num_tasks(int): number of type in node classification

    """
    def __init__(self,embedding_model, emb_dim,num_tasks):
        super(NodeClassifier, self).__init__()
        self.embeding_model=embedding_model
        self.emb_dim=embedding_model.output_dim
        self.JK=self.embedding_model.JK
        self.num_gnn_layer=self.embedding_model.num_layer
        if self.JK=="concat":
            self.classifier=nn.Linear(emb_dim*(self.num_gnn_layer+1),num_tasks)
        else:
            self.classifier=nn.Linear(emb_dim,num_tasks)

    def forward(self,*args):
        x=self.embeding_model(args)
        return self.classifier(x)

class GraphClassifier(nn.Module):
    """Graph classifier
    Args:
        embedding_model(nn.Module):model used for learning node embedding
        emb_dim(int): the size of output feature in embedding model
        pooling_method(str): method of pooling layer
        num_tasks(int): number of type in graph classification

    """

    def __init__(self, embedding_model, emb_dim,pooling_method, num_tasks,sort_pooling_ratio=0.5):
        super(GraphClassifier, self).__init__()
        self.embedding_model = embedding_model
        self.emb_dim=embedding_model.output_dim
        self.JK=self.embedding_model.JK
        self.num_gnn_layer=self.embedding_model.num_layer
        self.num_tasks=num_tasks
        self.pooling_method=pooling_method
        self.sort_pooling_ratio=sort_pooling_ratio
        #Different kind of graph pooling
        if pooling_method == "sum":
            self.pool = global_add_pool
        elif pooling_method == "mean":
            self.pool = global_mean_pool
        elif pooling_method == "max":
            self.pool = global_max_pool
        elif pooling_method == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_gnn_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif pooling_method == "set2set":
            #TODO:actively compute iteration time
            set2set_iter = 4
            if self.JK == "concat":
                self.pool = Set2Set((self.num_gnn_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        if pooling_method == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        #classifier
        if self.JK == "concat":
            self.classifier= nn.Linear(self.mult * (self.num_gnn_layer + 1) * emb_dim, num_tasks)
        else:
            self.classifier = nn.Linear(self.mult * emb_dim, num_tasks)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        #node representation
        x=self.embedding_model(x,edge_index,edge_attr)
        pool_x=self.pool(x,batch)
        return self.classifier(pool_x)