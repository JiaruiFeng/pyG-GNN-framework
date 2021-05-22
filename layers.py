"""
Jiarui Feng
file contains the layers for building training model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as c
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.inits import glorot, zeros,kaiming_uniform


def clones( module, N):
    """Layer clone function, used for concise code writing
    Args:
        module: the layer want to clone
        N: the time of clone
    """
    return nn.ModuleList(c(module) for _ in range(N))


class GCNLayer(MessagePassing):
    """
    Graph convolution layer with edge attribute
    Args:
        input_dim(int): the size of input feature
        output_dim(int): the size of output feature
        aggr(str): aggregation function in message passing network
        num_edge_type(int): number of edge type, 0 indicate no edge attribute
    """
    def __init__(self,input_dim,output_dim,aggr="add",num_edge_type=0):
        super(GCNLayer, self).__init__()
        self.aggr=aggr
        self.proj=nn.Linear(input_dim,output_dim,bias=False)
        self.bias=nn.Parameter(torch.Tensor(output_dim))
        if num_edge_type>0:
            self.edge_embedding = torch.nn.Embedding(num_edge_type, output_dim)
            nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight.data)
        zeros(self.bias)

    def forward(self,x,edge_index,edge_attr=None):
        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        x = self.proj(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if edge_attr is not None:
            #add features corresponding to self-loop edges, set as zeros.
            self_loop_attr = torch.zeros(x.size(0),dtype=torch.long)
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.edge_embedding(edge_attr)

            return self.propagate(edge_index, x=x, norm=norm,edge_attr=edge_embeddings)
        else:
            return self.propagate(edge_index, x=x,norm=norm, edge_attr=None)

    def message(self, x_j,edge_attr,norm):
        if edge_attr is not None:
            return norm.view(-1,1)*(x_j+edge_attr)
        else:
            return norm.view(-1,1)*x_j

    def update(self,aggr_out):
        return F.relu(aggr_out)


# GAT torch_geometric implementation
#Adapted from https://github.com/snap-stanford/pretrain-gnns
class GATLayer(MessagePassing):
    """Graph attention layer with edge attribute
    Args:
        input_dim(int): the size of input feature
        output_dim(int): the size of output feature
        head(int): the number of head in multi-head attention
        negative_slope(float): the slope in leaky relu function
        aggr(str): aggregation function in message passing network
        num_edge_type(int): number of edge type, 0 indicate no edge attribute

    """
    def __init__(self, input_dim,output_dim,head, negative_slope=0.2, aggr = "add",num_edge_type=0):
        super(GATLayer, self).__init__(node_dim=0)
        assert output_dim%head==0
        self.k=output_dim//head
        self.aggr = aggr

        self.output_dim = output_dim
        self.head = head
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(input_dim, output_dim,bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(1, head, 2 * self.k))
        self.bias = torch.nn.Parameter(torch.Tensor(output_dim))

        if num_edge_type>0:
            self.edge_embedding = torch.nn.Embedding(num_edge_type, output_dim)
            nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_linear.weight.data)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index,edge_attr=None):

        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        x = self.weight_linear(x).view(-1, self.head, self.k) # N * head * k

        if edge_attr is not None:
            #add features corresponding to self-loop edges, set as zeros.
            self_loop_attr = torch.zeros(x.size(0),dtype=torch.long)
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.edge_embedding(edge_attr)
            return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        else:
            return self.propagate(edge_index, x=x, edge_attr=None)


    def message(self, edge_index, x_i, x_j, edge_attr):
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1, self.head, self.k)
            x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) # E * head
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        return x_j * alpha.view(-1, self.head, 1) #E * head * k

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1,self.output_dim)

        aggr_out = aggr_out + self.bias

        return F.relu(aggr_out)



#Adapted from https://github.com/snap-stanford/pretrain-gnns
class GINLayer(MessagePassing):
    """
    GIN layer to incorporate edge information.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        eps(float): initial epsilon.
        train_eps(bool): whether the epsilon is trainable
        aggr(str): aggregation function in message passing network
        num_edge_type(int): number of edge type, 0 indicate no edge attribute
    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, eps=0.,train_eps=False, aggr="add",num_edge_type=0):
        super(GINLayer, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        if num_edge_type > 0:
            self.edge_embedding = torch.nn.Embedding(num_edge_type, emb_dim)
            nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            zeros(m.bias.data)

    def reset_parameters(self):
        self.mlp.apply(self.weights_init)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr=None):
        # don't need to add self loop in GIN
        #edge_index,_ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_attr is not None:

            edge_embeddings = self.edge_embedding(edge_attr)
            x_n= self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        else:
            x_n=self.propagate(edge_index, x=x, edge_attr=None)

        return self.mlp((1+self.eps)*x+x_n)


    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return x_j + edge_attr
        else:
            return x_j

    def update(self, aggr_out):
        return aggr_out


class GraphSAGELayer(MessagePassing):
    """GraphSAGE layer with edge attributes
    Args:
        input_dim(int): the size of input feature
        output_dim(int): the size of output feature
        aggr(str): aggregation function in message passing network
        num_edge_type(int): number of edge type, 0 indicate no edge attribute

    """
    def __init__(self,input_dim,output_dim,aggr="mean",num_edge_type=0):
        super(GraphSAGELayer, self).__init__()
        self.aggr=aggr
        self.proj=nn.Linear(input_dim*2,output_dim,bias=False)
        self.bias=nn.Parameter(torch.Tensor(output_dim))
        if num_edge_type > 0:
            self.edge_embedding = torch.nn.Embedding(num_edge_type, input_dim)
            torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight.data)
        zeros(self.bias)

    def forward(self,x,edge_index,edge_attr=None):
        # don't need to add self loop in GraphSAGE
        #edge_index,_ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_attr is not None:

            edge_embeddings = self.edge_embedding(edge_attr)
            x_n= self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        else:
            x_n=self.propagate(edge_index, x=x, edge_attr=None)

        return F.normalize(F.relu(self.proj(torch.cat([x,x_n],dim=-1))+self.bias),p=2,dim=-1)


    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return x_j + edge_attr
        else:
            return x_j

    def update(self, aggr_out):
        return aggr_out


#GAT torch implementation
def masking_softmax(att,A):
    """masking softmax in GAT layer
    Args:
        att: the unmasked attention score matrix
        A: masking matrix, <=0 for masking position, >0 for not masking position
    """
    masking=A>0 #B * N * N
    masking=masking.int()
    masking=masking.unsqueeze(1) #B * 1 * N * N
    att=att.masked_fill_(masking==0,-1e30)
    return F.softmax(att,dim=-1) #B * h * N * N


class GATLayerTorch(nn.Module):
    """GAT layer
    Args:
        input_size:the size of input feature
        output_size:the size of output feature
        head: number of head in multi-head attention
    """
    def __init__(self,input_size,output_size,head):
        super(GATLayerTorch, self).__init__()
        self.k=output_size//head
        self.head=head
        self.proj=nn.Linear(input_size,output_size,bias=False)
        self.att_proj_list=clones(nn.Linear(2*self.k,1),self.head)

    def forward(self,x,A):
        B=x.size(0)
        x=self.proj(x) # B * N * H
        x=x.view(B,-1,self.head,self.k).transpose(1,2).contiguous() # B * h * N * k
        att_input=self.attention_input(x) #h * B * N * N * 2k
        att=torch.cat([F.leaky_relu(self.att_proj_list[i](att_input[i]),negative_slope=0.2)for i in range(att_input.size(0))],dim=-1) # B * N * N * h
        att=masking_softmax(att.permute(0,3,1,2),A) # B * h * N * N
        x=F.relu(torch.matmul(att,x)) # B * h * N * k
        x=x.transpose(1,2).contiguous().view(B,-1,self.k*self.head)
        return x # B * N * hk(H)

    def attention_input(self,x):
        B,h,N,k=x.size()
        Wi=x.repeat_interleave(N,dim=2) # B * h * (N*N) * k
        Wj=x.repeat(1,1,N,1) # B * h * (N*N) * k
        cat=torch.cat([Wi,Wj],dim=-1) #B * h * (N*N) * 2k
        return cat.view(B,h,N,N,2*k).transpose(0,1) # h * B * N * N * 2k

