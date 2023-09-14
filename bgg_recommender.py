import os
from random import shuffle
from turtle import forward

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import GraphConv, to_hetero, GAE, HeteroConv
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import train_test_split_edges

from torch_sparse import SparseTensor

from tqdm import tqdm

def main():
    game_path = "./2022-01-08.csv"
    review_path = "./bgg-19m-reviews.csv"
    detailed_game_path = "./games_detailed_info.csv"

    def load_node_csv(path, index_col, encoders=None, **kwargs):
        df = pd.read_csv(path, index_col=index_col, **kwargs)
        mapping = {index: i for i, index in enumerate(df.index.unique())}

        x = None
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)
        else:
            return df, mapping

        return x, mapping

    class NormEncoder(object):
        def __init__(self):
            self.sep = None
        def __call__(self, df):
            res = []
            for val in df.values:
                res.append([float(val)])
            tens = torch.tensor(res)
            return (tens - torch.mean(tens)) / torch.std(tens)
    
    class NoEncoder(object):
        def __init__(self):
            self.sep = None
        def __call__(self, df):
            res = []
            for val in df.values:
                res.append([float(val)])
            tens = torch.tensor(res)
            return tens

    class YearEncoder(object):
        def __init__(self):
            self.sep = None
        def __call__(self, df):
            res = []
            for val in df.values:
                if val != 0:
                    res.append([float(val)])
                else:
                    res.append([2000])
            tens = torch.tensor(res)
            return (tens - torch.mean(tens)) / torch.std(tens)

    class GameEncoder(object):
        def __init__(self):
            self.sep = None
        def __call__(self, df):
            categories = set()
            cat_list = df.tolist()
            big_list = []
            for str_cat in cat_list:
                if type(str_cat) == str:
                    str_cat = str_cat[2:len(str_cat)-1]
                    str_cat = str_cat.replace(' \'', "")
                    str_cat = str_cat.replace('\'', "")
                    str_cat = str_cat.replace('\"', "")
                    str_list = str_cat.split(',')
                    big_list.append(str_list)
                    for cat in str_list:
                        categories.add(cat)
                else:
                    big_list.append([])
            mapping = {category: i for i, category in enumerate(categories)}

            x = torch.zeros(len(df), len(mapping))
            for i, col in enumerate(big_list):
                for category in col:
                    x[i, mapping[category]] = 1
            return x
        
    game_x, game_mapping = load_node_csv(
        detailed_game_path, index_col= 'primary', encoders={
            #'yearpublished': YearEncoder(),
            #'minplayers': NormEncoder(),
            #'maxplayers': NormEncoder(),
            #'playingtime': NormEncoder(),
            'averageweight': NormEncoder(),
            'boardgamemechanic': GameEncoder(),
            'boardgamecategory': GameEncoder()  
        }) 


    user_x, user_mapping = load_node_csv(
        review_path, index_col='user'
    )

    data = HeteroData()
    data['user'].num_nodes = len(user_mapping)
    data['game'].num_nodes = len(game_mapping)
    data['game'].x = game_x

    def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None, **kwargs):
        df = pd.read_csv(path, **kwargs)
        df = df[df[dst_index_col].isin(dst_mapping)]
        df = df[df[src_index_col].isin(src_mapping)]
        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])

        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)

        return edge_index, edge_attr.type(torch.LongTensor)
    
    def sparse_eye(size):
        """
        Returns the identity matrix as a sparse matrix
        """
        indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
        values = torch.tensor(1.0).expand(size)
        cls = getattr(torch.sparse, values.type().split(".")[-1])
        return cls(indices, values, torch.Size([size, size])) 

    edge_index, edge_attr = load_edge_csv(review_path, src_index_col='user', src_mapping=user_mapping, dst_index_col='name', dst_mapping=game_mapping, encoders = {"rating": NormEncoder()})
    
    data['user', 'rating', 'game'].edge_index = edge_index
    data['user', 'rating', 'game'].edge_attr = torch.flatten(edge_attr)
    data.edge_index = edge_index
    data.edge_weight = torch.flatten(edge_attr)
    data.edge_weight_dict = {('user', 'rating', 'game'): torch.flatten(edge_attr)}

    device = torch.device('mps')
    batch_size = 32000
    data.batch_size = batch_size

    data['user'].num_nodes = torch.tensor(len(user_mapping)) 
    data['game'].num_nodes = torch.tensor(len(game_mapping))

    data['user'].node_id = torch.arange(len(user_mapping))
    data['game'].node_id = torch.arange(len(game_mapping))


    data.edge_types = [('user', 'rating', 'game')]
    data = T.ToUndirected()(data)

    torch.manual_seed(79)
 
    transform = T.RandomLinkSplit(is_undirected=True, edge_types=[('user', 'rating', 'game')], rev_edge_types=[('game', 'rev_rating', 'user')])
    train_data, val_data, test_data = transform(data)
    
    edge_label_index = train_data['user', 'rating', 'game'].edge_index
    edge_label = train_data['user', 'rating', 'game'].edge_label
    train_loader = LinkNeighborLoader(train_data, num_neighbors=[6] * 3,  batch_size = batch_size, edge_label_index=(('user', 'rating', 'game'), edge_label_index), edge_label=edge_label)
    val_loader = LinkNeighborLoader(val_data, num_neighbors=[6] * 3,  batch_size = batch_size, edge_label_index=(('user', 'rating', 'game'), edge_label_index), edge_label=edge_label)

    class GNN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()

            self.convs = torch.nn.ModuleList()

            self.num_layers = 3

            self.convs.append(GraphConv((-1, -1), hidden_channels, aggr='mean'))
            self.convs.append(GraphConv((-1, -1), hidden_channels, aggr='mean'))
            self.convs.append(GraphConv((-1, -1), hidden_channels, aggr='mean'))


    #TODO: make this part use the sparse tensors

        def forward(self, x_dict, edge_index_dict, edge_weight_dict):
            x = self.convs[0](x_dict, edge_index_dict, edge_weight_dict) 
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.convs[1](x_dict, edge_index_dict, edge_weight_dict) 
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.convs[2](x_dict, edge_index_dict, edge_weight_dict) 
            return x

    class Classifier(torch.nn.Module):
        def forward(self, user_x, game_x, edge_label_index):
            edge_feat_user = user_x[edge_label_index[0]]
            edge_feat_game = game_x[edge_label_index[1]]    
            return (edge_feat_user * edge_feat_game).sum(dim = -1)

    class Model(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.user_emb = torch.nn.Embedding(data['user'].num_nodes, hidden_channels)
            self.game_emb = torch.nn.Embedding(data['game'].num_nodes, hidden_channels)  

            self.gnn = GNN(hidden_channels)
            self.gnn = to_hetero(self.gnn, metadata=data.metadata())
            self.classifier = Classifier()
        
        def forward(self, data):
            x_dict = {'user': self.user_emb(data['user'].node_id), 'game': self.game_emb(data['game'].node_id)}
            x_dict = self.gnn(x_dict, data.edge_index_dict, {('user', 'rating', 'game'): data['user', 'rating', 'game'].edge_attr})
            pred = self.classifier(x_dict['user'], x_dict['game'], data['user', 'rating', 'game'].edge_label_index)
            return pred

    hidden_channels = 32
    model = Model(hidden_channels)
    print(model)
    #model.to(device)      

    optimizer = torch.optim.Adam(model.parameters(), lr = .005)

    #TODO: lazy initialization by passing in one batch, look up lazy initialization
    #TODO: can we do predictions on the edge weights themselves somehow, key= edge_attr at splitting?, have to get size right

    def train():
        model.train()
        total_loss = total_examples = 0
        pbar = tqdm(train_loader)
        for batch in pbar:
            optimizer.zero_grad()
            #batch = batch.to(device)
            #print(batch, ' is in pbar')
            pred = model(batch)
            ground_truth = batch['user', 'rating', 'game'].edge_label
            #print(pred.shape)
            ground_truth = ground_truth.float()
            #print(ground_truth.shape)
            #TODO: fix up this loss function? can use the index tensors to find the attr of the sampled edges, then pass in
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_examples += pred.numel()
            total_loss += float(loss) * pred.numel()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / total_examples

    #TODO: The losses returned by train and test are scaled differently?

    def test(loader):
        model.eval()
        scores = []
        with torch.no_grad():
            with tqdm(loader) as tq:
                for batch in tq:
                    #batch.to(device)
                    pred = model(batch)
                    ground_truth = batch['user', 'rating', 'game'].edge_label
                    rmse = F.binary_cross_entropy_with_logits(pred, ground_truth)
                    scores.append(rmse)
                    tq.set_postfix({'loss': rmse.item()})
        return np.average(scores)

    for epoch in range(1, 40):
        print('Training...')
        loss = train()
        #train_rmse = test(train_loader)
        print('Validating...')
        val_rmse = test(val_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}, 'f'Val: {val_rmse:.6f}')

    torch.save(model.state_dict(), './model_6nbrs,3deep,40epochs,32kbatch,32hidden.py')

    #TODO: model.compile()

    #TODO: MAKE the edge decoder just a DOT PRODUCT or something that keeps LOCATIONS together!

if __name__ == "__main__":
    main()