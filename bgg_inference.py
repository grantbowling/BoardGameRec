import torch
import torch.nn.functional as F
from torch.nn import Linear

import pandas as pd
import csv

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero

def main():
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
    data.edge_attr = torch.flatten(edge_attr)

    device = torch.device('mps')
    batch_size = 15000
    data.batch_size = batch_size

    #need to deal with this, use a small learnable embedding instead? id like to use mps with sparse matrices (not implemented)
    #as hack use ones vector? 
    #if use just eye its too big for gpu and cpu
    #why doesnt cpu and sparse work?? weird error
    #'NotImplementedError: Could not run 'aten::scatter_add.out' with arguments from the 'SparseCPU' backend.' but SparseCPU in list
    #data['user'].x = torch.eye(data['user'].num_nodes) 
    #data['user'].x = sparse_eye(data['user'].num_nodes) #try different sparse id matrix?
    #data['user'].x = torch.nn.Embedding(len(user_mapping), 32)
    
    
    data['user'].x = torch.unsqueeze(torch.ones(data['user'].num_nodes), dim=-1) #TODO: TRY APPLYING AFTER BATCHING

    data['user'].num_nodes = torch.tensor(len(user_mapping)) 
    data['game'].num_nodes = torch.tensor(len(game_mapping))

    #data['user'].x_dict = {'user': torch.nn.Embedding(data['user'].num_nodes, dim_embd)}
    #data.x_dict = torch.nn.ModuleDict({'user': torch.nn.Embedding(data['user'].num_nodes, dim_embd), 
                                        #'game': torch.nn.Embedding(data['game'].num_nodes, dim_embd)})


    data.edge_types = [('user', 'rating', 'game')]
    data = T.ToUndirected()(data)
    del data['game', 'rev_rating', 'user'].edge_label
    del data['game', 'rev_rating', 'user'].edge_label_index

    class GNNEncoder(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden_channels)
            self.conv2 = SAGEConv((-1, -1), out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x    

    class EdgeDecoder(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.lin1 = Linear(2 * hidden_channels, hidden_channels)
            self.lin2 = Linear(hidden_channels, 1)

        def forward(self, z_dict, edge_label_index):
            row, col = edge_label_index
            z = torch.cat([z_dict['user'][row], z_dict['game'][col]], dim=-1)

            z = self.lin1(z).relu()
            z = self.lin2(z)
            return z.view(-1)
    
    class Model(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.encoder = GNNEncoder(hidden_channels, hidden_channels)
            self.encoder = to_hetero(self.encoder, data.metadata(), aggr='max')
            self.decoder = EdgeDecoder(hidden_channels)

        def forward(self, x_dict, edge_index_dict, edge_label_index):
            z_dict = self.encoder(x_dict, edge_index_dict)
            return self.decoder(z_dict, edge_label_index)
    
    model = Model(16)
    model.load_state_dict(torch.load('./model_8nbrs,2deep,10epochs.py'))
    model.eval()
    print(model)

    embeds = model.encoder(data.x_dict, data.edge_index_dict)
    print(embeds)

    #TODO: build a dictionary of primary name to index in the csv file, use this to look up the games entered by users
    #then average the embeddings and do a nearest neighbor type search
    #f = open('./games_names.csv', "r")
    #print(f.readline())

    names = pd.read_csv('./names.csv', encoding='utf8')
    print(names)
    
    dist = torch.norm(embeds['game'] - embeds['game'][35], dim=1, p=None)
    knn = dist.topk(6, largest=False)
    print(knn)


if __name__ == "__main__":
    main()