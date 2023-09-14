import torch
import torch.nn.functional as F
from torch.nn import Linear

import pandas as pd
import csv

import os
import openai
from dotenv import load_dotenv

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import GraphConv, to_hetero

def main():
    review_path = "./bgg-19m-reviews.csv"
    detailed_game_path = "./games_detailed_info.csv"

    def load_node_csv(path, index_col, encoders=None, **kwargs):
        df = pd.read_csv(path, index_col=index_col, low_memory=False, **kwargs)
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
    batch_size = 16
    data.batch_size = batch_size

    data['user'].num_nodes = torch.tensor(len(user_mapping)) 
    data['game'].num_nodes = torch.tensor(len(game_mapping))

    data.edge_types = [('user', 'rating', 'game')]
    data = T.ToUndirected()(data)

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
    
    model = Model(hidden_channels=32)
    model.load_state_dict(torch.load('./model_6nbrs,3deep,40epochs,32kbatch,32hidden.py'))
    model.eval()
    #print(model)

    embeds = data['game'].x

    names = pd.read_csv('./names.csv', encoding='utf8')
    names = names.drop(columns= names.columns[0])
    dict_names = dict(names['primary'])
    names_to_index = {name.lower() :index for (index, name) in dict_names.items()}

    #TODO: compile recommendations and save at the end
    load_dotenv()
    openai.api_key = os.getenv('API_KEY')

    print('Type \'Quit\' to exit and save your recommendations.')

    def get_response(game):
        return f'Write a short \'Board Game Geek\'-style description of a game with some thematic\
            and mechanical elements of the game {game}. Make sure the game has a title.'

    while True:
        try:
                
            game_chosen = input('What game would you like to see similar games to?\n')
            game_chosen_lower = game_chosen.lower()
            if game_chosen_lower== 'quit':
                break
            game_chosen_index = names_to_index[game_chosen_lower]

            dist = torch.norm(embeds - embeds[game_chosen_index], dim=1, p=None)
            knn = dist.topk(11, largest=False) 
            

            print('Some games similar to ', game_chosen, 'are:\n') 
            print([dict_names[nb.item()] for nb in knn[1][1:]])

            ai_games = input('Would you like to see a similar game that does not exit?\n')
            if ai_games.lower() == 'yes':
                
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": 'You are a creative, skilled board game designer.'},
                            {"role": "user", "content": get_response(game_chosen)}
                ])

                if response["choices"][0]["finish_reason"] == "stop":
                    print(response["choices"][0]["message"]["content"])

        except KeyError:
            print('That was not a game I know. Check spelling and punctuation or try a different game next time.\n')

#TODO: make it so the data is already premunged when I run inference
#TODO: put the model somewhere where i can just import the custom classes
#TODO: load sunmmary embeddings for nodes too?

if __name__ == "__main__":
    main()