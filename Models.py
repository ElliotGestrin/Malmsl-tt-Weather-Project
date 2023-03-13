import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm

class WeatherModel(nn.Module):

    class LinearModel(nn.Module):
        
        def __init__(self, in_dim: int, out_dim : int, depth: int, softmax: bool = False):
            super().__init__()
            self.lins = nn.ModuleList()
            sizes = [round(in_dim + (out_dim - in_dim) * i / depth) for i in range(depth)] + [out_dim]
            for i in range(len(sizes)-1):
                self.lins.append(nn.Linear(sizes[i], sizes[i+1]))
            if softmax:
                self.finish = torch.nn.Softmax(dim=-1)
            else:
                self.finish = torch.nn.Identity()

        def forward(self, X):
            for lin in self.lins[:-1]:
                X = nn.functional.relu(lin(X))
            X = self.lins[-1](X)
            return self.finish(X)

    def __init__(self, features: list[str], h_dims: list[int] = [], emb_dim = 3):
        super().__init__()
        self.features = features
        self.months = features.index("Month")
        self.days = features.index("Day")
        self.hours = features.index("Hour")
        self.categorical = ["Month","Day","Hour"]
        self.continuous = [i for i in range(len(features)) if features[i] not in self.categorical]

        self.emb_dim = emb_dim
        self.m_emb = nn.Embedding(13,emb_dim) # 0 is used for "unknown"
        self.h_emb = nn.Embedding(25,emb_dim) # 0 is used for "unknown"

        self.means = nn.Parameter(torch.zeros([len(self.continuous)]),requires_grad=False)
        self.stds = nn.Parameter(torch.ones([len(self.continuous)]),requires_grad=False)

        self.lins = nn.ModuleList()
        in_dim = len(self.continuous)*2 + 2*(emb_dim + 1) + 1 # Also include flag for if value is known or not
        sizes = [in_dim] + h_dims
        for i in range(len(sizes)-1):
            self.lins.append(nn.Linear(sizes[i], sizes[i+1]))

        self.cont_regressor = self.LinearModel(sizes[-1], len(self.continuous), 3)
        self.month_classifier = self.LinearModel(sizes[-1], 12, 1, True) #
        self.day_classifier = self.LinearModel(sizes[-1], 31, 1, True) #
        self.hour_classifier = self.LinearModel(sizes[-1], 24, 1, True) #

        self.double()

    def normalize(self, X: torch.Tensor):
        # Deal with non-batched
        if len(X.shape) == 1:
            X.unsqueeze_(0)
        X = X.float()

        # Extract the flags
        flags = X[:,len(self.continuous)+3:]
        cont_flags = flags[:, 3:].bool()

        # Deal with the continous
        cont = (X[:,self.continuous].float() - self.means)/self.stds
        cont[cont_flags] = 0 # Set to 0 for continous flags

        # Deal with the months
        black_month = torch.stack([(X[:,self.months] == 0)]*self.emb_dim,dim=-1) # If the month is unknown
        X[X[:,self.days] == 0,self.days] = 15 # Set to middle of month if unknown day
        close_month = torch.stack([(X[:,self.days] < 15)]*self.emb_dim,dim=-1) # Which month is the closest (last or next)
        last_emb = self.m_emb((X[:,self.months].long()-2)%12+1)
        next_emb = self.m_emb(X[:,self.months].long()%12+1)
        close_emb = torch.where(close_month,last_emb,next_emb)
        curr_emb = self.m_emb(X[:,self.months].long())
        c_weight = (15 - abs(X[:,self.days] - 15))/30 + 0.5 # Weight for current month. 1 at 15th, 0.5 at 1st or 30th
        c_weight.unsqueeze_(dim=-1)
        months = torch.where(black_month, curr_emb, curr_emb * c_weight + close_emb * (1-c_weight))
        
        # Deal with the hours
        hours = self.h_emb(X[:,self.hours].long()) # Already 0 if flagged

        return torch.concat([cont,months,hours,flags],dim=-1)

    def de_normalize(self, Y: torch.Tensor):
        return (Y*self.stds) + self.means
    
    def forward(self, X: torch.Tensor):
        X = self.normalize(X)
        for lin in self.lins[:]:
            X = nn.functional.relu(lin(X))
        cont = self.cont_regressor(X)
        cont = self.de_normalize(cont)
        month = self.month_classifier(X)
        day = self.day_classifier(X)
        hour = self.hour_classifier(X)
        return (cont,month,day,hour)
    
    def infer(self):
        num_feats = len(self.continuous) + 3
        num_flags = num_feats

        X = torch.zeros(num_feats)
        flags = torch.zeros(num_flags)
        for i,feat in enumerate(self.features):
            ans = input(f"{feat}? ")
            if "?" in ans:
                flags[i] = 1
            else:
                X[i] = float(ans)
        tot = torch.concat([X,flags],dim=-1)

        (p_c,p_m,p_d,p_h) = self(tot)

        print(f"{round(p_c[0,0].item(),1)}-{torch.argmax(p_m).item()+1}-{torch.argmax(p_d).item()+1} {torch.argmax(p_h).item()+1}:00")
        for i,cat in enumerate(self.continuous):
            if i == 0: continue
            print(f"   {self.features[cat]}: {p_c[0,i].item()}")

if __name__ == "__main__":
    train_data = pd.read_csv("Data/MalmslättFull_train.csv")
    cats = [c for c in train_data.columns]
    mod = WeatherModel(cats, [50, 40])
    mod.load_state_dict(torch.load("Models/Weather_c8_reg.pt"))
    print(mod)
    while True:
        mod.infer()