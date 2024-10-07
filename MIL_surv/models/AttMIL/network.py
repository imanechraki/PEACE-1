import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"




class FC(nn.Module):
    "Fully-connected model to generate final output."
    def __init__(self, in_features, out_features,dropout=0.25,batchnorm=False):
        super(FC, self).__init__()
        layer = nn.ModuleList()
        layer.append(nn.Dropout(dropout))
        layer.append(nn.Linear(in_features, out_features))
        layer.append(nn.ReLU())
        
        if batchnorm: ## No need bor batches of 1
            layer.append(nn.BatchNorm1d(out_features))
        self.fc = nn.Sequential(*layer)

    def forward(self, x):
        return self.fc(x)

class ClinicalNet(nn.Module):
    """Clinical data extractor.

    Handle categorical feature embeddings.
    """
    def __init__(self, output_vector_size = 512, dropout=0.25, embedding_dims=[
        (6, 3), (3, 1),(2, 1),(3, 1),(4,2),(5,2)]): 
        super(ClinicalNet, self).__init__()
        # Embedding layer
        self.embedding_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in embedding_dims])
        n_embeddings = sum([y for x, y in embedding_dims])
        # Linear Layers
        self.linear = nn.Linear(n_embeddings, 256)
        # Embedding dropout Layer
        self.embedding_dropout = nn.Dropout(dropout)
        # Output Layer
        self.output_layer = FC(256, output_vector_size, dropout=0.25)

    def forward(self, x):
   

        x = [emb_layer(x[:, i]) for i, emb_layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, 1)
        x = self.embedding_dropout(x)
        x = self.linear(x)
        out = self.output_layer(x) # (B,512)
        return out
    

class Fusion(nn.Module):
    "Multimodal data aggregator."
    def __init__(self, method ):
        super(Fusion, self).__init__()
        self.method = method

    def forward(self, x):
        # if self.method == 'attention':
        #     out = self.attention(x)
        if self.method == 'cat':
            out = torch.cat([m for m in x], dim=1)
        if self.method == 'max':
            out = x.max(dim=0)[0]
        if self.method == 'sum':
            out = x.sum(dim=0)
        if self.method == 'prod':
            out = x.prod(dim=0)
   

        return out
    
    
    
class DAttention(nn.Module):
    def __init__(self, dropout, act='relu', n_features=1024, output_vector_size=512):
        super(DAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1    #ATTENTION_BRANCHES       
        self.feature = [nn.Linear(n_features, 512)]

        if act.lower() == "gelu":
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K))
        self.fc = FC(self.L * self.K, output_vector_size )
        
   
    def forward(self, x):
        x = x.squeeze(0)
        feature = self.feature(x)
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        logits = self.fc(M)
        return logits
    


class MultiSurv(torch.nn.Module):
    """Deep Learning model for MULTImodal  SURVival prediction."""
    def __init__(self, dropout, act, n_features=1024,data_modalities=['clinical', 'wsi'], fusion_method='max',
                 n_output_intervals=4):
        super(MultiSurv, self).__init__()
        self.mfs = modality_feature_size = 512
        self.data_modalities = data_modalities
        # Numerical embedding of treatement class value 
        self.embed_RX_f = nn.Embedding(2,1)
        self.embed_RX_cf = nn.Embedding(2,1)

        if fusion_method == 'cat':
            self.num_features = 0
        else:
            self.num_features = self.mfs

        self.submodels = {}

        # Clinical -----------------------------------------------------------#
        if 'clinical' in self.data_modalities:
            self.clinical_submodel = ClinicalNet(
                output_vector_size=self.mfs)
            self.submodels['clinical'] = self.clinical_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # WSI patches --------------------------------------------------------#
        if 'wsi' in self.data_modalities:
            self.wsi_submodel = DAttention(dropout, act, n_features,self.mfs)
            self.submodels['wsi'] = self.wsi_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # Instantiate multimodal aggregator ----------------------------------#
        if len(data_modalities) > 1:
            self.aggregator = Fusion(fusion_method)
  
        # Fully-connected and risk layers ------------------------------------#
        # n_fc_layers = 1

        self.fc_block = FC(
            in_features=1+self.num_features, out_features=512)

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,
                            out_features=n_output_intervals),
            torch.nn.Sigmoid()
        )

    def forward(self, x):

        # Factual Treatement variable embedding
        rx_f = x['RX_f']
        delta_sign = 1 if rx_f.item() == 1 else -1 
        rx_f = self.embed_RX_f(rx_f)
        # Counter Factual Treatement variable embedding
        rx_cf = x['RX_cf']
        rx_cf = self.embed_RX_cf(rx_cf)

        
        ## Multimodal head
        multimodal_features = tuple()
        # Run data through modality sub-models (generate feature vectors) ----#
        for modality in self.data_modalities:
            multimodal_features += (self.submodels[modality](x[modality]),)
        
        

        # Feature fusion/aggregation -----------------------------------------#
        if len(multimodal_features) > 1:
            x_factual = self.aggregator(torch.stack(multimodal_features))
            feature_repr = {'modalities': multimodal_features, 'fused': x_factual}
        else:  # skip if running unimodal data
            x_factual = multimodal_features[0]
            feature_repr = {'modalities': multimodal_features[0]}

        x_counterfactual = x_factual.clone()
        
        ## Factual branch
        x_factual = torch.cat([rx_f,x_factual], dim = 1)

        # print( x_factual.shape, x_counter_factual.shape)

        x_factual = self.fc_block(x_factual)
        hazards_factual = self.risk_layer(x_factual)
        S_factual = torch.cumprod(1 - hazards_factual, dim = 1)
        risk_factual = 1 - torch.sum(S_factual, dim=1)


   
        ## Counter factual branch 
        x_counterfactual = torch.cat([rx_cf, x_counterfactual], dim = 1)
        x_counterfactual = self.fc_block(x_counterfactual)
        hazards_counterfactual = self.risk_layer(x_counterfactual)
        S_counterfactual = torch.cumprod(1 - hazards_counterfactual, dim = 1)
        risk_counterfactual = 1 - torch.sum(S_counterfactual, dim = 1)

        delta_pred = delta_sign * (risk_counterfactual - risk_factual)
    
        return hazards_factual, S_factual,delta_pred, (risk_factual,risk_counterfactual)
        # return (hazards_factual,hazards_counterfactual), (S_factual,S_counterfactual), (risk_factual,risk_counterfactual) #feature_repr
    
  



if __name__ == '__main__':
    model = MultiSurv(dropout=0.25, act='relu', n_features=25,data_modalities=['clinical', 'wsi'])

    data = {'wsi':torch.rand(2, 25),'clinical':torch.tensor([[0,1,1,0]]), 'RX_f': torch.tensor([1]), 'RX_cf': torch.tensor([0])}
            # prediction
    hazards, S, risks = model(data)
    print(hazards, S, risks)
