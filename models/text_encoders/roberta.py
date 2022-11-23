import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

class RobertaEncoder(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        model_name = 'roberta-base'
        self.feature_size = feature_size
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, attn_mask):
        if self.training == True:
            self.model.train()
            outputs = self.model(x, attn_mask).last_hidden_state[:, 0]
        else:    
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x, attn_mask).last_hidden_state[:, 0]
        
        return outputs
    
    @classmethod
    def code(cls) -> str:
        return 'roberta'

class BertFc(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        hidden_size = 768
        self.model = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, feature_size),
        )

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    @classmethod
    def code(cls) -> str:
        return 'bertfc'