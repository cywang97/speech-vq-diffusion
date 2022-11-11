import torch
import torch.nn as nn
from .base_embedding import BaseEmbedding

class Embedding(BaseEmbedding):
    def __init__(self,
                num_embed, 
                embed_dim, 
                padding_idx, 
                pos_emb_type,
                spatial_size=[750, 8]
                ):
        super().__init__()

        if isinstance(spatial_size, int):
            spatial_size = [spatial_size]

        self.spatial_size = spatial_size
        self.num_embed = num_embeddings
        self.embed_dim = embed_dim
        self.pos_emb_type = pos_emb_type

        assert self.pos_emb_type in ['embedding', 'paramter']

        self.emb = nn.Embedding(self.num_embed, embed_dim)
        if self.pos_emb_type == 'embedding':
            self.time_emb = nn.Embedding(self.spatial_size[0], embed_dim)
            if len(self.spatial_size) == 2:
                self.channel_emb = nn.Embedding(self.spatial_size[1], embed_dim)
        else:
            self.time_emb = nn.Parameter(torch.zeros(1, self.spatial_size[0], embed_dim)) # height #32,1024
            if len(self.spatial_size) == 2:
                self.channel_emb = nn.Parameter(torch.zeros(1, self.spatial_size[1], embed_dim)) # width   #32,1024
        
    def forward(self, index, **kwargs):
        assert index.dim() == 2
        emb = self.emb(index)

        if emb.shape[1] > 0:
            if self.pos_emb_type == 'embedding':