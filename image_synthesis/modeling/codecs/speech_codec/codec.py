import torch
import torch.nn as nn
from image_synthesis.modeling.codecs.base_codec import BaseCodec

class Tokenize:
    def __init__(self, 
                pad_value: int = 2,
                seq_len: int = 750,
                tokens_per_frame: int = 8,
                mask_half: bool = False,
                 ):
        self.pad_value = pad_value
        self.mask_half = mask_half
        self.seq_len = seq_len
        self.tokens_per_frame = tokens_per_frame
 

    def get_tokens(self, tokens, **kwargs):
        mask = None
        if (tokens == self.pad_value).any():
            mask = (tokens == self.pad_value).byte()
        
        if self.mask_half:
            bs = len(tokens)
            assert tokens.numel() == bs * self.seq_len * self.tokens_per_frame
            half_mask = torch.ones((bs, self.seq_len, self.tokens_per_frame)).byte()
            half_mask[:, :, :self.tokens_per_frame//2] = 0
            half_mask = half_mask.reshape(bs, -1)
            if mask is not None:
                mask = mask & half_mask
            else:
                mask = half_mask
        
        return {
            'token': tokens,
            'mask': mask
        }





