"""
    code by Brandon Theodorou
    Original GPT-2 Paper and repository here: https://github.com/openai/gpt-2
    Original GPT-2 Pytorch Model: https://github.com/huggingface/pytorch-pretrained-BERT
    GPT-2 Pytorch Model Derived From: https://github.com/graykode/gpt-2-Pytorch
"""
import polars as pl

class HALOConfig(object):
    def __init__(
        self,
        dv,
        # used for 128 swig
        #n_positions=100,
        #n_ctx=64,
        n_positions=48,
        n_ctx=48,
        #n_positions=100,
        #n_ctx=100,
        n_embd=384,
        n_layer=6,
        n_head=6,
        #n_embd=768,
        #n_layer=12,
        #n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        batch_size=128,
        sample_batch_size=512,
        epoch=100,
        pos_loss_weight=None,
        #lr=1e-5,
        lr=1e-4,
        num_workers=3,
        pin_memory=True,
    ):
        self.dv = dv
        # +3 for the start, end, and pad tokens
        self.total_vocab_size = dv["end_idx"].max() + 3 
        self.start_token_idx = dv["end_idx"].max()
        self.end_token_idx = self.start_token_idx + 1
        self.pad_token_idx = self.start_token_idx + 2
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.epoch = epoch
        self.pos_loss_weight = pos_loss_weight
        self.lr = lr
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def __str__(self):
        out = f'HALOConfig\n'
        for key,value in self.__dict__.items():
            out += f'\t{key}={value}\n'
        return out
