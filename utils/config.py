import sys
sys.path += ['../']
from data.process_fn import triple_process_fn, triple2dual_process_fn
from model.models import *
from transformers import (
    RobertaTokenizer,
    RobertaConfig
)


default_process_fn = triple_process_fn


class MSMarcoConfig:
    def __init__(self, name, model, process_fn=default_process_fn, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class


configs = [
    MSMarcoConfig(name="rdot_nll",
                  model=RobertaDot_NLL_LN,
                  use_mean=False,
                  ),
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}
