import sys
sys.path.append("..")

from adapt_longformer import LongformerConfig

from transformers import (
    RobertaTokenizer,
    RobertaConfig
)
from models import *
from process_fn import triple_process_fn, triple2dual_process_fn

default_train_path = "/home/kwtang/msmarco/100k_sample_head.tsv"
default_process_fn = triple_process_fn

class MSMarcoConfig:
    def __init__(self, name, model, train_path=default_train_path, process_fn=default_process_fn, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig):
        self.name = name
        self.train_path = train_path
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class

configs = [
    MSMarcoConfig(name = "cosine_sim_mean_triple",
                model = RobertaMeanEmbedTripleLoss,
                use_mean = True,
                ),
    MSMarcoConfig(name = "mean_nce",
                model = RobertaMeanNCE,
                ),
    MSMarcoConfig(name = "mean_nll",
                model = RobertaNLL,
                ),
    MSMarcoConfig(name = "first_nll",
                model = RobertaNLL,
                use_mean = False,
                ),
    MSMarcoConfig(name = "mean_bce",
                model = RobertaVanillaBCE,
                ),
    MSMarcoConfig(name = "first_bce",
                model = RobertaVanillaBCE,
                use_mean = False,
                ),
    MSMarcoConfig(name = "rdot_nll_first",
                model = RobertaDot_NLL,
                process_fn=triple2dual_process_fn,
                use_mean = False,
                ),
    MSMarcoConfig(name = "rdot_nll_mean",
                model = RobertaDot_NLL,
                process_fn=triple2dual_process_fn,
                use_mean = True,
                ),
    MSMarcoConfig(name = "qk_nll_first",
                model = RobertaQK_NLL,
                process_fn=triple2dual_process_fn,
                use_mean = False,
                ),
    MSMarcoConfig(name = "rdot_bm25_nll_first",
                model = RobertaBM25NLL,
                use_mean = False,
                ),

    MSMarcoConfig(name = "bm25_nll_first_ln",
                model = RobertaDot_NLL_LN,
                use_mean = False,
                ),
    MSMarcoConfig(name = "nll_first_ln",
                model = RobertaDot_IBBM25_LN,
                use_mean = False,
                ),
    MSMarcoConfig(name = "ib_first_ln",
                model = RobertaDot_IB_LN,
                use_mean = False,
                ),
    MSMarcoConfig(name = "nce_first_ln",
                model = RobertaDot_NCE_LN,
                use_mean = False,
                ),
    MSMarcoConfig(name = "eval_clf_ln",
                model = RobertaDot_CLF_ANN,
                use_mean = False,
                process_fn=triple2dual_process_fn,
                ),
    # Longformer
    MSMarcoConfig(name = "lfm_dot_bm25_nll_first",
            model = Longformer_CLF_Dot_ANN,
            use_mean = False,
            config_class = LongformerConfig,
            ),
    MSMarcoConfig(name = "lfm_qk_bm25_nll_first",
            model = Longformer_CLF_QK_ANN,
            use_mean = False,
            config_class = LongformerConfig,
            ),

    MSMarcoConfig(name = "qk_bm25_nll_first",
                model = RobertaQK_BM25NLL,
                use_mean = False,
                ),
    MSMarcoConfig(name = "nll_first_200d",
                model = RobertaTriple_NLL_200d,
                use_mean = False,
                ),
    MSMarcoConfig(name = "cnn8x_bm25_nll_first",
                model = RobertaCNN8x_BM25NLL,
                use_mean = False,
                ),
    MSMarcoConfig(name = "cnn8x_qk_bm25_nll_first",
                model = RobertaCNN8x_QK_BM25NLL,
                use_mean = False,
                ),
    MSMarcoConfig(name = "cnn8x_qk_bce_first",
                model = RobertaCNN8x_QK_BM25BCE,
                process_fn=triple2dual_process_fn,
                use_mean = False,
                ),
    MSMarcoConfig(name = "cnn8x_bce_first",
                model = RobertaCNN8x_BM25BCE,
                process_fn=triple2dual_process_fn,
                use_mean = False,
                ),
]

MSMarcoConfigDict = {cfg.name:cfg for cfg in configs}