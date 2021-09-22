#from transformers.utils import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



from .modules import (
    LayerNorm,
    get_activation_fn,
    MultiheadAttention,

)
from .modules import quant_noise as apply_quant_noise_


from .transformer_sentence_encoder import TransformerSentenceEncoder,TransformerDecoder,EncoderOut


import os
from transformers.modeling_utils import PreTrainedModel


#logger = logging.get_logger(__name__)

import logging
logger = logging.getLogger(__name__)


from model.SEED_Encoder import SEEDEncoderConfig





class SEEDEncoderPretrainedModel(PreTrainedModel):

    config_class = SEEDEncoderConfig
    base_model_prefix = "seed_encoder"  

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()   
        elif isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.k_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.v_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)




class RobertaEncoder(nn.Module):
    """RoBERTa encoder."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=args.pad_token_id,
            vocab_size=args.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
        )
        #args.untie_weights_roberta = getattr(args, 'untie_weights_roberta', False)

        

    def forward(self, src_tokens, return_all_hiddens=False,  **unused):
        
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
        )
        x = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C

        # x_origin=x
        # if not features_only:    
        #     x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, {'inner_states': inner_states if return_all_hiddens else None}




class SEEDEncoderModel(SEEDEncoderPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.encoder=RobertaEncoder(config)
        self.init_weights()


    def forward(self, src_tokens, prev_tokens, return_all_hiddens=False, **kwargs):
    
        x_encoder, extra = self.encoder(src_tokens, return_all_hiddens, **kwargs)

        return x_encoder ,extra

    def get_input_embeddings(self):
        
        return self.encoder.sentence_encoder.embed_tokens

    def set_input_embeddings(self, value):
        
        self.encoder.sentence_encoder.embed_tokens = value


class SEEDEncoderForMaskedLM(SEEDEncoderPretrainedModel):
    """docstring for ClassName"""
    # _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    # _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]


    def __init__(self, config):
        super().__init__(config)
        self.seed_encoder = SEEDEncoderModel(config)
        self.decoder=TransformerDecoder(config,self.encoder.sentence_encoder.embed_tokens,no_encoder_attn=config.no_cross_attention)
        self.lm_head = RobertaLMHead(
            embed_dim=config.encoder_embed_dim,
            output_dim=config.vocab_size,
            activation_fn=config.activation_fn,
            weight=self.seed_encoder.encoder.sentence_encoder.embed_tokens.weight )
        self.train_ratio=config.train_ratio
        self.decoder_atten_window=config.decoder_atten_window 

        self.init_weights()


    def forward( src_tokens,prev_tokens, masked_tokens=None,**kwargs):
        x_encoder,_=self.seed_encoder(src_tokens)

        h=x_encoder[:,0:1,:]
        h=h.transpose(0,1)
        h=EncoderOut(
            encoder_out=h,  # T x B x C
            encoder_padding_mask=None,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )
        decoder_output=self.decoder(prev_tokens, encoder_out=h,local_attn_mask=self.decoder_atten_window)[0]


        features=self.lm_head(x_encoder, masked_tokens)
        return features, decoder_output

    def get_output_embeddings(self):
        return self.lm_head.weight

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.weight = new_embeddings



class SEEDEncoderForSequenceClassification(SEEDEncoderPretrainedModel):
    """docstring for ClassName"""
    def __init__(self, config):
        super().__init__(config)
        self.seed_encoder = SEEDEncoderModel(config)
        self.classification_heads=RobertaClassificationHead(
            config.encoder_embed_dim,
            config.encoder_embed_dim,
            config.num_labels,
            config.pooler_activation_fn,
            config.pooler_dropout,
            config.quant_noise_pq,
            config.quant_noise_pq_block_size,)

        self.init_weights()

    def forward(src_tokens,return_all_hiddens=False,**kwargs):

        x_encoder, extra = self.seed_encoder.encoder(src_tokens, return_all_hiddens, **kwargs)
        x = self.classification_heads(x_encoder,**kwargs)

        return x






class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout, q_noise=0, qn_block_size=8):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x




