from transformers.configuration_utils import PretrainedConfig
#from transformers.utils import logging
#logger = logging.get_logger(__name__)

import logging
logger = logging.getLogger(__name__)

# DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
#     "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/config.json",
#     "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/config.json",
#     "microsoft/deberta-v2-xlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/config.json",
#     "microsoft/deberta-v2-xxlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/config.json",
# }


class SEEDEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.DebertaV2Model`. It is used
    to instantiate a DeBERTa-v2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DeBERTa
    `microsoft/deberta-v2-xlarge <https://huggingface.co/microsoft/deberta-base>`__ architecture.
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.
    Arguments:
        vocab_size (:obj:`int`, `optional`, defaults to 128100):
            Vocabulary size of the DeBERTa-v2 model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.DebertaV2Model`.
        hidden_size (:obj:`int`, `optional`, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 24):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 6144):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"`, :obj:`"gelu"`, :obj:`"tanh"`, :obj:`"gelu_fast"`,
            :obj:`"mish"`, :obj:`"linear"`, :obj:`"sigmoid"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 0):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.DebertaModel` or
            :class:`~transformers.TFDebertaModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-7):
            The epsilon used by the layer normalization layers.
        relative_attention (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether use relative position encoding.
        max_relative_positions (:obj:`int`, `optional`, defaults to -1):
            The range of relative positions :obj:`[-max_position_embeddings, max_position_embeddings]`. Use the same
            value as :obj:`max_position_embeddings`.
        pad_token_id (:obj:`int`, `optional`, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (:obj:`List[str]`, `optional`):
            The type of relative position attention, it can be a combination of :obj:`["p2c", "c2p", "p2p"]`, e.g.
            :obj:`["p2c"]`, :obj:`["p2c", "c2p"]`, :obj:`["p2c", "c2p", 'p2p"]`.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """
    model_type = "seed_encoder"

    def __init__(
        self,
        pad_token_id=1,
        vocab_size=32769,
        encoder_layers=12,
        encoder_embed_dim=768,
        encoder_ffn_embed_dim=3072,
        encoder_attention_heads=12,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        encoder_layerdrop=0.0,
        max_positions=512,
        activation_fn='gelu',
        quant_noise_pq=0.0,
        quant_noise_pq_block_size=8,
        train_ratio='0.5:0.5',
        decoder_atten_window=2,
        pooler_activation_fn='tanh',
        pooler_dropout=0.0,
        encoder_layers_to_keep=None,
        decoder_layers=3,
        decoder_embed_path=None,
        decoder_embed_dim=768,
        decoder_ffn_embed_dim=3072,
        decoder_attention_heads=12,
        decoder_normalize_before=True,
        decoder_learned_pos=True,
        adaptive_softmax_cutoff=None,
        adaptive_softmax_dropout=0,
        share_decoder_input_output_embed=True,
        share_all_embeddings=True,
        no_token_positional_embeddings=False,
        adaptive_input=False,
        no_cross_attention=False,
        cross_self_attention=False,
        no_scale_embedding=True,
        layernorm_embedding=True,
        tie_adaptive_weights=True,
        decoder_layers_to_keep=None,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.pad_token_id=pad_token_id
        self.vocab_size=vocab_size
        self.encoder_layers=encoder_layers
        self.encoder_embed_dim=encoder_embed_dim
        self.encoder_ffn_embed_dim=encoder_ffn_embed_dim
        self.encoder_attention_heads=encoder_attention_heads

        self.dropout=dropout
        self.attention_dropout=attention_dropout
        self.activation_dropout=activation_dropout
        self.encoder_layerdrop=encoder_layerdrop
        self.max_positions=max_positions
        self.activation_fn=activation_fn
        self.quant_noise_pq=quant_noise_pq
        self.quant_noise_pq_block_size=quant_noise_pq_block_size


        self.train_ratio=train_ratio
        self.decoder_atten_window=decoder_atten_window
        self.pooler_activation_fn=pooler_activation_fn
        self.pooler_dropout=pooler_dropout


        self.encoder_layers_to_keep=encoder_layers_to_keep
        self.decoder_layers=decoder_layers
        self.decoder_embed_path=decoder_embed_path
        self.decoder_embed_dim=decoder_embed_dim
        self.decoder_ffn_embed_dim=decoder_ffn_embed_dim
        self.decoder_attention_heads=decoder_attention_heads
        self.decoder_normalize_before=decoder_normalize_before
        self.decoder_learned_pos=decoder_learned_pos
        self.adaptive_softmax_cutoff=adaptive_softmax_cutoff
        self.adaptive_softmax_dropout=adaptive_softmax_dropout
        self.share_decoder_input_output_embed=share_decoder_input_output_embed
        self.share_all_embeddings=share_all_embeddings
        self.no_token_positional_embeddings=no_token_positional_embeddings

        self.adaptive_input=adaptive_input
        self.no_cross_attention=no_cross_attention
        self.cross_self_attention=cross_self_attention

        self.decoder_output_dim=decoder_embed_dim
        self.decoder_input_dim=decoder_embed_dim

        self.no_scale_embedding=no_scale_embedding
        self.layernorm_embedding=layernorm_embedding
        self.tie_adaptive_weights=tie_adaptive_weights
        self.decoder_layers_to_keep=decoder_layers_to_keep


        self.decoder_layerdrop=0

        self.max_source_positions=max_positions
        self.max_target_positions=max_positions

        self.initializer_range = initializer_range






        

















