import torch
from torch import nn
from transformers import (
    BertPreTrainedModel, 
    RobertaModel,
    RobertaForSequenceClassification
)
import torch.nn.functional as F
from adapt_longformer import Longformer, LongformerConfig

# from longformer.sliding_chunks import pad_to_window_size # TODO: double check if special padding is needed when running on longer input

class EmbeddingMixin(RobertaForSequenceClassification):
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained - model_argobj reads model params from config.py, then pass them into the model through from_pretrained
    """
    def __init__(self, config, model_argobj):
        super().__init__(config)
        self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)
        self.bce_logit_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t*mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s/d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:,0]


    def query_emb(self, input_ids, attention_mask, token_type_ids=None):
        # subclass should override/add normalization if needed
        outputs1 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
        # apparently this works better than outputs1[1]
        return self.masked_mean_or_first(outputs1, attention_mask)

    def body_emb(self, input_ids, attention_mask, token_type_ids=None):
        return self.query_emb(input_ids, attention_mask)

    def gen_embed_labels(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        # take q-pos-neg and convert to q-psg-label format for bce/nce/nll
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        labels = torch.tensor([1]*a_embs.shape[0]+[0]*b_embs.shape[0]).to(q_embs.device)
        q_embs = torch.cat([q_embs, q_embs], dim=0)
        a_embs = torch.cat([a_embs, b_embs], dim=0)
        return q_embs, a_embs, labels
    
    def triplet_loss(self, q_embs, a_embs, b_embs):
        # encoder should have normalized embs before calling this
        distance_pos = 1 - F.cosine_similarity(q_embs, a_embs)
        distance_neg = 1 - F.cosine_similarity(q_embs, b_embs)
        loss = F.relu(distance_pos - distance_neg + self.triplet_margin)
        assert len(loss.shape)==1
        return loss

    def nll_loss(self, q_embs, a_embs, labels=None):
        if labels is None:
            labels = torch.ones(a_embs.shape[0]).float().to(q_embs.device)
        logit_matrix = torch.matmul(q_embs, a_embs.t()) #[Q, Q]
        pos_indices = torch.where(labels==1.0)[0]
        pos_logit_matrix = logit_matrix[pos_indices]
        lsm = F.log_softmax(pos_logit_matrix, dim=1)
        loss = -1.0*lsm.gather(1, pos_indices.view(-1,1)).squeeze()
        return loss 

    def bce_losses(self, q_embs, a_embs, labels=None):
        assert q_embs.shape == a_embs.shape

        if labels is None:
            labels = torch.ones(a_embs.shape[0]).float().to(q_embs.device)

        labels = labels.float()

        # bce loss
        eval_logits = (q_embs*a_embs).sum(-1)
        eval_loss = self.bce_logit_loss(eval_logits, labels)

        # nce loss from L1
        logit_matrix = torch.matmul(q_embs, a_embs.t())
        logit_masked = logit_matrix + -1e12*torch.eye(logit_matrix.shape[0]).to(logit_matrix.device)
        m_sample = 1
        nce_topk_idx = torch.topk(logit_masked, logit_masked.shape[-1])[1].detach()[:, :m_sample]
        neg_embs = a_embs[nce_topk_idx].reshape(-1, a_embs.shape[-1])
        repeated_q_embs = torch.repeat_interleave(q_embs, m_sample, dim=0)
        nce_logits = (repeated_q_embs*neg_embs).sum(-1)
        nce_labels = torch.zeros(labels.shape[0]*m_sample).to(nce_logits.device).float()
        assert len(nce_logits.shape)==1
        nce_loss = self.bce_logit_loss(nce_logits, nce_labels)
        assert len(eval_loss.shape)==1

        return eval_loss, nce_loss

class RobertaMeanEmbedTripleLoss(EmbeddingMixin):
    base_model_prefix = "roberta"
    def __init__(self, config, model_args):
        super().__init__(config, model_args)
        self.use_mean = True
        self.triplet_margin = 1e-5
    
    def query_emb(self, input_ids, attention_mask):
        q_embs = super().query_emb(input_ids, attention_mask)
        return F.normalize(q_embs, p=2, dim=1)

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        trip_loss = self.triplet_loss(q_embs, a_embs, b_embs)
        return (trip_loss.mean(),)

class RobertaVanillaBCE(EmbeddingMixin):
    """
    This model takes a q-pos-neg input, converts it into q-psg-label, and computes BCE loss.
    This guarantees that the negative sample appears in the same batch as the positive.
    """
    base_model_prefix = "roberta"
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs, a_embs, labels = self.gen_embed_labels(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        eval_loss, nce_loss = self.bce_losses(q_embs, a_embs, labels)
        return (eval_loss.mean(),)

class RobertaMeanNCE(RobertaVanillaBCE):
    """
    After converting, also generates additional negative samples - for each query, pick the passage with highest dot product that is not original passage, then adding BCE loss from these samples. Adopted from L1 model.
    """
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)
    
    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs, a_embs, labels = self.gen_embed_labels(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        eval_loss, nce_loss = self.bce_losses(q_embs, a_embs, labels)
        loss = eval_loss.mean() + nce_loss.mean()
        return (loss,)

class RobertaNLL(RobertaVanillaBCE):
    """
    For each positive label, mark everything else in the batch as negative, then calculate negative log-likelihood loss as in FB paper.
    """
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs, a_embs, labels = self.gen_embed_labels(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        nll_loss = self.nll_loss(q_embs, a_embs, labels)
        return (nll_loss.mean(),)

class RobertaDot_NLL(EmbeddingMixin):
    """
    Compress embedding to 200d, then computes NLL loss. 
    """
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)
        self.embeddingHead = nn.Linear(config.hidden_size, 200)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.embeddingHead(full_emb)
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, labels):
        q_embs = self.query_emb(query_ids, attention_mask_q) # output should be [batch, emb/3]
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        nll_loss = self.nll_loss(q_embs, a_embs, labels)
        return (nll_loss.mean(),)

class RobertaQK_NLL(RobertaDot_NLL):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)
        self.downsample_q = nn.Linear(config.hidden_size, 200)
        self.downsample_k = nn.Linear(config.hidden_size, 200)
        self.apply(self._init_weights)

    def get_qk(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)
        tu = (outputs1[0], outputs1[0][:,0])
        full_emb = self.masked_mean_or_first(tu, attention_mask)
        # try separate embeddings
        q = self.downsample_q(full_emb)
        k = self.downsample_k(full_emb)
        return q, k

    def query_emb(self, input_ids, attention_mask):
        q, k = self.get_qk(input_ids, attention_mask)
        return q

    def body_emb(self, input_ids, attention_mask):
        q, k = self.get_qk(input_ids, attention_mask)
        return k

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, labels):
        q_embs = self.query_emb(query_ids, attention_mask_q) # output should be [batch, emb/3]
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        nll_loss = self.nll_loss(q_embs, a_embs, labels)
        return (nll_loss.mean(),)

class RobertaTriple_NLL_200d(RobertaDot_NLL):
    """200d compression, then identical to RobertaMeanNLL"""
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs, a_embs, labels = self.gen_embed_labels(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        nll_loss = self.nll_loss(q_embs, a_embs, labels)
        return (nll_loss.mean(),)

class RobertaBM25NLL(RobertaDot_NLL):
    """Only uses triplet to compute nll (CE)."""
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0]
        return (loss.mean(),)

class RobertaQK_BM25NLL(RobertaQK_NLL):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

class RobertaDot_NLL_LN(EmbeddingMixin):
    """
    Compress embedding to 200d, then computes NLL loss. 
    """
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0]
        return (loss.mean(),)

class RobertaDot_IBBM25_LN(RobertaDot_NLL_LN):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs, a_embs, labels = self.gen_embed_labels(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        nll_loss = self.nll_loss(q_embs, a_embs, labels)
        return (nll_loss.mean(),)

class RobertaDot_IB_LN(RobertaDot_NLL_LN):
    """
    For each positive, use all other positives as negative samples, then NLL - no BM25
    """
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        nll_loss = self.nll_loss(q_embs, a_embs)
        return (nll_loss.mean(),)

class RobertaDot_NCE_LN(RobertaDot_NLL_LN):
    """
    For each positive, only sample one other positive with highest dot product, then NLL - no BM25
    """
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        logit_matrix = torch.matmul(q_embs, a_embs.t())
        logit_masked = logit_matrix + -1e12*torch.eye(logit_matrix.shape[0]).to(logit_matrix.device)
        m_sample = 1
        nce_topk_idx = torch.topk(logit_masked, logit_masked.shape[-1])[1].detach()[:, :m_sample]
        b_embs = a_embs[nce_topk_idx].reshape(-1, a_embs.shape[-1])

        logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0]
        return (loss.mean(),)
    
class RobertaQK_BM25NLL(RobertaQK_NLL):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0]
        return (loss.mean(),)

# ============================================
# Longformer Support
class Longformer_CLF_Dot_ANN(BertPreTrainedModel):
    def __init__(self, config, model_argobj = None):
        super(Longformer_CLF_Dot_ANN,self).__init__(config)

        self.roberta = Longformer(config) 
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        attention_mask[:, [0,]] =  2
        
        # Special padding
        # input_ids, attention_mask = pad_to_window_size(
        #             input_ids, attention_mask, self.config.attention_window[0], 1) # pad id = 1, tokenizer.pad_token_id for roberta

        outputs1 = self.roberta(input_ids,
                                    attention_mask=attention_mask)
        # try separate embeddings
        compressed_output1 = self.embeddingHead(outputs1[0]) # [batch, len, dim]

        query1 = self.norm(compressed_output1[:, 0, :]) #CLF

        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0]
        return (loss.mean(),)


class Longformer_CLF_QK_ANN(Longformer_CLF_Dot_ANN):
    def __init__(self, config, model_argobj):
        super(Longformer_CLF_QK_ANN,self).__init__(config, model_argobj)
        self.KeyEmbeddingHead = nn.Linear(config.hidden_size, 768)
        
        self.apply(self._init_weights)
        self.config = config

    def body_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        attention_mask[:, [0,]] =  2
        # input_ids, attention_mask = pad_to_window_size(
        #             input_ids, attention_mask, self.config.attention_window[0], 1) # pad id = 1, tokenizer.pad_token_id for roberta

        outputs1 = self.roberta(input_ids,
                                    attention_mask=attention_mask)
        # try separate embeddings
        compressed_output1 = self.KeyEmbeddingHead(outputs1[0]) # [batch, len, dim]

        query1 = self.norm(compressed_output1[:, 0, :]) #CLF
        return query1        


class RobertaCNN8x_BM25NLL(RobertaBM25NLL):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 200, 4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.apply(self._init_weights)
    
    def _self_run_cnn(self, sequence_out):
        # sequence_out [batch, len, embedding]
        sequence_out = torch.transpose(sequence_out, 1,2) #[batch, embedding, len]
        x = sequence_out
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)

        x = torch.transpose(x, 1, 2)
        return x

    def _downsample_mask(self, mask, factor):
        [batchS, seq_len] = mask.size()
        mask = torch.reshape(mask, (batchS, seq_len // factor, factor))
        return mask[:, :, 0]

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        downsampled_mask = self._downsample_mask(attention_mask, 8).float()
        tu = (compressed_output1, compressed_output1[:,0])
        emb = self.masked_mean_or_first(tu, downsampled_mask)
        return emb

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        return super().forward(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)

class RobertaCNN8x_QK_BM25NLL(RobertaCNN8x_BM25NLL):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)
        self.downsample1 = nn.Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/2 compression, pending padding
        self.downsample2 = nn.Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/4 compression, pending padding
        self.downsample3 = nn.Conv1d(config.hidden_size, 400, 4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') # ~1/8 compression, pending padding
        
        self.apply(self._init_weights)

    def get_qk(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        compressed_output1 = self._self_run_cnn(outputs1[0]) # [batch, len/8, dim]
        [batchS, compressed_len, embeddingS] = compressed_output1.size()
        complex_emb1 = torch.reshape(compressed_output1, (batchS, compressed_len, 2, embeddingS//2))
        downsampled_mask = self._downsample_mask(attention_mask, 8).float()
        q_seq = complex_emb1[:,:,0,:]
        k_seq = complex_emb1[:,:,1,:]
        q_tu = (q_seq, q_seq[:,0,:])
        k_tu = (k_seq, k_seq[:,0,:])
        q_emb = self.masked_mean_or_first(q_tu, downsampled_mask)
        k_emb = self.masked_mean_or_first(k_tu, downsampled_mask)
        return q_emb, k_emb        

    def query_emb(self, input_ids, attention_mask):
        q, k = self.get_qk(input_ids, attention_mask)
        return q

    def body_emb(self, input_ids, attention_mask):
        q, k = self.get_qk(input_ids, attention_mask)
        return k

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        return super().forward(query_ids, attention_mask_q, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)

class RobertaCNN8x_QK_BM25BCE(RobertaCNN8x_QK_BM25NLL):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, labels):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        eval_logits = (q_embs*a_embs).sum(-1)
        eval_loss = self.bce_logit_loss(eval_logits, labels.float())
        return (eval_loss.mean(),)

class RobertaCNN8x_BM25BCE(RobertaCNN8x_BM25NLL):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, labels):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        eval_logits = (q_embs*a_embs).sum(-1)
        eval_loss = self.bce_logit_loss(eval_logits, labels.float())
        return (eval_loss.mean(),)

class RobertaDot_CLF_ANN(RobertaForSequenceClassification):
    def __init__(self, config, model_argobj):
        super().__init__(config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.bce_logit_loss = nn.BCEWithLogitsLoss(reduction='none')
 
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
 
    def query_emb(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs1 = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)
        # try separate embeddings
        # compressed_output1 = self.embeddingHead(outputs1[1]) # [batch, len/8, dim]
        # query1 = compressed_output1[:, :]
 
        compressed_output1 = self.embeddingHead(outputs1[0]) # [batch, len/8, dim]
        query1 = self.norm(compressed_output1[:, 0, :])
        return query1
 
    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a, labels):
        # query, key, value
        query1 = self.query_emb(query_ids, attention_mask_q) # output should be [batch, emb/3]
        key2 = self.body_emb(input_ids_a, attention_mask_a)

        a12 = torch.matmul(query1.unsqueeze(1), key2.unsqueeze(2)) # [batch, 1, 1]

        logits = a12[:, 0, 0]
        loss = self.bce_logit_loss(logits, labels.float())
        return (loss.mean(), logits)



