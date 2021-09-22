# coding=utf-8
# Copyright 2020 Microsoft and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization class for model DeBERTa."""

import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as sp
import six

from transformers.tokenization_utils import PreTrainedTokenizer
from tokenizers import BertWordPieceTokenizer, normalizers, pre_tokenizers
import re

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/seed-encoder-3-layer-decoder": "./vocab.txt",
        "microsoft/seed-encoder-1-layer-decoder": "./vocab.txt"
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/seed-encoder-3-layer-decoder": 512,
    "microsoft/seed-encoder-1-layer-decoder": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/seed-encoder-3-layer-decoder": {"do_lower_case": False},
    "microsoft/seed-encoder-1-layer-decoder": {"do_lower_case": False},
}

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}



class SEEDTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a DeBERTa-v2 tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.
    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to lowercase the input when tokenizing.
        bos_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The end of sequence token. When building a sequence using special tokens, this is not the token that is
            used for the end of sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        sp_model_kwargs (:obj:`dict`, `optional`):
            Will be passed to the ``SentencePieceProcessor.__init__()`` method. The `Python wrapper for SentencePiece
            <https://github.com/google/sentencepiece/tree/master/python>`__ can be used, among other things, to set:
            - ``enable_sampling``: Enable subword regularization.
            - ``nbest_size``: Sampling parameters for unigram. Invalid for BPE-Dropout.
              - ``nbest_size = {0,1}``: No sampling is performed.
              - ``nbest_size > 1``: samples from the nbest_size results.
              - ``nbest_size < 0``: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.
            - ``alpha``: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="<mask>",
        fb_model_kwargs:Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        self.fb_model_kwargs = {} if fb_model_kwargs is None else fb_model_kwargs

        super().__init__(
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            fb_model_kwargs=self.fb_model_kwargs,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.do_lower_case = do_lower_case
        
        #self._tokenizer = BertWordPieceTokenizer(vocab_file, clean_text=False, strip_accents=False, lowercase=False)
        self._tokenizer = FastBERTTokenizer(vocab_file, fb_model_kwargs=self.fb_model_kwargs)

        #print('???',self.cls_token_id,self.sep_token_id,self.pad_token_id)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.get_added_vocab())
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        if self.do_lower_case:
            escaped_special_toks = [re.escape(s_tok) for s_tok in ['[CLS]','[PAD]','[UNK]','[SEP]']]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)
        return self._tokenizer.tokenize(text)


    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        #return self._tokenizer.spm.PieceToId(token)
        return self._tokenizer._convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        #return self._tokenizer.spm.IdToPiece(index) if index < self.vocab_size else self.unk_token
        return self._tokenizer._convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        return self._tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:
        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return self._tokenizer.save_pretrained(save_directory, filename_prefix=filename_prefix)

    def encode(self,full_text, add_special_tokens,max_length):
        if self.do_lower_case:
            escaped_special_toks = [re.escape(s_tok) for s_tok in ['[CLS]','[PAD]','[UNK]','[SEP]']]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            full_text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), full_text)
        return self._tokenizer.bert_tokenizer.encode(full_text, add_special_tokens=add_special_tokens).ids[:max_length]



class FastBERTTokenizer:
    r"""
    Constructs a tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`__.
    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        sp_model_kwargs (:obj:`dict`, `optional`):
            Will be passed to the ``SentencePieceProcessor.__init__()`` method. The `Python wrapper for SentencePiece
            <https://github.com/google/sentencepiece/tree/master/python>`__ can be used, among other things, to set:
            - ``enable_sampling``: Enable subword regularization.
            - ``nbest_size``: Sampling parameters for unigram. Invalid for BPE-Dropout.
              - ``nbest_size = {0,1}``: No sampling is performed.
              - ``nbest_size > 1``: samples from the nbest_size results.
              - ``nbest_size < 0``: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.
            - ``alpha``: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    def __init__(self, vocab_file, fb_model_kwargs: Optional[Dict[str, Any]] = None):
        self.vocab_file = vocab_file
        self.fb_model_kwargs = {} if fb_model_kwargs is None else fb_model_kwargs
        
        assert os.path.exists(vocab_file), "no existing vocab file."
        
        #spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)   
        #spm.load(vocab_file)
        #bpe_vocab_size = spm.GetPieceSize()
        #self.spm = spm

        self.bert_tokenizer = BertWordPieceTokenizer(vocab_file, clean_text=False, strip_accents=False, lowercase=False)

        self.vocab={}
        self.ids_to_tokens=[]
        self.add_from_file(open(vocab_file,'r'))
        self.add_symbol('<mask>')
        self.vocab_size=len(self.vocab)

    
    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """

        lines = f.readlines()

        for line in lines:
            line = line.rstrip()
            word = line
            self.add_symbol(word, overwrite=False)         

    def add_symbol(self, word, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.vocab and not overwrite:
            idx = self.vocab[word]
            return idx
        else:
            idx = len(self.ids_to_tokens)
            self.vocab[word] = idx
            self.ids_to_tokens.append(word)
            return idx

    def tokenize(self, text):
        return self.bert_tokenizer.encode(text, add_special_tokens=False).tokens
        

    def convert_ids_to_tokens(self, index):
        return self.ids_to_tokens[index] if index < self.vocab_size else self.unk

    def _convert_token_to_id(self,token):
        return self.vocab[token]


    def decode(self, x:str) ->str:
        return self.bert_tokenizer.decode([
            int(tok) for tok in x.split()
        ])

    def pad(self):
        return "[PAD]"

    def bos(self):
        return "[CLS]"

    def eos(self):
        return "[SEP]"

    def unk(self):
        return "[UNK]"

    def mask(self):
        return "<mask>"

    def sym(self, id):
        return self.ids_to_tokens[id]

    def id(self, sym):
        return self.vocab[sym] if sym in self.vocab else 1

    
    def save_pretrained(self, path: str, filename_prefix: str = None):
        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        if filename_prefix is not None:
            filename = filename_prefix + "-" + filename
        full_path = os.path.join(path, filename)
        with open(full_path, "w") as fs:
            #fs.write(self.spm.serialized_model_proto())
            for item in self.ids_to_tokens:
                fs.write(str(item)+'\n')
        return (full_path,)
        #pass


    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError(f"Unsupported string type: {type(text)}")
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError(f"Unsupported string type: {type(text)}")
    else:
        raise ValueError("Not running on Python2 or Python 3?")