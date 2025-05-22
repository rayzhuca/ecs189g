import glob
import itertools
import os
from typing import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe

from local_code.base_class.dataset import dataset

class Dataset_Loader_Generation(dataset):
    data = None
    dataset_source_file_path = "../../data/stage_4_data/text_generation/data"

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def read_data(self, k, tokenizer):
        X, y = [], []
        max_joke = 0
        with open(self.dataset_source_file_path, "r", encoding="utf-8") as f:
            for i, joke in enumerate(f):
                if i == 0: continue
                joke = joke.split(",")[1]
                tokens = tokenizer(joke)
                max_joke = max(max_joke, len(tokens))
                for i in range(len(tokens) - k):
                    X.append(tokens[i: i + k])
                    y.append(tokens[i + k])
        print('max_joke', max_joke)
        return X, y


    @staticmethod
    def yield_tokens(texts: list[str], tokenizer) -> Iterable[list[str]]:
        for text in texts:
            yield tokenizer(text)

    def load(self):
        print(f'loading dataset {self.dataset_name} ...')

        tokenizer = get_tokenizer('basic_english')
        X, y = self.read_data(3, tokenizer)

        vocab = build_vocab_from_iterator(
            itertools.chain(*X, [[label] for label in y]),
            specials=["<unk>", "<pad>"]
        )
        vocab.set_default_index(vocab['<unk>'])

        glove = GloVe(name='6B', dim=100)
        embedding_matrix = torch.zeros(len(vocab), glove.dim)

        for idx, token in enumerate(vocab.get_itos()):
            if token in glove.stoi:
                embedding_matrix[idx] = glove[token]

        X = [vocab(joke) for joke in X]
        y = [vocab[label] for label in y]

        data_loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y)),
            batch_size=32,
            shuffle=True,
            num_workers=4,
        )

        embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=vocab['<pad>'])

        return {
            'data_loader': data_loader,
            'embedding': embedding_layer,
            'vocab': vocab
        }