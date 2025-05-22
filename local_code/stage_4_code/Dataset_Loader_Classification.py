import glob
import os
from typing import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe

from local_code.base_class.dataset import dataset


def truncate_head_tail(seq, max_len):
    if len(seq) <= max_len:
        return seq
    head_len = max_len // 2
    tail_len = max_len - head_len
    return seq[:head_len] + seq[-tail_len:]

class Dataset_Loader_Classification(dataset):
    data = None
    dataset_source_folder_path = "../../data/stage_4_data/text_classification/"

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def read_data(self, dataset_name: str):
        texts, labels = [], []

        for label in ['pos', 'neg']:
            path = os.path.join(self.dataset_source_folder_path, dataset_name, label)
            if os.path.exists(path):
                files = glob.glob(os.path.join(path, '*.txt'))
                for file_path in files:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        texts.append(text)
                        labels.append(1 if label == 'pos' else 0)

        return texts, labels

    @staticmethod
    def yield_tokens(texts: list[str], tokenizer) -> Iterable[list[str]]:
        for text in texts:
            yield tokenizer(text)

    @staticmethod
    def pad_sequence(seq, vocab, max_len):
        if len(seq) < max_len:
            return seq + [vocab['<pad>']] * (max_len - len(seq))
        return seq[:max_len]

    def load(self):
        print(f'loading dataset {self.dataset_name} ...')

        train_texts, train_labels = self.read_data("train")
        test_texts, test_labels = self.read_data("test")

        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(
            self.yield_tokens(train_texts, tokenizer),
            specials=["<unk>", "<pad>"]
        )
        vocab.set_default_index(vocab['<unk>'])

        glove = GloVe(name='6B', dim=100)
        embedding_matrix = torch.zeros(len(vocab), glove.dim)

        for idx, token in enumerate(vocab.get_itos()):
            if token in glove.stoi:
                embedding_matrix[idx] = glove[token]

        text_pipeline = lambda x: vocab(tokenizer(x))
        tokenized_train = [text_pipeline(text) for text in train_texts]
        tokenized_test = [text_pipeline(text) for text in test_texts]

        max_seq_len = 256

        tokenized_train = [self.pad_sequence(truncate_head_tail(seq, max_seq_len), vocab, max_seq_len) for seq in tokenized_train]
        tokenized_test = [self.pad_sequence(truncate_head_tail(seq, max_seq_len), vocab, max_seq_len) for seq in tokenized_test]

        train_dataset = TensorDataset(torch.tensor(tokenized_train), torch.tensor(train_labels))
        test_dataset = TensorDataset(torch.tensor(tokenized_test), torch.tensor(test_labels))

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
        )

        embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=vocab['<pad>'])

        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'embedding': embedding_layer,
            'vocab': vocab
        }