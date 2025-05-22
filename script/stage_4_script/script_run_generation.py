import pickle

import torch

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_Generation import Method_Generation

model_file_path = '../../result/stage_4_result/gru_generation_model.pth'
vocab_file_path = '../../result/stage_4_result/gru_vocab.pkl'
embedding_file_path = '../../result/stage_4_result/gru_embedding.pkl'


if __name__ == '__main__':
    with open(vocab_file_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(embedding_file_path, 'rb') as f:
        embedding = pickle.load(f)

    model = Method_Generation('rnn', '')
    model.data = {}
    model.data["vocab"] = vocab
    model.data["embedding"] = embedding
    model.build()
    model.load_state_dict(torch.load(model_file_path))
    model.eval()

    def run_model(tokens):
        token_ids = vocab(tokens)
        with torch.no_grad():
            logits = model(torch.tensor([token_ids], dtype=torch.long))
            probs = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()


        return vocab.get_itos()[predicted_id]

    last_tokens = None
    while True:
        cmd = input('input 3 words: ')
        if cmd == 'exit': break
        tokens = cmd.split()
        if len(tokens) == 2:
            if tokens[0] == 'cont':
                if last_tokens is None:
                    print('give input first')
                    continue
                n = int(tokens[1])

                for i in range(n):
                    last_tokens.append(run_model(last_tokens[-3:]))

                print(" ".join(last_tokens))
                continue
            else:
                print('invalid cmd')
                continue

        last_tokens = tokens

        if len(tokens) != 3:
            print('wrong input')
            continue

        print("next word: ", run_model(tokens))
