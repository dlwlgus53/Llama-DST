import numpy as np
import json
import random
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import pdb
import os
# input arguments

'''
TRAIN_FN = args.train_fn

SAVE_NAME = args.save_name

PRETRAINED_MODEL = args.pretrained_index_dir
MODEL_NAME = args.pretrained_model

EPOCH = args.epoch
TOPK = args.topk
TOPRANGE = args.toprange
# ------------ CONFIG ends here ------------
with open(TRAIN_FN) as f:
    train_set = json.load(f)

# prepare pretrained retreiver for fine-tuning
pretrained_train_retriever = IndexRetriever(datasets=[train_set],
                                            embedding_filenames=[
    f"{PRETRAINED_MODEL_SAVE_PATH}/mw21_train_{PRETRAINED_MODEL}.npy"],
    search_index_filename=f"{PRETRAINED_MODEL_SAVE_PATH}/mw21_train_{PRETRAINED_MODEL}.npy",
    sampling_method="pre_assigned",
)

'''


class MWDataset_turn:

    def __init__(self, file_name):

        # Only care domain in test
        DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']
        with open(file_name, 'r') as f:
            data = json.load(f)

        self.turn_labels = []  # store [SMUL1843.json_turn_1, ]
        self.turn_utts = []  # store corresponding text
        self.turn_states = []  # store corresponding states. [['attraction-type-mueseum',],]

        for turn in data:
            # filter the domains that not belongs to the test domain
            if not set(turn["domains"]).issubset(set(DOMAINS)):
                continue

            # update dialogue history
            sys_utt = turn["dialog"]['sys'][-1]
            usr_utt = turn["dialog"]['usr'][-1]

            if sys_utt == 'none':
                sys_utt = ''
            if usr_utt == 'none':
                usr_utt = ''

            context = turn["last_slot_values"]
            history = self.input_to_string(context, sys_utt, usr_utt)

            current_state = turn["turn_slot_values"]
            # convert to list of strings
            current_state = [self.important_value_to_string(s, v) for s, v in current_state.items()
                             if s.split('-')[0] in DOMAINS]

            self.turn_labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
            self.turn_utts.append(history)
            self.turn_states.append(current_state)

        self.n_turns = len(self.turn_labels)
        print(f"there are {self.n_turns} turns in this dataset")

    def as_dict(self):
        return {'id': self.turn_labels, 'text': self.turn_utts, 'label': self.turn_states}

    def __len__(self):
        return self.n_turns

    def __getitem__(self, idx):
        # TODO add tokenizer, perform padding
        return {
            "turn_label": self.turn_labels[idx],
            "turn_utt": self.turn_utts[idx],
            "turn_state": self.turn_states[idx]
        }

    def collate_fn(self, batch):
        return {k: [d[k] for d in batch] for k in batch[0]}

    def important_value_to_string(self, slot, value):
        if value in ["none", "dontcare"]:
            return f"{slot}{value}"  # special slot
        return f"{slot}-{value}"

    def _state_to_NL(self, slot_value_dict):
        output = "[CONTEXT] "
        for k, v in slot_value_dict.items():
            output += f"{' '.join(k.split('-'))}: {v.split('|')[0]}, "
        return output

    def input_to_string(self, context_dict, sys_utt, usr_utt):
        history = self._state_to_NL(context_dict)
        if sys_utt == 'none':
            sys_utt = ''
        if usr_utt == 'none':
            usr_utt = ''
        history += f" [SYS] {sys_utt} [USER] {usr_utt}"
        return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # e.g. "../../data/mw21_10p_train_v3.json"
    parser.add_argument('--train_fn', type=str,
                        default="data/mw21_10p_train_v3.json",
                        help="training data file (few-shot or full shot)")

    args = parser.parse_args()

    dataset = MWDataset(args.train_fn)
    data_loader = DataLoader(dataset, batch_size=4,
                             shuffle=False, collate_fn=dataset.collate_fn)

    for batch in data_loader:
        pdb.set_trace()
