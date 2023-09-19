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


class MWDataset:
    def __init__(self, file_name, data_type):
        self.DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']
        # Only care domain in test
        with open(file_name, 'r') as f:
            data = json.load(f)
        self.data_type = data_type
        self.turn_labels = []  # store [SMUL1843.json_turn_1, ]
        self.turn_utts = []  # store corresponding text
        self.states = []  # store corresponding states. [['attraction-type-mueseum',],]
        self.input_text = []  # store prompt
        self.processing_data(data)

    def pocessing_data(self, data):
        raise NotImplementedError

    def make_prompt(self, history, current_state, data_type):
        prompt = '### Instruction: perfome the dialogue state tracking for the following dialogue.'
        if data_type != 'test':
            prompt += f'### Input: {history}'
            prompt += f'### Dialogue State: {current_state} [EOS]'
        else:
            prompt += f'### Input: {history}'
            prompt += f'### Dialogue State:'

        return prompt.replace("\n", " ")

    def as_dict(self, short=False):
        if short:
            return {'id': self.turn_labels[:10], 'text': self.input_text[:10], 'output': self.states[:10]}
        else:
            return {'id': self.turn_labels, 'text': self.input_text, 'label': self.states}

    def __len__(self):
        return len(self.input_text)

    def important_value_to_string(self, slot, value):
        if value in ["none", "dontcare"]:
            return f"[s]{slot}{value}"  # special slot
        return f"[s]{slot}-{value}"

    def _state_to_NL(self, slot_value_dict):
        output = "[CONTEXT] "
        for k, v in slot_value_dict.items():
            output += f"{' '.join(k.split('-'))}: {v.split('|')[0]}, "
        return output

    def make_history(self):
        raise NotImplementedError


class MWDataset_turn(MWDataset):
    def __init__(self, file_name, data_type):
        super().__init__(file_name, data_type)

    def processing_data(self, data):
        for turn in data:
            # filter the domains that not belongs to the test domain
            if not set(turn["domains"]).issubset(set(self.DOMAINS)):
                continue

            # update dialogue history
            sys_utt = turn["dialog"]['sys'][-1]
            usr_utt = turn["dialog"]['usr'][-1]

            if sys_utt == 'none':
                sys_utt = ''
            if usr_utt == 'none':
                usr_utt = ''

            context = turn["last_slot_values"]
            history = self.make_history(context, sys_utt, usr_utt)

            current_state = turn["turn_slot_values"]
            # convert to list of strings
            current_state = [self.important_value_to_string(s, v) for s, v in current_state.items()
                             if s.split('-')[0] in self.DOMAINS]
            current_state = ' '.join(current_state)
            self.turn_labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
            self.turn_utts.append(history)
            self.states.append(current_state)
            self.input_text.append(self.make_prompt
                                   (history, current_state, self.data_type))

        self.n_turns = len(self.turn_labels)
        print(f"there are {self.n_turns} turns in this dataset")

    def make_history(self, context_dict, sys_utt, usr_utt):
        history = self._state_to_NL(context_dict)
        if sys_utt == 'none':
            sys_utt = ''
        if usr_utt == 'none':
            usr_utt = ''
        history += f" [SYS] {sys_utt} [USER] {usr_utt}"
        return history


class MWDataset_dial(MWDataset):
    def __init__(self, file_name, data_type):
        super().__init__(file_name, data_type)

    def processing_data(self, data):
        dial_id, turn_num = '', 0
        for turn in data:
            pre_dial_id, pre_turn_num = dial_id, turn_num
            dial_id, turn_num = turn['ID'], int(turn['turn_id'])
            if turn_num != 0:
                assert pre_dial_id == dial_id
                assert pre_turn_num == turn_num-1

            # filter the domains that not belongs to the test domain
            if not set(turn["domains"]).issubset(set(self.DOMAINS)):
                continue

            # update dialogue history
            sys_utt = turn["dialog"]['sys']
            usr_utt = turn["dialog"]['usr']
            history = self.make_history(sys_utt, usr_utt)

            state = turn["slot_values"]
            # convert to list of strings
            state = [self.important_value_to_string(s, v) for s, v in state.items()
                     if s.split('-')[0] in self.DOMAINS]
            state = ' '.join(state)
            self.turn_labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
            self.turn_utts.append(history)
            self.states.append(state)
            self.input_text.append(self.make_prompt
                                   (history, state, self.data_type))

        self.n_turns = len(self.turn_labels)
        print(f"there are {self.n_turns} turns in this dataset")

    def make_history(self, sys_utt, usr_utt):
        history = ''
        for s, u in zip(sys_utt, usr_utt):
            history += f" [SYS] {s} [USER] {u}"
        return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # e.g. "../../data/mw21_10p_train_v3.json"
    parser.add_argument('--train_fn', type=str,
                        default="data/mw21_10p_train_v3.json",
                        help="training data file (few-shot or full shot)")

    args = parser.parse_args()

    dataset = MWDataset_dial(args.train_fn, data_type='train')
    dataset = dataset.as_dict()
    pdb.set_trace()
