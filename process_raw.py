import json
import argparse
import pdb
from utils.typo_fix import typo_fix
from config import CONFIG

parser = argparse.ArgumentParser()


parser.add_argument('--mwz_ver', type=str, required=True,
                    help="training data file (few-shot or full shot)")
parser.add_argument('--save_dir', type=str, required=True,
                    help="training data file (few-shot or full shot)")
args = parser.parse_args()


if args.mwz_ver == '2.1':
    ontology_path = CONFIG["ontology_21"]
else:
    ontology_path = CONFIG["ontology_24"]
with open(ontology_path) as f:
    ontology = json.load(f)


def str_to_dict(result):
    result_dict = {}
    dsv = result.split('[s]')
    for dsv_item in dsv:
        if len(dsv_item.strip()) == 0:
            continue
        else:
            try:
                dsv_item = dsv_item.strip()
                d, s, v = dsv_item.split(
                    '-')[0], dsv_item.split('-')[1], dsv_item.split('-')[2]
                ds = d + '-' + s
                result_dict[ds] = v

            except:
                print("parsing error")
                continue
    result_dict = typo_fix(
        result_dict, ontology=ontology, version=args.mwz_ver)
    return result_dict


def stack_result(ids, preds, labels):
    stack_preds, stack_labels = [], []
    p_dial, p_turn = '', ''
    for id, pred, label in zip(ids, preds, labels):
        dial, turn = id.split('_turn_')
        if turn == '0':
            stack_preds.append(pred)
            stack_labels.append(label)
        else:
            assert p_dial == dial
            assert int(p_turn) == int(turn)-1
            last_label = stack_labels[-1].copy()
            last_label.update(label)
            stack_labels.append(last_label)

            last_preds = []
            for pre in pred:
                last_pred = stack_preds[-1][0].copy()
                last_pred.update(pre)
                last_preds.append(last_pred)

            stack_preds.append(last_preds)
        p_dial, p_turn = dial, turn

    return stack_preds, stack_labels


def save_result(ids, stack_preds, stack_labels, preds, labels):

    # save with gold
    result = [
        {'id': id,
         'stack_preds': p,
         'stack_gold': g,
         'turn_preds': p_t,
         'turn_gold': g_t}
        for (id, p, g, p_t, g_t) in zip(ids, stack_preds, stack_labels, preds, labels)
    ]
    with open(f"{args.save_dir}/result.json", 'w') as f:
        json.dump(result, f, indent=4)


def r_duplicate(dict_list):

    unique_list = []

    for d in dict_list:
        if d not in unique_list:
            unique_list.append(d)

    return unique_list


def calculate_JGA(preds, labels):
    all = len(preds)
    correct = 0
    for p, l in zip(preds, labels):
        if p[0] == l:
            correct += 1
    return correct/all


def check_beam_JGA(preds, labels):
    all = len(preds)
    correct = 0
    for p, l in zip(preds, labels):
        if l in p:
            correct += 1
    return correct/all


if __name__ == '__main__':
    raw_file = args.save_dir + '/raw_result.json'
    score_file = args.save_dir + '/score.json'

    raw_file = json.load(open(raw_file))
    stack_pred, stack_gold = [], []
    ids = [item['id'] for item in raw_file]
    labels = [str_to_dict(item['gold']) for item in raw_file]
    preds = [item['pred']for item in raw_file]
    new_preds = []
    for pred in preds:
        new_pred = r_duplicate([str_to_dict(p) for p in pred])
        new_preds.append(new_pred)
    preds = new_preds
    stack_preds, stack_labels = stack_result(ids, preds, labels)
    save_result(ids, stack_preds, stack_labels, preds, labels)

    score = json.load(open(score_file))

    score['beam_JGA'] = check_beam_JGA(stack_preds, stack_labels)
    score['JGA'] = calculate_JGA(stack_preds, stack_labels)

    print(score)
    with open(f"{args.save_dir}/score.json", 'w') as f:
        json.dump(score, f, indent=4)
