
from transformers import pipeline
from tqdm import tqdm
import json
import pdb
from config import CONFIG


def str_to_dict(result):
    result_dict = {}
    dsv = result.split('[s]')
    for dsv_item in dsv:
        if len(dsv_item.strip()) == 0:
            continue
        else:
            dsv_item = dsv_item.strip()
            d, s, v = dsv_item.split(
                '-')[0], dsv_item.split('-')[1], dsv_item.split('-')[2]
            ds = d + '-' + s
            result_dict[ds] = v
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


def run(args, test_data, model, tokenizer):

    generator = pipeline('text-generation', model=model, num_beams=args.beam_size,
                         do_sample=False, tokenizer=tokenizer)
    MAX_NEW_LENGTH = 128

    # to no gradient
    model.eval()

    preds, labels = [], []

    for text, gold in zip(tqdm(test_data['text']), test_data['label']):
        pred = []
        generated_texts = generator(
            text, max_new_tokens=MAX_NEW_LENGTH, num_return_sequences=args.beam_size)
        for generated_text in generated_texts:
            generated_text = generated_text['generated_text']
            s_idx = generated_text.find(
                "### Dialogue State: ") + len("### Dialogue State: ")
            e_idx = generated_text.find("[EOS]")
            p = generated_text[s_idx:e_idx]
            # p = str_to_dict(p)
            # p = typo_fix(p, ontology=ontology, version=args.mwz_ver)
            pred.append(p)

        preds.append(pred)
        labels.append(gold)
        # labels.append(str_to_dict(gold))

    return test_data['id'], preds, labels

    # # save with gold
    # new_preds, new_labels = stack_result(test_data['id'], preds, labels)
    # result = [
    #     {'id': id,
    #      'stack_preds': p,
    #      'stack_gold': g,
    #      'turn_preds': p_t,
    #      'turn_gold': g_t}
    #     for (id, p, g, p_t, g_t) in zip(test_data['id'], new_preds, new_labels, preds, labels)
    # ]
    # with open(f"{args.save_dir}/result.json", 'w') as f:
    #     json.dump(result, f, indent=4)
