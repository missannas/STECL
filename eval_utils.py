# This file contains the evaluation functions

import re
import editdistance

sentiment_word_list = ['positive', 'negative', 'neutral']
aspect_cate_list = ['location general',
 'food prices',
 'food quality',
 'ambience general',
 'service general',
 'restaurant prices',
 'drinks prices',
 'restaurant miscellaneous',
 'drinks quality',
 'drinks style_options',
 'restaurant general',
 'food style_options']


def extract_spans_extraction(seq):
    extractions = []
    all_pt = seq.split('; ')
    for pt in all_pt:
        try:
            a, b, c = pt.split(', ')
        except ValueError:
            a, b, c = '', '', ''
        extractions.append((a, b, c))
            
    return extractions


def extract_triplets(seq):
    aps = re.findall('\[.*?\]', seq)
    aps = [ap[1:-1] for ap in aps]
    triplets = []
    for ap in aps:
        try:
            a, b, c = ap.split('|')
        except ValueError:
            a, b, c = '', '', ''
        
        # for ASTE
        if b in sentiment_word_list:
            if ',' in c:
                for op in c.split(', '):
                    triplets.append((a, b, op))
            else:
                triplets.append((a, b, c))
        # for acsd
        else:
            if ',' in b:
                for ac in b.split(', '):
                    triplets.append((a, ac, c))
            else:
                triplets.append((a, b, c))

    return triplets


def recover_terms_with_editdistance(original_term, sent):
    words = original_term.split(' ')
    new_words = []
    for word in words:
        edit_dis = []
        for token in sent:
            edit_dis.append(editdistance.eval(word, token))#计算editdistance
        smallest_idx = edit_dis.index(min(edit_dis))
        new_words.append(sent[smallest_idx])
    new_term = ' '.join(new_words)
    return new_term





# for ASTE
def fix_preds_aste(all_pairs, sents):
    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:  # none
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # two formats have different orders
                sents_and_null = ' '.join(sents[i]) + 'NULL'
                p0, p1, p2 = pair
                at = ''
                if p2 in sentiment_word_list:
                    at, ot, ac = p0, p1, p2
                    io_format = 'extraction'
                else:
                    print(pair)

                if at not in sents_and_null:
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(at, sents[i])
                else:
                    new_at = at

                if ac not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(ac, sentiment_word_list)
                else:
                    new_sentiment = ac

                # OT not in the original sentence
                if ot not in sents_and_null:
                    # print('Issue')
                    new_ot = recover_terms_with_editdistance(ot, sents[i])
                else:
                    new_ot = ot
                if io_format == 'extraction':
                    new_pairs.append((new_at, new_ot, new_sentiment))
                else:
                    new_pairs.append((new_at, new_sentiment, new_ot))
            all_new_pairs.append(new_pairs)
    return all_new_pairs


def fix_preds_acsd(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                #print(pair)
                # AT not in the original sentence
                sents_and_null = ' '.join(sents[i]) + 'NULL'
                if pair[0] not in sents_and_null:
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]
                
                # AC not in the list
                if pair[1] not in aspect_cate_list:
                    # print('Issue')
                    new_ac = recover_terms_with_editdistance(pair[1], aspect_cate_list)
                else:
                    new_ac = pair[1]
                
                if pair[2] not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(pair[2], sentiment_word_list)
                else:
                    new_sentiment = pair[2]
            
                new_pairs.append((new_at, new_ac, new_sentiment))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)
    
    return all_new_pairs


def fix_pred_with_editdistance(all_predictions, sents, task):
    if task == 'aste':
        fixed_preds = fix_preds_aste(all_predictions, sents)
    elif task == 'acsd':
        fixed_preds = fix_preds_acsd(all_predictions, sents)
    else:
        print("*** Unimplemented Error ***")
        fixed_preds = all_predictions

    return fixed_preds


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sents, task):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs) 
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []

    for i in range(num_samples):
        gold_list = extract_spans_extraction(gold_seqs[i])
        pred_list = extract_spans_extraction(pred_seqs[i])
            

        all_labels.append(gold_list)
        all_predictions.append(pred_list)
    
    print("\nResults of raw output")
    raw_scores = compute_f1_scores(all_predictions, all_labels)
    print(raw_scores)

    # fix the issues due to generation
    all_predictions_fixed = fix_pred_with_editdistance(all_predictions, sents, task)
    print("\nResults of fixed output")
    fixed_scores = compute_f1_scores(all_predictions_fixed, all_labels)
    print(fixed_scores)
    
    return raw_scores, fixed_scores, all_labels, all_predictions, all_predictions_fixed