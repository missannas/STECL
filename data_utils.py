# This file contains all data loading and transformation functions

import time
from torch.utils.data import Dataset
from transformers import T5Tokenizer

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}


def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    raw, sents, labels = [], [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                raw.append(words.split())
                sents.append(words.split())
                labels.append(eval(tuples))
    print(f"Total examples = {len(sents)}")
    return sents, labels


def get_extraction_acsd_targets(sents, labels):
    targets = []
    for label in labels:
        label_strs = [', '.join(l) for l in label]
        target = '; '.join(label_strs)
        targets.append(target)
    return targets


def get_extraction_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri = []
        for tri in label:
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])
            c = senttag2word[tri[2]]
            all_tri.append((a, b, c))
        label_strs = [', '.join(l) for l in all_tri]
        targets.append('; '.join(label_strs))
    return targets


def get_transformed_io(data_path, task):
    """
    The main function to transform the Input & Output according to 
    the specified task
    """
    sents, labels = read_line_examples_from_file(data_path)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    # Get target
    if task == 'aste':
        targets = get_extraction_aste_targets(sents, labels)
    elif task == 'acsd':
        targets = get_extraction_acsd_targets(sents, labels)
    else:
        raise NotImplementedError

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, task, max_len=128):
        # 'data/aste/rest16/train.txt'
        self.data_path = f'data/{task}/{data_dir}/{data_type}.txt'
        self.task = task
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()

        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()      # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.task)

        for i in range(len(inputs)):

            input = ' '.join(inputs[i]) 
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


def write_results_to_log(log_file_path, best_test_result, args, dev_results, test_results, global_steps):
    """
    Record dev and test results to log file
    """
    local_time = time.asctime(time.localtime(time.time()))
    exp_settings = "Exp setting: {0} on {1} | {2:.4f} | ".format(
        args.task, args.dataset, best_test_result
    )
    train_settings = "Train setting: bs={0}, lr={1}, num_epochs={2}".format(
        args.train_batch_size, args.learning_rate, args.num_train_epochs
    )
    results_str = "\n* Results *:  Dev  /  Test  \n"

    metric_names = ['f1', 'precision', 'recall']
    for gstep in global_steps:
        results_str += f"Step-{gstep}:\n"
        for name in metric_names:
            name_step = f'{name}_{gstep}'
            results_str += f"{name:<8}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}"
            results_str += ' '*5
        results_str += '\n'

    log_str = f"{local_time}\n{exp_settings}\n{train_settings}\n{results_str}\n\n"

    with open(log_file_path, "a+") as f:
        f.write(log_str)


def get_dataset(tokenizer, type_path, dataset, task, max_seq_length):
    return ABSADataset(tokenizer=tokenizer, data_dir=dataset, data_type=type_path,
                       task=task, max_len=max_seq_length)


if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    data = get_dataset(tokenizer, "trains", 'rest14', 'aste', 128)
    print(data[0]['target_ids'].size())
    print(data[0]['target_ids'])
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in data[0]['target_ids'][:, 1:]]
    print(texts)
