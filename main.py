import argparse
import os
import logging
import pickle
from tqdm import tqdm
import random

import torch

import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer

from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import write_results_to_log, read_line_examples_from_file
from eval_utils import compute_scores, extract_spans_extraction, compute_f1_scores
logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='aste', type=str, 
                        help="The name of the task, selected from: [aste, acsd]")
    parser.add_argument("--dataset", default='rest14', type=str, 
                        help="The name of the dataset, selected from: [laptop14, rest14, rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # CL parameters
    parser.add_argument('--k', type=int, default=1, help="numbers of generate negative example")
    parser.add_argument('--T', type=float, default=0.07, help="tempreture")
    parser.add_argument('--cl', type=bool, default=False, help="whether or not use constractive learning")
    parser.add_argument("--element", default='all', type=str,
                        help="The name of the elements in triple, selected from: [aspect, opinion, cate, pola, ins, all]")
    parser.add_argument('--instance_weight', type=float, default=1.0, help="weight of contrastive learning loss ")
    parser.add_argument('--tri_weight', type=float, default=0.5, help="weight of contrastive learning loss ")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--dropt", default=0.1, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './aste/rest14/extraction/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    task_dir = f"./outputs/{args.task}"
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    task_dataset_dir = f"{task_dir}/{args.dataset}"
    if not os.path.exists(task_dataset_dir):
        os.mkdir(task_dataset_dir)

    args.output_dir = task_dataset_dir

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type=type_path, 
                       task=args.task, max_len=args.max_seq_length)

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

class STECL(pl.LightningModule):
    def __init__(self, hparams):
        super(STECL, self).__init__()
        self.hparams = hparams
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        self.d_model = self.model.model_dim
        self.projection = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU())
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.con_loss = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.dropout = nn.Dropout(hparams.dropt)
        self.k = self.hparams.k

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None, cl=False):

        neg_cl = decoder_input_ids.clone()
        pos_cl = decoder_input_ids.clone()
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        encoder_out = encoder(input_ids, attention_mask)
        hidden_states = encoder_out[0]
        decoder_out = decoder(input_ids=decoder_input_ids,
                              attention_mask=decoder_attention_mask,
                              encoder_hidden_states=hidden_states,
                              encoder_attention_mask=attention_mask,
                              )
        sequence_out = decoder_out[0]
        sequence_out = sequence_out * (self.d_model ** -0.5)
        lm_logits = self.model.lm_head(sequence_out)

        vocab_size = lm_logits.size(-1)

        nll = self.loss_fct(lm_logits.view(-1, vocab_size), labels.view(-1))
        if cl:
            proj_enc_h = self.projection(hidden_states)
            avg_ini = self.avg_pool(proj_enc_h, attention_mask)

            pos_out = encoder(pos_cl, decoder_attention_mask)[0]
            pro_pos = self.projection(pos_out)
            avg_pos = self.avg_pool(pro_pos, decoder_attention_mask)

            l_pos = torch.exp(torch.sum(avg_ini * avg_pos, dim=-1) / self.hparams.T)

            for j in range(self.k):
                neg_tri, tri_mask = self.get_neg_token(neg_cl, input_ids, "ins")
                neg_id = self.model._shift_right(neg_tri)
                neg_out = encoder(neg_id.cuda(), tri_mask.cuda())[0]

                pro_neg = self.projection(neg_out)
                avg_neg = self.avg_pool(pro_neg, tri_mask)

                if j == 0:
                    l_ins = torch.exp(torch.sum(avg_ini * avg_neg, dim=-1) / self.hparams.T)
                else:
                    i_ins = torch.exp(torch.sum(avg_ini * avg_neg, dim=-1) / self.hparams.T)
                    l_ins = torch.add(l_ins, i_ins)

            loss_ins = self.hparams.instance_weight * torch.mean(-torch.log(l_pos / (l_pos + l_ins)))

            l_tri = self.get_all(avg_ini, neg_cl, input_ids, encoder)

            # norm_sum = torch.exp(torch.ones(avg_ini.size(0))/self.hparams.T).cuda()
            loss_tri = self.hparams.tri_weight * torch.mean(-torch.log(l_pos / (l_pos + l_tri)))

            loss = nll + loss_ins + loss_tri
            return loss
        else:
            return nll

    def avg_pool(self, tokens, mask):
        lens = torch.sum(mask, 1, keepdim=True).float().cuda()
        mask = mask.unsqueeze(2).cuda()
        to = tokens.masked_fill(mask == 0, 0.0)
        avg_token = torch.sum(to, 1)/lens
        return avg_token

    def sent(self, tokens):
        return tokens[:, 0]

    def replace_e(self, rel_idx, init_t, batch_t):
        listt = init_t.split(' ')
        randid = random.randint(0, len(listt) - 1)
        ran = listt[randid]
        if ';' not in batch_t:
            tri = batch_t.split(', ')
            tri[rel_idx] = ran
            rep_tri = ', '.join(tri)
        else:
            tri = batch_t.split(';')
            t = []
            for tridx in tri:
                x = tridx.split(', ')
                x[rel_idx] = ran
                xl = ', '.join(x)
                t.append(xl)
            rep_tri = '; '.join(t)
        return rep_tri

    def replace_s(self, rel_idx, batch_t, rep_set):
        listt = rep_set.copy()
        if ';' not in batch_t:
            tri = batch_t.split(', ')
            listt.remove(tri[rel_idx])
            randid = random.randint(0, len(listt) - 1)
            ran = listt[randid]
            tri[rel_idx] = ran
            rep_tri = ', '.join(tri)
            listt = rep_set.copy()
        else:
            tri = batch_t.split(';')
            t = []
            # print(batch_t)
            for tridx in tri:
                x = tridx.split(', ')
                # print('tri is:'+tridx)
                # print('x is:' + x[rel_idx])
                # print(listt)
                listt.remove(x[rel_idx])
                randid = random.randint(0, len(listt) - 1)
                ran = listt[randid]
                x[rel_idx] = ran
                xl = ', '.join(x)
                t.append(xl)
                listt = rep_set.copy()
            rep_tri = '; '.join(t)
        return rep_tri

    def replace_ins(self, n_batch, text, idx):
        randid = random.randint(0, n_batch - 1)
        if randid == idx:
            randid = random.randint(0, n_batch - 1)
        rep_tri = text[randid]
        return rep_tri

    def get_all(self, ini, neg, inputs, encoder):
        neg_token_a, neg_mask_a = self.get_neg_token(neg, inputs, "aspect")
        neg_id_a = self.model._shift_right(neg_token_a)
        neg_out_a = encoder(neg_id_a.cuda(), neg_mask_a.cuda())[0]
        pro_neg_a = self.projection(neg_out_a)
        avg_neg_a = self.avg_pool(pro_neg_a, neg_mask_a)
        a_neg = torch.exp(torch.sum(ini * avg_neg_a, dim=-1) / self.hparams.T)

        neg_token_p, neg_mask_p = self.get_neg_token(neg, inputs, "pola")
        neg_id_p = self.model._shift_right(neg_token_p)
        neg_out_p = encoder(neg_id_p.cuda(), neg_mask_p.cuda())[0]
        pro_neg_p = self.projection(neg_out_p)
        avg_neg_p = self.avg_pool(pro_neg_p, neg_mask_p)
        p_neg = torch.exp(torch.sum(ini * avg_neg_p, dim=-1) / self.hparams.T)
        a_neg = torch.add(a_neg, p_neg)

        ele = "opinion"
        if self.hparams.task == "acsd":
            ele = "cate"

        neg_token_o, neg_mask_o = self.get_neg_token(neg, inputs, ele)
        neg_id_o = self.model._shift_right(neg_token_o)
        neg_out_o = encoder(neg_id_o.cuda(), neg_mask_o.cuda())[0]
        pro_neg_o = self.projection(neg_out_o)
        avg_neg_o = self.avg_pool(pro_neg_o, neg_mask_o)
        o_neg = torch.exp(torch.sum(ini * avg_neg_o, dim=-1) / self.hparams.T)
        a_neg = torch.add(a_neg, o_neg)

        return a_neg


    def get_neg_token(self, token, ini, rep):
        texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in token[:, :]]
        init = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in ini]

        n_batch = token.size(0)
        n_token = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')
        n_mask = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')

        for i in range(n_batch):
            batch_t = texts[i]
            ini_t = init[i]

            if rep == 'aspect':
                rep_tri = self.replace_e(rel_idx=0, init_t=ini_t, batch_t=batch_t)
            elif rep == 'opinion':
                rep_tri = self.replace_e(rel_idx=1, init_t=ini_t, batch_t=batch_t)
            elif rep == 'cate':
                rep_tri = self.replace_s(rel_idx=-2, batch_t=batch_t, rep_set=aspect_cate_list)
            elif rep == 'pola':
                rep_tri = self.replace_s(rel_idx=-1, batch_t=batch_t, rep_set=sentiment_word_list)
            elif rep == 'ins':
                rep_tri = self.replace_ins(n_batch=n_batch, text=texts, idx=i)
            else:
                pass

            tt = self.tokenizer.batch_encode_plus([rep_tri], max_length=self.hparams.max_seq_length,
                                                  pad_to_max_length=True, truncation=True, return_tensors="pt")
            input = tt['input_ids']
            mask = tt['attention_mask']
            if i == 0:
                n_token = input
                n_mask = mask
            else:
                n_token = torch.cat((n_token, input), 0)
                n_mask = torch.cat((n_mask, mask), 0)
        return n_token, n_mask

    def compute(self, pred_seqs, gold_seqs, task):
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
        value = raw_scores['f1']
        return value

    def evl(self, batch):
        """
        Compute scores given the predictions and gold labels
        """
        outs = self.model.generate(input_ids=batch['source_ids'].cuda(),
                                   attention_mask=batch['source_mask'].cuda(),
                                   max_length=128
                                   )
        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        targets = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]


        #labels_hat = torch.argmax(outs, dim=1)
        val_f1 = self.compute(dec, targets, self.hparams.task)

        return float(val_f1)

    def _step(self, batch, cl=False):
        lm_labels = batch["target_ids"].clone()
        dec_inputs = batch["target_ids"].clone()
        dec_inputs = self.model._shift_right(dec_inputs)
        # all labels set to -100 are ignored(masked)
        # the loss is only computed for labels in [0,...,vocab_size]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        dec_mask = torch.sign(dec_inputs)
        dec_mask[:, 0] = 1

        loss = self(input_ids=batch["source_ids"],
                    attention_mask=batch["source_mask"],
                    labels=lm_labels,
                    decoder_input_ids=dec_inputs,
                    decoder_attention_mask=dec_mask,
                    cl=cl
                    )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.hparams.cl)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        f1 = self.evl(batch)
        loss = self._step(batch)
        return {"val_loss": loss, "val_f1": torch.tensor(f1)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()

        tensorboard_logs = {"val_loss": avg_loss, "val_f1": avg_f1}
        return {"avg_val_loss": avg_loss, "avg_val_f1": avg_f1, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if not self.trainer.use_tpu:
            # xm.optimizer_step(optimizer)
        # else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        gpus = self.hparams.n_gpu.split(',')
        if len(gpus) > 1:
            gpu_len = len(gpus)
        else:
            gpu_len = 1
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, gpu_len)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, task, sents):
    """
    Compute scores given the predictions and gold labels
    """
    #device = torch.device(f'cuda:{args.n_gpu}')
    #model.model.to(device)
    model.model.cuda()
    
    model.model.eval()
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                                    attention_mask=batch['source_mask'].cuda(),
                                    max_length=128)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(outputs, targets, sents, task)
    results = {'raw_scores': raw_scores, 'fixed_scores': fixed_scores, 'labels': all_labels,
               'preds': all_preds, 'preds_fixed': all_preds_fixed}
    pickle.dump(results, open(f"{args.output_dir}/results-{args.task}-{args.dataset}.pickle", 'wb'))

    return raw_scores, fixed_scores


# initialization
args = init_args()
print("\n", "="*30, f"NEW EXP: {args.task.upper()} on {args.dataset}", "="*30)
print("\n", "="*30, f"cl is {args.cl} k: {args.k} epochs: {args.num_train_epochs} batch_size: {args.train_batch_size}", "="*30)

seed_everything(args.seed)

tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

# show one sample to check the sanity of the code and the expected output
print(f"Here is an example (from dev set) :")
dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='dev', 
                      task=args.task, max_len=args.max_seq_length)
data_sample = dataset[0]  # a random data sample
s = tokenizer.encode

print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))


# training process
if args.do_train:
    print("\n****** Conduct Training ******")
    model = STECL(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="ckt", monitor='val_f1', mode='max', save_top_k=5
    )

    # prepare for trainer
    train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        gradient_clip_val=1.0,
        #amp_level='O1',
        max_epochs=args.num_train_epochs,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
        distributed_backend='ddp'
    )
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    # save the final model
    model.model.save_pretrained(args.output_dir)

    print("Finish training and saving the model!")


if args.do_eval:

    print("\n****** Conduct Evaluating ******")

    # model = T5FineTuner(args)
    dev_results, test_results = {}, {}
    best_f1, best_checkpoint, best_epoch = -999999.0, None, None
    all_checkpoints, all_epochs = [], []

    # retrieve all the saved checkpoints for model selection
    saved_model_dir = args.output_dir
    for f in os.listdir(saved_model_dir):
        file_name = os.path.join(saved_model_dir, f)
        if 'cktepoch' in file_name:
            all_checkpoints.append(file_name)

    # conduct some selection (or not)
    print(f"We will perform validation on the following checkpoints: {all_checkpoints}")

    # load dev and test datasets
    dev_dataset = ABSADataset(tokenizer, data_dir=args.dataset, data_type='dev',
                    task=args.task, max_len=args.max_seq_length)
    dev_loader = DataLoader(dev_dataset, batch_size=32, num_workers=4)

    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, data_type='test', 
                    task=args.task, max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    
    for checkpoint in all_checkpoints:
        epoch = checkpoint.split('=')[-1][:-5] if len(checkpoint) > 1 else ""
        # only perform evaluation at the specific epochs ("15-19")
        # eval_begin, eval_end = args.eval_begin_end.split('-')
        print(epoch)
        if 0 <= float(epoch) < 100:
            all_epochs.append(epoch)

            # reload the model and conduct inference
            print(f"\nLoad the trained model from {checkpoint}...")
            model_ckpt = torch.load(checkpoint)
            model = STECL(model_ckpt['hyper_parameters'])
            model.load_state_dict(model_ckpt['state_dict'])
            sents, _ = read_line_examples_from_file(f'data/{args.task}/{args.dataset}/dev.txt')
            dev_result, _ = evaluate(dev_loader, model, args.task, sents)
            if dev_result['f1'] > best_f1:
                best_f1 = dev_result['f1']
                best_checkpoint = checkpoint
                best_epoch = epoch

            # add the global step to the name of these metrics for recording
            # 'f1' --> 'f1_1000'
            dev_result = dict((k + '_{}'.format(epoch), v) for k, v in dev_result.items())
            dev_results.update(dev_result)
            sents, _ = read_line_examples_from_file(f'data/{args.task}/{args.dataset}/test.txt')
            test_result, _ = evaluate(test_loader, model, args.task, sents)
            test_result = dict((k + '_{}'.format(epoch), v) for k, v in test_result.items())
            test_results.update(test_result)

    # print test results over last few steps
    print(f"\n\nThe best checkpoint is {best_checkpoint}")
    best_step_metric = f"f1_{best_epoch}"
    print(f"F1 scores on test set: {test_results[best_step_metric]:.4f}")

    print("\n* Results *:  Dev  /  Test  \n")
    metric_names = ['f1', 'precision', 'recall']
    for epoch in all_epochs:
        print(f"Epoch-{epoch}:")
        for name in metric_names:
            name_step = f'{name}_{epoch}'
            print(f"{name:<10}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}", sep='  ')
        print()

    results_log_dir = './results_log'
    if not os.path.exists(results_log_dir):
        os.mkdir(results_log_dir)
    log_file_path = f"{results_log_dir}/{args.task}-{args.dataset}.txt"
    write_results_to_log(log_file_path, test_results[best_step_metric], args, dev_results, test_results, all_epochs)

