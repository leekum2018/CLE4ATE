import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from metric import calculate_f1
import pdb
from model import Model
from transformers import AdamW, get_linear_schedule_with_warmup

class XHTrainer(object):
    def __init__(self, args, config):
        self.args = args
        self.model = Model(args, config)   
        self.config = config
        self.model.network.cuda()
        summ = 1850 if 'Res' in args.data_dir else 2895
        t_total = int(args.num_epoch * summ // (args.batch_size*args.accu_step))


        param_optimizer = [(k, v) for k, v in self.model.network.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        

        self.optimizer = AdamW(optimizer_grouped_parameters,
                            lr=self.config["training"]["learning_rate"])  
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config["training"]["warmup_proportion"], num_training_steps=t_total) 
        self.global_step = 0
    # load model_state and args
    def load(self, filename):
        try:
            self.model.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
       
    # save model_state and args
    def save(self, filename):
        try:
            self.model.save(filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def update(self, batch):
        self.model.train()
        self.global_step += 1
        
        outputs = self.model.network(batch)
        loss, score = outputs[:2]
        loss.backward()
        if self.global_step%self.args.accu_step==0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        return loss.item()


    def evaluate(self, batch):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.network(batch)
            loss, score = outputs[:2]
       
        return loss.item()

    def predict(self, batchs):
        self.load(self.args.save_dir+'/best_model.pt')        
        self.model.eval()
        pred = []
        with torch.no_grad():
            for i, batch in enumerate(batchs):  
                outputs = self.model.network(batch)
                _, predi = outputs[:2] 
                               
                predi_ = self.bpe2word(predi.cpu().numpy(), batch[4].cpu().tolist(), batch[2].cpu().tolist())
                predi_ = predi_.argmax(axis=2).tolist()
                pred.extend(predi_)

        assert len(pred) in [800, 676]
        f1_score = calculate_f1(pred, self.args.data_dir)
        return f1_score

    def bpe2word(self, batch_pred_y, transit, lengths):
        new_batch_pred_y = np.zeros([1, 83, 3])

        length = lengths[0]
        target = -1
        for j in range(1, length+1): #no cls and sep
            if target == transit[j]:
                continue
            if target > 0 and transit[j]==0:
                raise Exception
            target = transit[j]  # exclude cls

            new_batch_pred_y[0][target] = batch_pred_y[j]

        return new_batch_pred_y