from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import logging
import numpy as np
from utils.metrics import Metric
import json
import os


logger = logging.getLogger("__main__")


class Framework(object):
    def __init__(self, args, model):
        self.config = args
        self.model = model.to(args.device)
        
        
    def set_learning_params(self, examples):
        train_steps = int(examples * self.config.epochs // self.config.train_batch_size)
                
        no_decay = ['bias', 'LayerNorm.weight']
        prefix = 'model'
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if prefix in n and any(nd in n for nd in no_decay)], 'lr': self.config.bart_learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if prefix in n and not any(nd in n for nd in no_decay)], 'lr': self.config.bart_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if prefix not in n and any(nd in n for nd in no_decay)], 'lr': self.config.learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if prefix not in n and not any(nd in n for nd in no_decay)], 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay}
        ]   
        optimizer = AdamW(optimizer_grouped_parameters)
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps * self.config.warmup_ratio, num_training_steps=train_steps)

        if torch.cuda.device_count() > 1:
            logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)

        return train_steps, scheduler, optimizer
    
    
    def train(self, train_loader, valid_loader):
        train_steps, scheduler, optimizer = self.set_learning_params(len(train_loader.dataset))
        
        logger.info("***** Running Training *****")
        logger.info(f"  Num Examples = {len(train_loader.dataset)}")
        logger.info(f"  Num Epochs = {self.config.epochs}")
        logger.info(f"  Batch Size = {self.config.train_batch_size}")
        logger.info(f"  Total Steps = {train_steps}")
    
        best_f1 = 0.0
        best_epoch = 1
        train_result = dict()
        train_result["log_history"] = []
        for epoch in range(1, self.config.epochs+1):
            train_loss = 0.0
            log_loss = 0.0
            train_examples = 0
            log_examples = 0
            
            # train
            self.model.train()
            self.model.zero_grad()
            for batch_id, data in enumerate(train_loader):
                examples = len(data["encoder_input_ids"]) 
                _, _, loss = self.model(
                    mission = "train",
                    encoder_input_ids = data["encoder_input_ids"].to(self.config.device),
                    encoder_attention_mask = data["encoder_attention_mask"].to(self.config.device),
                    context_mask = data["context_mask"].to(self.config.device),
                    token_nums = data["token_nums"].to(self.config.device),
                    old_token2new_index = data["old_token2new_index"].to(self.config.device),
                    entity_seq_labels = data["entity_seq_labels"].to(self.config.device),
                    entity_lists = data["entity_lists"].to(self.config.device),
                    role_spans = data["role_spans"].to(self.config.device),
                    role_entity_spans = data["role_entity_spans"].to(self.config.device),
                    role_start_labels = data["role_start_labels"].to(self.config.device),
                    role_end_labels = data["role_end_labels"].to(self.config.device)
                )
                
                # loss是一批数据损失和
                if torch.cuda.device_count() > 1:
                    # 分布式返回多个GPU的拼接结果
                    loss = torch.sum(loss)
                train_loss += loss.item()
                log_loss += loss.item()
                train_examples += examples
                log_examples += examples
                if (batch_id+1) % self.config.logging_steps == 0:
                    logger.info(f'Epoch: {epoch}, Step: {batch_id+1}, Training Loss:  {log_loss/log_examples}')
                    log_loss = 0.0
                    log_examples = 0
                
                loss = loss / examples / self.config.gradient_accumulation_steps
                loss.backward()
                if (batch_id+1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    
            logger.info('Epoch {}: \t Avgerage Training Loss = {:.6f}'.format(epoch, train_loss/train_examples))
            
            # eval
            avg_eval_loss, f1_score, metrics = self.evaluate(valid_loader)
            epoch_log = {"epoch": epoch, "train_loss": round(train_loss/train_examples,6), "eval_loss": round(avg_eval_loss,6), "f1_socre": round(f1_score,4)}
            epoch_log.update(metrics)
            train_result["log_history"].append(epoch_log)
            
            # save model
            if f1_score > best_f1:
                logger.info("F1 score increases from {:.4f} to {:.4f}. Saving model.".format(best_f1, f1_score))
                best_f1 = f1_score
                best_epoch = epoch
                self.save_model(self.config.output_model_path)
        logger.info("The Best F1 Is: {:.4f}, When Epoch Is: {}".format(best_f1, best_epoch))
        train_result["best_epoch"] = best_epoch
        train_result["best_f1"] = round(best_f1, 4)
        save_result(train_result, os.path.join(self.config.output_dir, "train_result.json"))
        logger.info("Training End.")
       
          
    def evaluate(self, valid_loader, output_result=False):
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model
            
        logger.info("***** Running Evaluation *****")
        logger.info(f"  Num Examples = {len(valid_loader.dataset)}")
        logger.info(f"  Batch Size = {self.config.eval_batch_size}")
        logger.info("  Total Steps = {}".format(int(len(valid_loader.dataset) // self.config.eval_batch_size)))
        
        model.eval()
        
        eval_loss = 0.0
        eval_examples = 0
        ner_targets, ner_outputs, eae_targets, eae_outputs = [], [], [], []
        new_piece2old_index, role_ids = [], []
        with torch.no_grad():
            for batch_id, data in enumerate(valid_loader):
                examples = len(data["encoder_input_ids"]) 
                entity_list_preds, eae_scores, loss = model(
                    mission = "eval",
                    encoder_input_ids = data["encoder_input_ids"].to(self.config.device),
                    encoder_attention_mask = data["encoder_attention_mask"].to(self.config.device),
                    context_mask = data["context_mask"].to(self.config.device),
                    token_nums = data["token_nums"].to(self.config.device),
                    old_token2new_index = data["old_token2new_index"].to(self.config.device),
                    entity_seq_labels = data["entity_seq_labels"].to(self.config.device),
                    entity_lists = data["entity_lists"].to(self.config.device),
                    role_spans = data["role_spans"].to(self.config.device),
                    role_entity_spans = data["role_entity_spans"].to(self.config.device),
                    role_start_labels = data["role_start_labels"].to(self.config.device),
                    role_end_labels = data["role_end_labels"].to(self.config.device)
                )
                
                logger.info(f"Batch {batch_id+1} Loss: {loss.item()/examples}")
                eval_loss += loss.item()
                eval_examples += examples
                ner_targets.extend(data["entity_lists"].cpu().detach().numpy().tolist())
                ner_outputs.extend(entity_list_preds.cpu().detach().numpy().tolist())
                eae_targets.extend(data["argument_lists"])
                eae_outputs.extend(eae_scores.cpu().detach().numpy())
                new_piece2old_index.extend(data["new_piece2old_index"])
                role_ids.extend(data["role_ids"])
        
        avg_eval_loss = eval_loss/eval_examples
        logger.info('Evaluation Loss = {:.6f}'.format(avg_eval_loss))
        
        metrics = dict()
        final_eae_outputs = None
        ner_outputs = remove_pad(ner_outputs, [0,0,-1])
        if self.config.task == "NER":
            f1_score, metrics = self.evaluate_ner_task(ner_targets, ner_outputs, metrics)
        elif self.config.task == "EAE":
            f1_score, metrics, best_thresholds, final_eae_outputs = self.evaluate_eae_task(eae_targets, eae_outputs, metrics, new_piece2old_index, role_ids)
        else:
            _, metrics = self.evaluate_ner_task(ner_targets, ner_outputs, metrics)
            f1_score, metrics, best_thresholds, final_eae_outputs = self.evaluate_eae_task(eae_targets, eae_outputs, metrics, new_piece2old_index, role_ids)
        best_metric = metrics["threshold_"+str(best_thresholds[0])+"_"+str(best_thresholds[1])]
        logger.info('Best Result: Threshold = ({:.2f}, {:.2f}), AI-P = {:.4f}, AI-R = {:.4f}, AI-F1 = {:.4f}, AC-P = {:.4f}, AC-R = {:.4f}, AC-F1 = {:.4f}'\
                    .format(best_thresholds[0], best_thresholds[1], best_metric["ai-p"], best_metric["ai-r"], best_metric["ai-f1"], best_metric["ac-p"], best_metric["ac-r"], best_metric["ac-f1"]))
            
        if output_result:
            save_result(metrics, os.path.join(self.config.output_dir,"eval_result.json"))
            self.output_predict_result(valid_loader.dataset[:eval_examples], ner_outputs, final_eae_outputs, \
            os.path.join(self.config.output_dir,"eval_pred_result.json"))
        logger.info("Evaluation End.")
        
        return avg_eval_loss, f1_score, metrics
    
    
    def predict(self, test_loader):
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model
            
        logger.info("***** Running Prediction *****")
        logger.info(f"  Num Examples = {len(test_loader.dataset)}")
        logger.info(f"  Batch Size = {self.config.eval_batch_size}")
        logger.info("  Total Steps = {}".format(int(len(test_loader.dataset) // self.config.eval_batch_size)))

        model.eval()
        
        test_examples = 0
        ner_outputs, eae_outputs = [], []
        new_piece2old_index, role_ids = [], []
        with torch.no_grad():
            for data in test_loader:
                examples = len(data["encoder_input_ids"]) 
                entity_list_preds, eae_scores, _ = model(
                    mission = "test",
                    encoder_input_ids = data["encoder_input_ids"].to(self.config.device),
                    encoder_attention_mask = data["encoder_attention_mask"].to(self.config.device),
                    context_mask = data["context_mask"].to(self.config.device),
                    token_nums = data["token_nums"].to(self.config.device),
                    old_token2new_index = data["old_token2new_index"].to(self.config.device),
                    entity_seq_labels = data["entity_seq_labels"].to(self.config.device),
                    entity_lists = data["entity_lists"].to(self.config.device),
                    role_spans = data["role_spans"].to(self.config.device),
                    role_entity_spans = data["role_entity_spans"].to(self.config.device),
                    role_start_labels = data["role_start_labels"].to(self.config.device),
                    role_end_labels = data["role_end_labels"].to(self.config.device)
                )
                
                test_examples += examples
                ner_outputs.extend(entity_list_preds.cpu().detach().numpy().tolist())
                eae_outputs.extend(eae_scores.cpu().detach().numpy())
                new_piece2old_index.extend(data["new_piece2old_index"])
                role_ids.extend(data["role_ids"])
                
        ner_outputs = remove_pad(ner_outputs, [0,0,-1])
        if self.config.task != "NER":
            eae_outputs = self.decode(eae_outputs, new_piece2old_index, role_ids, self.config.thresholds[0], self.config.thresholds[0])

        self.output_predict_result(test_loader.dataset[:test_examples], ner_outputs, eae_outputs, \
        os.path.join(self.config.output_dir,"test_pred_result.json"))
        logger.info("Prediction End.")
    
    
    def evaluate_ner_task(self, ner_targets, ner_outputs, metrics):
        ner_targets = remove_pad(ner_targets, [0,0,-1])
        ei_metric, ec_metric = self.calculate_metric(ner_targets, ner_outputs)
        logger.info('EI-P = {:.4f}, EI-R = {:.4f}, EI-F1 = {:.4f}, EC-P = {:.4f}, EC-R = {:.4f}, EC-F1 = {:.4f}'\
        .format(ei_metric.p, ei_metric.r, ei_metric.f1, ec_metric.p, ec_metric.r, ec_metric.f1))
        f1_score = ec_metric.f1
        metrics.update(ei_metric.to_dict("ei"))
        metrics.update(ec_metric.to_dict("ec"))
        return f1_score, metrics
        
    
    def evaluate_eae_task(self, eae_targets, eae_outputs, metrics, new_piece2old_index, role_ids):
        f1_score = 0
        best_thresholds = [round(self.config.thresholds[0],2), round(self.config.thresholds[0],2)]
        final_eae_outputs = None
        for threshold1 in self.config.thresholds:
            for threshold2 in self.config.thresholds:
                threshold_str = "threshold_" + str(round(threshold1,2)) + "_" + str(round(threshold2,2))
                metrics[threshold_str] = dict()
                new_eae_outputs = self.decode(eae_outputs, new_piece2old_index, role_ids, threshold1, threshold2)
                ai_metric, ac_metric = self.calculate_metric(eae_targets, new_eae_outputs)
                logger.info('Threshold = ({:.2f}, {:.2f}), AI-P = {:.4f}, AI-R = {:.4f}, AI-F1 = {:.4f}, AC-P = {:.4f}, AC-R = {:.4f}, AC-F1 = {:.4f}'\
                .format(threshold1, threshold2, ai_metric.p, ai_metric.r, ai_metric.f1, ac_metric.p, ac_metric.r, ac_metric.f1))
                if ac_metric.f1 > f1_score:
                    f1_score = ac_metric.f1
                    best_thresholds = [round(threshold1,2), round(threshold2,2)] 
                    final_eae_outputs = new_eae_outputs
                metrics[threshold_str].update(ai_metric.to_dict("ai"))
                metrics[threshold_str].update(ac_metric.to_dict("ac"))
        return f1_score, metrics, best_thresholds, final_eae_outputs
    
    
    def decode(self, eae_outputs, new2old_index, role_ids, threshold1, threshold2):
        argument_lists = []
        start_scores = [np.squeeze(one[:,:,:1], axis=-1) for one in eae_outputs]
        end_scores = [np.squeeze(one[:,:,-1:], axis=-1) for one in eae_outputs] 
        for start_score, end_score, new2old, roles in zip(start_scores, end_scores, new2old_index, role_ids):
            argument_list = []
            for index, role_id in enumerate(roles):
                starts = np.where(start_score[index] > threshold1)[0]
                ends = np.where(end_score[index] > threshold2)[0]
                for i in starts:
                    if i == 0:
                        continue
                    if i >= len(new2old) or new2old[i] == -1:
                        break
                    for j in ends:
                        if j == 0:
                            continue     
                        if j >= len(new2old) or new2old[j] == -1:
                            break
                        if j >= i:    
                            argument_list.append([new2old[i], new2old[j] + 1, role_id])
                            break
            argument_list.sort(key=lambda x:(x[0],x[1]))
            argument_lists.append(argument_list)
        return argument_lists
    
    
    def calculate_metric(self, targets=None, outputs=None):
        i_metric, c_metric = None, None
        i_targets = [list(set([(item[0], item[1]) for item in one])) for one in targets]
        i_outputs = [list(set([(item[0], item[1]) for item in one])) for one in outputs]
        c_targets = [list(set([(item[0], item[1], item[2]) for item in one])) for one in targets]
        c_outputs = [list(set([(item[0], item[1], item[2]) for item in one])) for one in outputs]
        i_metric, c_metric = Metric(), Metric()
        i_metric.compute(i_targets, i_outputs)
        c_metric.compute(c_targets, c_outputs)
        return i_metric, c_metric
    
    
    def save_model(self, model_path): 
        if not os.path.exists(model_path):
            os.mkdir(model_path)   
        if hasattr(self.model, "module"):
            self.model.module.save_pretrained(model_path)
        else:
            self.model.save_pretrained(model_path)    
                        
    
    def output_predict_result(self, dataset, ner_outputs, eae_outputs, output_path):
        result = [
                    {
                        "doc_id": dataset["doc_id"][i],
                        "sentence_id": dataset["sentence_id"][i],
                        "text": dataset["text"][i],
                        "event_id": dataset["event_id"][i],
                        "event_type": self.config.schema["event_id2type"][dataset["event_type_id"][i]]
                    }
                    for i in range(len(dataset["doc_id"]))
        ]
        if self.config.task == "ALL":
            for i, (entity_data, argument_data) in enumerate(zip(dataset["entity_list"], dataset["argument_list"])):
                result[i]["ner_gold"], result[i]["ner_pred"] = \
                organize_gold_pred(entity_data, ner_outputs[i], self.config.schema["entity_id2type"], dataset["text"][i])
                result[i]["eae_gold"], result[i]["eae_pred"] = \
                organize_gold_pred(argument_data, eae_outputs[i], self.config.schema["role_id2type"], dataset["text"][i])
        elif self.config.task == "NER":
            for i, data in enumerate(dataset["entity_list"]):
                result[i]["ner_gold"], result[i]["ner_pred"] = \
                organize_gold_pred(data, ner_outputs[i], self.config.schema["entity_id2type"], dataset["text"][i])
        else:
            for i, data in enumerate(dataset["argument_list"]):
                result[i]["eae_gold"], result[i]["eae_pred"] = \
                organize_gold_pred(data, eae_outputs[i], self.config.schema["role_id2type"], dataset["text"][i])
        save_result(result, output_path)
        

def remove_pad(pad_lists, pad_value):
    remove_pad_lists = list()
    for one in pad_lists:
        if pad_value in one:
            remove_pad_lists.append(one[:one.index(pad_value)])
        else:
            remove_pad_lists.append(one)
    return remove_pad_lists
    

def organize_gold_pred(gold_data, pred_data, id2type, text):
    gold, pred = [], []
    for item in gold_data:
        new_item = item[:2] + [id2type[item[2]]]
        gold.append(new_item + [text[item[0]:item[1]]])
    for item in pred_data:
        new_item = item[:2] + [id2type[item[2]]]
        pred.append(new_item + [text[item[0]:item[1]]])
    return gold, pred


def save_result(result, output_path):
    with open(output_path,"w",encoding="utf-8") as fs:
        json.dump(result,fs,indent=4,ensure_ascii=False)
        
