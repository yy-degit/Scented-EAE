import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel
from models.ner import NER 
from utils.metrics import Metric
from utils.dataloader import pad_and_align
import math


class ScentedEAE(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)
        self.ner = NER(config.d_model, config.entity_type2id, config.bio_type2id, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.cat_type_linear = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        self.start_cls = nn.Parameter(torch.rand(config.d_model, ))
        self.end_cls = nn.Parameter(torch.rand(config.d_model, ))
        self.constrain_start_cls = nn.Parameter(torch.rand(config.d_model, ))
        self.constrain_end_cls = nn.Parameter(torch.rand(config.d_model, ))
        self.eae_loss_function = nn.BCELoss(reduction="none")
        
        
    def forward(
        self,
        mission,
        encoder_input_ids,
        encoder_attention_mask,
        context_mask,
        token_nums,
        old_token2new_index,
        entity_seq_labels,
        entity_lists,
        role_spans,
        role_entity_spans,
        role_start_labels,
        role_end_labels
    ):
        assert mission in ["train", "eval", "test"], "Invalid Mission Name!" 
    
        
        model_outputs = self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True
        )
        hidden_states = model_outputs.last_hidden_state
    
        
        entity_list_preds = entity_lists                      
        if self.config.task != "EAE":
            # [batch_size x max_token_num x hidden_size], [batch_size x max_token_num]
            token_hidden_states, token_masks = regather(encoder_input_ids, hidden_states, old_token2new_index)
            token_hidden_states = self.dropout(token_hidden_states)
            ner_scores, ner_loss = self.ner(token_hidden_states, token_masks, token_nums, entity_seq_labels, self.model.device)
            if self.config.task == "NER":
                if mission == "train":
                    return entity_list_preds, ner_scores, -ner_loss.sum()
                else:
                    entity_nums, entity_list_preds = self.ner.decode(ner_scores, token_nums, self.config.max_entity_num, self.model.device)
                    return entity_list_preds, ner_scores, -ner_loss.sum()
        if self.config.task == "ALL" and mission != "train":
            entity_nums, entity_list_preds = self.ner.decode(ner_scores, token_nums, self.config.max_entity_num, self.model.device)
        
        
        bio_spans, bio_ids = get_bio_spans(entity_list_preds, old_token2new_index, self.config.entity_bio_map)
        if bio_ids.size(1) != 0:
            # [batch_size x max_bio_num]
            bio_masks = encoder_attention_mask.new_ones(bio_ids.size())
            bio_masks[bio_ids == -1] = 0
            bio_ids[bio_ids == -1] = 0
            # [batch_size x max_bio_num x hidden_size]
            bio_embeddings = self.ner.get_bio_embeddings(bio_ids)
            bio_embeddings = self.dropout(bio_embeddings)
            bio_embeddings = bio_embeddings * bio_masks.unsqueeze(-1).expand_as(bio_embeddings)
            # [batch_size x max_seq_length x hidden_size], [batch_size x max_seq_length]
            bio_type_addend, bio_type_masks = type_addend(hidden_states, bio_embeddings, bio_spans)   
        else:
            bio_type_addend = hidden_states.new_zeros(hidden_states.size())
            bio_type_masks = old_token2new_index.new_zeros(encoder_attention_mask.size())
        o_masks = 1 - bio_type_masks
        # [batch_size x 1]
        o_ids = bio_ids.new_zeros((o_masks.size(0), 1))
        # [batch_size x 1 x hidden_size]
        o_embeddings = self.ner.get_bio_embeddings(o_ids)
        o_embeddings = self.dropout(o_embeddings)
        # [batch_size x max_seq_length x hidden_size]
        o_type_addend = torch.matmul(o_masks.unsqueeze(-1).float(), o_embeddings.float())
        final_type_addend = bio_type_addend + o_type_addend
        
        
        # [batch_size x max_role_num x hidden_size], [batch_size x max_role_num]
        role_hidden_states, role_masks = regather(encoder_input_ids, hidden_states, role_spans)
        role_entity_states, role_entity_masks = regather(encoder_input_ids, hidden_states, role_entity_spans)
        hidden_states = self.cat_type_linear(torch.cat([hidden_states, final_type_addend], dim=-1))
        hidden_states = self.dropout(hidden_states)
        final_states = hidden_states
        
        
        role_start_cls = role_hidden_states * self.start_cls.unsqueeze(0).unsqueeze(0).expand_as(role_hidden_states)
        role_end_cls = role_hidden_states * self.end_cls.unsqueeze(0).unsqueeze(0).expand_as(role_hidden_states)
        # 使用完整bart上下文编码防止梯度爆炸的关键
        scale = math.sqrt(self.config.d_model)
        start_scores = torch.sigmoid(torch.matmul(role_start_cls, final_states.transpose(1,2)) / scale)
        end_scores = torch.sigmoid(torch.matmul(role_end_cls, final_states.transpose(1,2)) / scale)
        constrain_start_cls = role_entity_states * self.constrain_start_cls.unsqueeze(0).unsqueeze(0).expand_as(role_entity_states)
        constrain_end_cls = role_entity_states * self.constrain_end_cls.unsqueeze(0).unsqueeze(0).expand_as(role_entity_states)
        constrain_start_scores = torch.sigmoid(torch.matmul(constrain_start_cls, final_states.transpose(1,2)) / scale)
        constrain_end_scores = torch.sigmoid(torch.matmul(constrain_end_cls, final_states.transpose(1,2)) / scale)
        start_scores = start_scores * constrain_start_scores
        end_scores = end_scores * constrain_end_scores
        eae_mask = torch.matmul(role_masks.unsqueeze(-1), context_mask.unsqueeze(-1).transpose(1,2))
        eae_start_loss = self.eae_loss_function(start_scores, role_start_labels)
        eae_end_loss = self.eae_loss_function(end_scores, role_end_labels)
        eae_start_loss = torch.sum(eae_start_loss * eae_mask)
        eae_end_loss = torch.sum(eae_end_loss * eae_mask)
        eae_loss = eae_start_loss + eae_end_loss
        # [batch_size x max_role_num x max_seq_length x 2]
        eae_scores = torch.cat([start_scores.unsqueeze(-1), end_scores.unsqueeze(-1)], dim=-1)
        
        
        if self.config.task == "ALL":
            return entity_list_preds, eae_scores, -ner_loss.sum()+eae_loss
        else:
            return entity_list_preds, eae_scores, eae_loss
    
    
def regather(input_ids, hidden_states, old2new_index, mode="average"):
    batch_size = input_ids.size(0)
    dmodel = hidden_states.size(-1)
    # [batch_size x [max_token_num x max_token_len]]
    max_token_num, max_token_len, idxs, masks = token_to_piece_idxs(old2new_index, mode)
    # [batch_size x [max_token_num x max_token_len x [hidden_size]]]
    idxs = input_ids.new(idxs).unsqueeze(-1).expand(batch_size, -1, dmodel)
    # [batch_size x [max_token_num x max_token_len x [1]]]
    masks = hidden_states.new(masks).unsqueeze(-1)
    # [batch_size x [max_token_num x max_token_len x [hidden_size]]]
    hidden_states= torch.gather(hidden_states, 1, idxs) * masks
    # [batch_size x [max_token_num x [max_token_len x [hidden_size]]]]
    hidden_states = hidden_states.view(batch_size, max_token_num, max_token_len, dmodel)
    # [batch_size x [max_token_num x [hidden_size]]]
    hidden_states = hidden_states.sum(2) if mode != "max" else hidden_states.max(2).values
    # [batch_size x [max_token_num]]
    masks = masks.squeeze(-1).view(batch_size, max_token_num, max_token_len).sum(2)
    masks = masks.masked_fill(masks!=0.0, 1.0)
    return hidden_states, masks


def type_addend(hidden_states, type_embeddings, old2new_index):
    # type_embeddings: [batch_size x type_num x hidden_size]
    assert type_embeddings.size(1) == old2new_index.size(1), "Size(type_embeddings) ≠ Size(old2new_index)!"
    batch_size = hidden_states.size(0)
    max_seq_length = hidden_states.size(1)
    # [batch_size x [max_token_num x max_token_len]]
    max_token_num, max_token_len, idxs, _ = token_to_piece_idxs(old2new_index)
    idxs = old2new_index.new(idxs).view(batch_size, max_token_num, max_token_len)
    indexes = old2new_index.new_zeros((batch_size, max_token_num, max_seq_length))
    for i, tokens in enumerate(idxs):
        for j, token_indexes in enumerate(tokens):
            indexes[i][j][token_indexes] = 1
            indexes[i][j][0] = 0
    # [batch_size x max_seq_length]
    masks = indexes.sum(1)
    masks = masks.masked_fill(masks!=0, 1)
    # [batch_size x [max_token_num x [max_seq_length x [hidden_size]]]]
    addend = torch.matmul(indexes.unsqueeze(-1).float(), type_embeddings.unsqueeze(-2).float())
    # [batch_size x [max_seq_length x [hidden_size]]]
    addend = addend.sum(1)
    
    return addend, masks


def token_to_piece_idxs(old2new_index, mode="average"):
    max_token_num = old2new_index.size(1)
    if mode == "average":
        max_token_len = max([max([(new_index[1]-new_index[0]).item() for new_index in one]) for one in old2new_index])
        idxs, masks = [], []
        for one in old2new_index:
            seq_idxs, seq_masks = [], []
            for new_index in one:
                token_len = (new_index[1] - new_index[0]).item()
                seq_idxs.extend([i for i in range(new_index[0], new_index[1])] + [0] * (max_token_len-token_len))
                seq_masks.extend([Metric.safe_div(1.0, token_len)] * token_len + [0.0] * (max_token_len-token_len))
            idxs.append(seq_idxs)
            masks.append(seq_masks)
    elif mode == "first":
        max_token_len = 1
        idxs, masks = [], []
        for one in old2new_index:
            seq_idxs, seq_masks = [], []
            for new_index in one:
                token_len = (new_index[1] - new_index[0]).item()
                if token_len > 0:
                    seq_idxs.append(new_index[0])
                    seq_masks.append(1.0)
                else:
                    seq_idxs.append(0)
                    seq_masks.append(0.0)
            idxs.append(seq_idxs)
            masks.append(seq_masks)
    elif mode == "bound":
        max_token_len = 2
        idxs, masks = [], []
        for one in old2new_index:
            seq_idxs, seq_masks = [], []
            for new_index in one:
                token_len = (new_index[1] - new_index[0]).item()
                if token_len > 0:
                    seq_idxs.extend([new_index[0], new_index[1]-1])
                    seq_masks.extend([0.5, 0.5])
                else:
                    seq_idxs.extend([0, 0])
                    seq_masks.extend([0.0, 0.0])
            idxs.append(seq_idxs)
            masks.append(seq_masks)
    elif mode == "max":
        max_token_len = max([max([(new_index[1]-new_index[0]).item() for new_index in one]) for one in old2new_index])
        idxs, masks = [], []
        for one in old2new_index:
            seq_idxs, seq_masks = [], []
            for new_index in one:
                token_len = (new_index[1] - new_index[0]).item()
                seq_idxs.extend([i for i in range(new_index[0], new_index[1])] + [0] * (max_token_len-token_len))
                seq_masks.extend([1.0] * token_len + [0.0] * (max_token_len-token_len))
            idxs.append(seq_idxs)
            masks.append(seq_masks)
    else:
        Exception("Invalid entity_mode!")
    return max_token_num, max_token_len, idxs, masks


def get_bio_spans(entity_lists, old2new_index, entity_bio_map):
    bio_spans = list()
    bio_ids = list()
    for i, entity_list in enumerate(entity_lists):
        one_bio_spans = list()
        one_bio_ids = list()
        for entity in entity_list:
            if entity[-1] == -1:
                break
            one_bio_spans.append([old2new_index[i][entity[0]][0].item(), old2new_index[i][entity[0]][1].item()])
            one_bio_ids.append(entity_bio_map[entity[-1].item()][0])
            if entity[0]+1 < entity[1]:
                one_bio_spans.append([old2new_index[i][entity[0]+1][0].item(), old2new_index[i][entity[1]-1][1].item()])
                one_bio_ids.append(entity_bio_map[entity[-1].item()][1])
        bio_spans.append(one_bio_spans)
        bio_ids.append(one_bio_ids)
    bio_spans = pad_and_align(bio_spans, [0,0], device=entity_lists.device)
    bio_ids = pad_and_align(bio_ids, -1, device=entity_list.device)
    return bio_spans, bio_ids