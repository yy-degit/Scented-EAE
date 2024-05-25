import torch 
from torch import nn
from models.crf import CRF
from models.attention import one_head_attention
from utils.dataloader import pad_and_align


class NER(nn.Module):
    def __init__(self, hidden_size, entity_type2id, bio_type2id, dropout):
        super(NER, self).__init__()
        self.entity_type_num = len(entity_type2id)
        self.bio_type_num = len(bio_type2id)
        self.entity_type2id = entity_type2id
        self.bio_type2id = bio_type2id
        self.bio_id2type = {v:k for k,v in bio_type2id.items()}
        self.bio_type_embeddings = nn.Embedding(self.bio_type_num, hidden_size)
        self.bio_type_index = torch.arange(self.bio_type_num).long()
        self.dropout = nn.Dropout(dropout)
        self.cat_bio_linear = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU()
        )
        self.crf_linear = nn.Linear(hidden_size, self.bio_type_num)
        self.crf = CRF(bio_type2id)
    
    
    def forward(self, token_hidden_states, token_masks, token_nums, entity_seq_labels, device):
        now_max_token_num = max(token_nums)
        # [bio_type_num x hidden_size]
        bio_embeddings = self.get_bio_embeddings(self.bio_type_index.to(device))
        bio_embeddings = self.dropout(bio_embeddings)
        # [batch_size x bio_type_num x hidden_size]
        batch_bio_embeddings = bio_embeddings.unsqueeze(0).expand(token_hidden_states.size(0), -1, -1)
        # [batch_size x bio_type_num]
        bio_masks = token_masks.new_ones((batch_bio_embeddings.size(0), batch_bio_embeddings.size(1)))
        # [batch_size x max_token_num x hidden_size]
        aware_bio_states = one_head_attention(token_hidden_states, batch_bio_embeddings, batch_bio_embeddings, bio_masks, self.dropout)
        aware_bio_states = self.dropout(aware_bio_states)
        token_hidden_states = torch.cat([token_hidden_states, aware_bio_states, token_hidden_states*aware_bio_states], dim=-1)
        token_hidden_states = self.cat_bio_linear(token_hidden_states)
        token_hidden_states = self.dropout(token_hidden_states)
        # [batch_size x max_token_num x bio_type_num]
        entity_seq_scores = self.crf_linear(token_hidden_states)
        ner_scores = self.crf.pad_logits(entity_seq_scores)
        # [batch_size]
        ner_loss = self.crf.loglik(ner_scores[:, :now_max_token_num], entity_seq_labels[:, :now_max_token_num], token_nums)
        return ner_scores, ner_loss 

        
    def decode(self, ner_scores, token_nums, max_entity_num, device):
        _, ner_preds = self.crf.viterbi_decode(ner_scores, token_nums)
        entity_list_preds = tag_paths_to_entities(ner_preds, token_nums, self.bio_id2type, self.entity_type2id)
        entity_nums = token_nums.new([len(one) for one in entity_list_preds])
        entity_list_preds = pad_and_align(entity_list_preds, [0,0,-1], max_num=max_entity_num, device=device)
        return entity_nums, entity_list_preds
    
    
    def get_bio_embeddings(self, index):
        bio_embeddings = self.bio_type_embeddings(index)
        return bio_embeddings
            

def tag_paths_to_entities(paths, token_nums, bio_id2type, entity_type2id):
    entity_lists = []
    for i, path in enumerate(paths):
        entity_list = []
        entity = None
        path = path.cpu().detach().numpy().tolist()[:token_nums[i].item()]
        for j, tag in enumerate(path):
            tag = bio_id2type[tag]
            if tag == 'O':
                prefix = tag = 'O'
            else:
                prefix, tag = tag.split('-', 1)
            if prefix == 'B':
                if entity:
                    entity_list.append(entity)
                entity = [j, j + 1, entity_type2id[tag]]
            elif prefix == 'I':
                if entity is None:
                    entity = [j, j + 1, entity_type2id[tag]]
                elif entity[-1] == entity_type2id[tag]:
                    entity[1] = j + 1
                else:
                    entity_list.append(entity)
                    entity = [j, j + 1, entity_type2id[tag]]
            else:
                if entity:
                    entity_list.append(entity)
                entity = None
        if entity:
            entity_list.append(entity)
        entity_lists.append(entity_list)
    return entity_lists