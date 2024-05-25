from torch.utils.data import Dataset, DataLoader
import torch
import json
import re
import copy


class Data(Dataset):
    def __init__(self, mission, dataset, tokenizer, args):
        self.mission = mission
        self.tokenizer = tokenizer
        self.schema = args.schema
        self.template = args.template
        self.encoder_max_seq_length= args.encoder_max_seq_length
        self.max_entity_num = args.max_entity_num
        self.max_encoder_length = 0
        
        self.doc_ids = list()
        self.sentence_ids = list()
        self.texts = list()
        self.event_ids = list()
    
        self.encoder_input_ids = list() 
        self.encoder_attention_mask = list()
        self.context_mask = list()
        self.token_nums = list()
        self.old_token2new_index = list()
        self.new_piece2old_index = list()
        self.entity_seq_labels = list()
        self.entity_lists = list()
        self.event_type_ids = list()
        self.role_ids = list()
        self.role_spans = list()
        self.role_entity_spans = list()
        self.role_start_labels = list()
        self.role_end_labels = list()
        self.argument_lists = list()
        
        for data in dataset:
            doc_id = data["doc_id"]
            sentence_id = data["sentence_id"]
            text = data["text"]
            entities = data["entities"]
            if len(entities) == 0:
                continue
            event_id = 0
            for event in data["events"]:
                self.doc_ids.append(doc_id)
                self.sentence_ids.append(sentence_id)
                self.texts.append(text)
                self.event_ids.append(event_id)
                
                encoder_input_ids, encoder_attention_mask, context_mask, token_num, old_token2new_index, new_piece2old_index, \
                entity_seq_labels, entity_list, event_type_id, role_ids, role_spans, role_entity_spans, role_start_labels, role_end_labels, argument_list = \
                self.prepare_data(text, event, entities) 
        
                self.encoder_input_ids.append(encoder_input_ids)
                self.encoder_attention_mask.append(encoder_attention_mask)
                self.context_mask.append(context_mask)
                self.token_nums.append(token_num)
                self.old_token2new_index.append(old_token2new_index)
                self.new_piece2old_index.append(new_piece2old_index)
                self.entity_seq_labels.append(entity_seq_labels)
                self.entity_lists.append(entity_list)
                self.event_type_ids.append(event_type_id)
                self.role_ids.append(role_ids)
                self.role_spans.append(role_spans)
                self.role_entity_spans.append(role_entity_spans)
                self.role_start_labels.append(role_start_labels)
                self.role_end_labels.append(role_end_labels)
                self.argument_lists.append(argument_list)
            
                event_id += 1
                
        print("Max Piece Length (No Padding) :", str(self.max_encoder_length))
    

    def __len__(self):
        return len(self.doc_ids)


    def __getitem__(self, index):
        if self.mission != "train":
            return  {
                        "doc_id": self.doc_ids[index],
                        "sentence_id": self.sentence_ids[index],
                        "text": self.texts[index],
                        "event_id": self.event_ids[index],
                        "encoder_input_ids": self.encoder_input_ids[index],
                        "encoder_attention_mask": self.encoder_attention_mask[index],
                        "context_mask": self.context_mask[index],
                        "token_num": self.token_nums[index],
                        "old_token2new_index": self.old_token2new_index[index],
                        "new_piece2old_index": self.new_piece2old_index[index],
                        "entity_seq_labels": self.entity_seq_labels[index],
                        "entity_list": self.entity_lists[index],
                        "event_type_id": self.event_type_ids[index],
                        "role_ids": self.role_ids[index],
                        "role_spans": self.role_spans[index],
                        "role_entity_spans": self.role_entity_spans[index],
                        "role_start_labels": self.role_start_labels[index],
                        "role_end_labels": self.role_end_labels[index],
                        "argument_list": self.argument_lists[index]
                    }
        else:
            return  {
                        "encoder_input_ids": self.encoder_input_ids[index],
                        "encoder_attention_mask": self.encoder_attention_mask[index],
                        "context_mask": self.context_mask[index],
                        "token_num": self.token_nums[index],
                        "old_token2new_index": self.old_token2new_index[index],
                        "entity_seq_labels": self.entity_seq_labels[index],
                        "entity_list": self.entity_lists[index],
                        "role_spans": self.role_spans[index],
                        "role_entity_spans": self.role_entity_spans[index],
                        "role_start_labels": self.role_start_labels[index],
                        "role_end_labels": self.role_end_labels[index]
                    }


    def prepare_data(self, text, event, entities):
        event_type2id = self.schema["event_type2id"]
        role_type2id = self.schema["role_type2id"]
        entity_type2id = self.schema["entity_type2id"]
        bio_type2id = self.schema["bio_type2id"]
        
        token_num = len(text)
        event_type_id = event_type2id[event["type"]]
        event_type = event["type"].replace(".", "_")
        role_entity_template = self.template[event_type_id]["role_entity_template"]
        role_list = self.template[event_type_id]["new_role_list"]
        role_entity_list = self.template[event_type_id]["role_entity_list"]
        
        encoder_text = " ".join(text) + " <s> </s> In the "
        encoder_text += event_type + " event triggered by "
        encoder_text += " ".join(event["trigger"]["word"]) + " , "
        role2char_index = [[role[0]+len(encoder_text), role[1]+len(encoder_text)-1] for role in role_list]
        role_entity2char_index = [[entity[0]+len(encoder_text), entity[1]+len(encoder_text)-1] for entity in role_entity_list]
        encoder_text += role_entity_template
        encoder_inputs, encoder_input_ids, encoder_attention_mask, self.max_encoder_length \
        = self.encode_to_id(encoder_text, self.encoder_max_seq_length, self.max_encoder_length)
    
        old_token2char_index = list() 
        index = 0
        for token in text:
            old_token2char_index.append([index, index+len(token)-1]) # 注意end包含在内
            index += len(token)+1
        old_token2new_index = old_to_new(encoder_inputs, old_token2char_index)
        new_piece2old_index = [-1] * old_token2new_index[-1][1]
        for i, span in enumerate(old_token2new_index):
            for j in range(span[0], span[1]):
                new_piece2old_index[j] = i
        text2new_index = [old_token2new_index[0][0], old_token2new_index[-1][1]]
        context_mask = [0] * self.encoder_max_seq_length
        for i in range(text2new_index[0]-1, text2new_index[1]+1):
            context_mask[i] = 1
        
        entity_seq_labels = ["O"] * token_num
        for entity in entities:
            entity_span = entity["span"]
            if any([entity_seq_labels[i] != "O" for i in range(entity_span[0], entity_span[1])]):
                continue
            entity_seq_labels[entity_span[0]] = "B-" + entity["type"]
            for i in range(entity_span[0]+1, entity_span[1]):
                entity_seq_labels[i] = "I-" + entity["type"]
        entity_seq_labels = [bio_type2id[i] for i in entity_seq_labels]
        entity_list = [[entity["span"][0], entity["span"][1], entity_type2id[entity["type"]]] for entity in entities]
           
        role_ids = [role[-1] for role in role_list]
        role_spans= old_to_new(encoder_inputs, role2char_index)
        role_entity_spans = old_to_new(encoder_inputs, role_entity2char_index)
        
        role_start_labels = [[0] * self.encoder_max_seq_length for i in range(len(role_ids))]
        role_end_labels = [[0] * self.encoder_max_seq_length for i in range(len(role_ids))]
        for argument in event["mentions"]:
            span = argument["span"]
            role_index = role_ids.index(role_type2id[argument["role"]])
            argument2new_index = [old_token2new_index[span[0]][0], old_token2new_index[span[1]-1][1]]
            role_start_labels[role_index][argument2new_index[0]] = 1
            role_end_labels[role_index][argument2new_index[1]-1] = 1 
        argument_list = [[argument["span"][0], argument["span"][1], role_type2id[argument["role"]]] for argument in event["mentions"]]
        
        return  encoder_input_ids, encoder_attention_mask, context_mask, token_num, old_token2new_index, new_piece2old_index, \
                entity_seq_labels, entity_list, event_type_id, role_ids, role_spans, role_entity_spans, role_start_labels, role_end_labels, argument_list

                
    def encode_to_id(self, text, max_seq_length, max_real_length):
        inputs = self.tokenizer(text)
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        max_real_length = max(max_real_length, len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (max_seq_length-len(input_ids))
        attention_mask += [0] * (max_seq_length-len(attention_mask))
        return inputs, input_ids, attention_mask, max_real_length


    def collate_fn(self, batch):
        encoder_input_ids = torch.LongTensor([data["encoder_input_ids"] for data in batch])
        encoder_attention_mask = torch.FloatTensor([data["encoder_attention_mask"] for data in batch])
        context_mask = torch.FloatTensor([data["context_mask"] for data in batch])
        token_nums = torch.LongTensor([data["token_num"] for data in batch])
        old_token2new_index = [data["old_token2new_index"] for data in batch]
        old_token2new_index = pad_and_align(old_token2new_index, [0,0])
        entity_seq_labels = [data["entity_seq_labels"] for data in batch]
        entity_seq_labels = pad_and_align(entity_seq_labels, 0)
        entity_lists = [data["entity_list"] for data in batch]
        entity_lists = pad_and_align(entity_lists, [0,0,-1], max_num=self.max_entity_num)
        role_spans = [data["role_spans"] for data in batch]
        role_spans = pad_and_align(role_spans, [0,0])
        role_entity_spans = [data["role_entity_spans"] for data in batch]
        role_entity_spans = pad_and_align(role_entity_spans, [0,0])
        role_start_labels = [data["role_start_labels"] for data in batch]
        role_start_labels = pad_and_align(role_start_labels, 0, uni=False, float_type=True)
        role_end_labels = [data["role_end_labels"] for data in batch]
        role_end_labels = pad_and_align(role_end_labels, 0, uni=False, float_type=True)
            
        if self.mission == "train":
            return  {
                        "encoder_input_ids": encoder_input_ids,
                        "encoder_attention_mask": encoder_attention_mask,
                        "context_mask": context_mask,
                        "token_nums": token_nums,
                        "old_token2new_index": old_token2new_index,
                        "entity_seq_labels": entity_seq_labels,
                        "entity_lists": entity_lists,
                        "role_spans": role_spans,
                        "role_entity_spans": role_entity_spans,
                        "role_start_labels": role_start_labels,
                        "role_end_labels": role_end_labels
                    }
        else:
            doc_ids = [data["doc_id"] for data in batch]
            sentence_ids = [data["sentence_id"] for data in batch]
            texts = [data["text"] for data in batch]
            event_ids = [data["event_id"] for data in batch]
            new_piece2old_index = [data["new_piece2old_index"] for data in batch]
            event_type_ids = [data["event_type_id"] for data in batch]
            role_ids = [data["role_ids"] for data in batch]
            argument_lists = [data["argument_list"] for data in batch]
            return  {
                        "doc_ids": doc_ids,
                        "sentence_ids": sentence_ids,
                        "texts": texts,
                        "event_ids": event_ids,
                        "encoder_input_ids": encoder_input_ids,
                        "encoder_attention_mask": encoder_attention_mask,
                        "context_mask": context_mask,
                        "token_nums": token_nums,
                        "old_token2new_index": old_token2new_index,
                        "new_piece2old_index": new_piece2old_index,
                        "entity_seq_labels": entity_seq_labels,
                        "entity_lists": entity_lists,
                        "event_type_ids": event_type_ids,
                        "role_ids": role_ids,
                        "role_spans": role_spans, 
                        "role_entity_spans": role_entity_spans,
                        "role_start_labels": role_start_labels,
                        "role_end_labels": role_end_labels,
                        "argument_lists": argument_lists
                    }
            

def old_to_new(inputs, old_token2char_index):
    old_token2new_index = list() 
    for (char_start, char_end) in old_token2char_index:
        new_start = inputs.char_to_token(char_start)
        new_end = inputs.char_to_token(char_end) + 1
        old_token2new_index.append([new_start, new_end])    
    return old_token2new_index


def pad_and_align(item2index, pad, uni=True, float_type=False, max_num=None, device=None):
    if max_num is None:
        max_num = max([len(data) for data in item2index])
    if uni:
        pad_data = [data[:max_num] + [pad] * max((max_num-len(data)),0) for data in item2index]
    else:
        pad_data = []
        max_len = max([max([len(item) for item in data]) for data in item2index])
        for data in item2index:
            pad_item = []
            for item in data:
                pad_item.append(item + [pad] * (max_len-len(item)))
            pad_data.append(pad_item + [[pad] * max_len for i in range(max_num-len(data))])
    item2index = torch.LongTensor(pad_data) if not float_type else torch.FloatTensor(pad_data)
    if device != None:
        item2index = item2index.to(device=device) 
    return item2index


def data_loader(mission, args, tokenizer, shuffle=False):
    assert mission in ["train", "eval", "test"], "Invalid Mission Name!"
    batch_size = args.eval_batch_size
    if mission == "train":
        file = args.train_file
        batch_size = args.train_batch_size
    elif mission == "eval":
        file = args.valid_file
    else:
        file = args.test_file
    with open(file,"r",encoding="utf-8") as fs:
        dataset = json.loads(fs.read())
    
    dataset = dataset[:int(args.sample_ratio*len(dataset))]
    data = Data(mission, dataset, tokenizer, args)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=data.collate_fn, drop_last=args.drop_last)

    return loader


def get_schema_dict(schema_file):
    res = dict()
    with open(schema_file,"r",encoding="utf-8") as fs:
        schema = json.loads(fs.read())
    res["entity_type2id"] = {item:i for i,item in enumerate(schema["entity_types"])}
    res["entity_id2type"] = {i:item for i,item in enumerate(schema["entity_types"])}
    res["event_type2id"] = {item:i for i,item in enumerate(schema["event_types"])}
    res["event_id2type"] = {i:item for i,item in enumerate(schema["event_types"])}
    res["role_type2id"] = {item:i for i,item in enumerate(schema["roles"])}
    res["role_id2type"] = {i:item for i,item in enumerate(schema["roles"])}
    res["merge_entity_type2id"] = {item:i for i,item in enumerate(schema["merge_entities"])}
    res["merge_entity_id2type"] = {i:item for i,item in enumerate(schema["merge_entities"])}
    res["bio_type2id"] = {"O": 0}
    res["bio_id2type"] = {0: "O"}
    index = 1
    for name in schema["entity_types"]:
        for pre in ["B-", "I-"]:
            res["bio_type2id"][pre+name] = index
            res["bio_id2type"][index] = pre+name
            index += 1
    res["entity_bio_map"] = {k: [res["bio_type2id"]["B-"+v], res["bio_type2id"]["I-"+v]] for k,v in res["entity_id2type"].items()}
    res["entity_bio_map"][len(res["entity_id2type"])] = [res["bio_type2id"]["O"], res["bio_type2id"]["O"]]
      
    event_role_map = dict()
    for event_type, roles in schema["event_to_role_to_entities"].items():
        event_role_map[res["event_type2id"][event_type]] = [res["role_type2id"][role] for role in roles]   
    res["event_role_map"] = event_role_map   
    
    event_role_entity_map = dict()
    for event_type, roles in schema["event_role_entity_map"].items():
        event_type_id = res["event_type2id"][event_type]
        event_role_entity_map[event_type_id] = dict()
        for role, entity_type in roles.items():
            event_role_entity_map[event_type_id][res["role_type2id"][role]] = res["merge_entity_type2id"][entity_type]
    res["event_role_entity_map"] = event_role_entity_map
      
    return res


def get_template_dict(template_file, schema):
    res = dict()
    with open(template_file,"r",encoding="utf-8") as fs:
        template = json.loads(fs.read())
    for event_type, role_template in template.items():
        event_type_id = schema["event_type2id"][event_type]
        res[event_type_id] = dict()
        res[event_type_id]["role_template"] = role_template
        roles = schema["event_role_map"][event_type_id]
        role_list = list()
        for role_id in roles:
            role = schema["role_id2type"][role_id]
            matched = re.search(r'\b'+re.escape(role)+r'\b', role_template)
            assert matched!=None, f"{role} is not in the template-{role_template}!"
            role_list.append([matched.span()[0], matched.span()[1], role_id])
        role_list.sort(key=lambda x:(x[0], x[1]))
        res[event_type_id]["role_list"] = role_list
        role_entity_list = list()
        new_role_list = list()
        role_entity_template = ""
        role_last_end = 0
        for role_tuple in role_list:
            entity_id = schema["event_role_entity_map"][event_type_id][role_tuple[-1]]
            entity_type = schema["merge_entity_id2type"][entity_id]
            role_entity_template += role_template[role_last_end: role_tuple[0]]
            new_role_list.append([len(role_entity_template), len(role_entity_template)+role_tuple[1]-role_tuple[0], role_tuple[-1]])
            role_entity_template += role_template[role_tuple[0]: role_tuple[1]] + " ( "       
            role_entity_list.append([len(role_entity_template), len(role_entity_template)+len(entity_type), entity_id])
            role_entity_template += entity_type + " )"
            role_last_end = role_tuple[1]
        role_entity_template += role_template[role_last_end:]
        res[event_type_id]["role_entity_template"] = role_entity_template
        res[event_type_id]["new_role_list"] = new_role_list
        res[event_type_id]["role_entity_list"] = role_entity_list
    return res