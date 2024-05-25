import argparse
import os
import json
import copy
from collections import OrderedDict


def deal(args):
    schema = init_schema()
    argument_nums = schema["argument_nums"]
    for file in [args.train_file, args.valid_file, args.test_file]:
        print(f"{file} is dealing...")
        with open(os.path.join(args.input_path,file),"r",encoding="utf-8") as fs:
            data = json.loads(fs.read())
        res = dataset_map_func[args.dataset](data)
        with open(os.path.join(args.output_path,file),"w",encoding="utf-8") as fs:
            json.dump(res,fs,indent=4,ensure_ascii=False)
        schema = update_schema(schema, res)
        schema[file] = len(res)
        schema[file + "_argument_nums"] = schema["argument_nums"] - argument_nums
        argument_nums = schema["argument_nums"]
    schema = get_role2entity_groups(schema)
    schema = get_entity_map(schema, args.entity_file)
    schema = schema_to_json(schema)
    with open(os.path.join(args.output_path,args.schema_file),"w",encoding="utf-8") as fs:
        json.dump(schema,fs,indent=4,ensure_ascii=False)


def init_schema():
    schema = OrderedDict()
    schema["entity_types"] = set()
    schema["event_types"] = set()
    schema["roles"] = set()
    schema["role_entity_types"] = set()
    schema["role_entity_map"] = OrderedDict()
    schema["event_to_role_to_entities"] = OrderedDict()
    schema["event_to_role_limits"] = OrderedDict()
    schema["role_to_entity_groups"] = list()
    schema["merge_entities"] = set()
    schema["event_role_entity_map"] = OrderedDict()
    schema["entity_type_nums"] = 0
    schema["event_type_nums"] = 0
    schema["role_nums"] = 0
    schema["argument_nums"] = 0
    schema["group_nums"] = 0
    schema["max_entity_num"] = 0
    return schema


def update_schema(schema, data):
    for example in data:
        schema["max_entity_num"] = max(schema["max_entity_num"], len(example["entities"]))
        for entity in example["entities"]:
            schema["entity_types"].add(entity["type"])
        for event in example["events"]:
            schema["event_types"].add(event["type"])
            if event["type"] not in schema["event_to_role_to_entities"]:
                schema["event_to_role_to_entities"][event["type"]] = OrderedDict()
                schema["event_to_role_limits"][event["type"]] = OrderedDict()
            role_nums = dict()
            for argument in event["mentions"]:
                schema["roles"].add(argument["role"])
                if argument["role"] not in role_nums:
                    role_nums[argument["role"]] = 0
                role_nums[argument["role"]] += 1
                if argument["role"] not in schema["event_to_role_to_entities"][event["type"]]:
                    schema["event_to_role_to_entities"][event["type"]][argument["role"]] = set()
                    schema["event_to_role_limits"][event["type"]][argument["role"]] = [10,0]
                cor_entity_type = example["entities"][argument["entity_id"]]["type"]
                schema["event_to_role_to_entities"][event["type"]][argument["role"]].add(cor_entity_type)
                schema["argument_nums"] += 1
            for role,num in role_nums.items():
                schema["event_to_role_limits"][event["type"]][role][1] = max(num, schema["event_to_role_limits"][event["type"]][role][1])
                schema["event_to_role_limits"][event["type"]][role][0] = min(num, schema["event_to_role_limits"][event["type"]][role][0])
    for example in data:
        if len(example["entities"]) == 0:
            continue
        for event in example["events"]:
            role_nums = {role: 0 for role in schema["event_to_role_limits"][event["type"]]}
            for argument in event["mentions"]:
                role_nums[argument["role"]] += 1
            for role,num in role_nums.items():
                schema["event_to_role_limits"][event["type"]][role][0] = min(num, schema["event_to_role_limits"][event["type"]][role][0])
    return schema   


def get_role2entity_groups(schema):
    for role_to_entities in schema["event_to_role_to_entities"].values():
        for entities in role_to_entities.values():
            if list(entities) not in schema["role_to_entity_groups"]:
                schema["role_to_entity_groups"].append(list(entities))
    schema["group_nums"] = len(schema["role_to_entity_groups"])
    return schema


def get_entity_map(schema, entity_file):
    with open(entity_file, "r", encoding="utf-8") as fs:
        data = json.load(fs)
    schema["role_entity_types"] = data["role_entity_types"]
    schema["role_entity_map"] = data["role_entity_map"]
    schema["merge_entities"] = data["merge_entities"]
    schema["event_role_entity_map"] = data["event_role_entity_map"]
    return schema

    
def schema_to_json(schema):
    schema["entity_type_nums"] = len(schema["entity_types"])
    schema["event_type_nums"] = len(schema["event_types"])
    schema["role_nums"] = len(schema["roles"])
    schema["entity_types"] = list(schema["entity_types"])
    schema["event_types"] = list(schema["event_types"])
    schema["roles"] = list(schema["roles"])
    event_to_role_to_entities = OrderedDict()
    for event_type, role_to_entities in schema["event_to_role_to_entities"].items():
        event_to_role_to_entities[event_type] = {k: list(v) for k,v in role_to_entities.items()}
    schema["event_to_role_to_entities"] = event_to_role_to_entities
    return schema
    

def ace(data):
    res = list()
    for row in data:
        doc_id = row["doc_key"]
        texts = row["sentences"]
        all_entities = row["ner"]
        all_events = row["events"]
        sentence_starts = row["sentence_start"]
        assert len(texts) == len(all_entities) == len(all_events) == len(sentence_starts), f"{row['doc_key']} contains different size of items!"
        sentence_id = 0
        for text, entities, events, s_start in zip(texts, all_entities, all_events, sentence_starts):
            example = OrderedDict()
            example["doc_id"] = doc_id 
            example["sentence_id"] = sentence_id
            example["text"] = text
            entity_list = list()
            entity_span_list = list()
            for entity in entities:
                entity_list.append({"id": 0, "type": ace_entity_mapping[entity[2]], "span": [entity[0]-s_start, entity[1]-s_start+1], "word": text[entity[0]-s_start: entity[1]-s_start+1]})
            entity_list.sort(key=lambda x:(x["span"][0],x["span"][1]))
            for i,entity in enumerate(entity_list):
                entity_list[i]["id"] = i
                entity_span_list.append(entity["span"])
            example["entities"] = entity_list
            event_list = list()
            for event in events:
                one_event = {"type": event[0][1], "trigger": {"span": [event[0][0]-s_start, event[0][0]-s_start+1], "word": text[event[0][0]-s_start: event[0][0]-s_start+1]}, "mentions": []}
                for argument in event[1:]:
                    argument_span = [argument[0]-s_start, argument[1]-s_start+1]
                    one_event["mentions"].append({"role": argument[2], "span": argument_span, "word": text[argument_span[0]: argument_span[1]], "entity_id": entity_span_list.index(argument_span)})
                one_event["mentions"].sort(key=lambda x:(x["entity_id"],x["span"][0],x["span"][1]))
                event_list.append(one_event)
            event_list.sort(key=lambda x:(x["trigger"]["span"][0],x["trigger"]["span"][1]))
            example["events"] = event_list
            res.append(example)
            sentence_id += 1
    return res


def aceplus(data):
    res = list()
    for row in data:
        example = OrderedDict()        
        example["doc_id"] = row["doc_id"]
        example["sentence_id"] = row["sent_id"]
        example["text"] = row["tokens"]
        text = row["tokens"]
        entities = row["entity_mentions"]
        events = row["event_mentions"]
        entity_list = list()
        entity_span_dict = dict()
        entity_id_dict = dict()
        for entity in entities:
            entity_list.append({"id": entity["id"], "type": ace_entity_mapping[entity["entity_type"]], "span": [entity["start"], entity["end"]], "word": text[entity["start"]: entity["end"]]})
            entity_span_dict[entity["id"]] = [entity["start"], entity["end"]]
        entity_list.sort(key=lambda x:(x["span"][0],x["span"][1]))
        for i,entity in enumerate(entity_list):
            entity_id_dict[entity["id"]] = i
            entity_list[i]["id"] = i
        example["entities"] = entity_list
        event_list = list()
        for event in events:
            one_event = {"type": event["event_type"].replace(":","."), "trigger": {"span": [event["trigger"]["start"], event["trigger"]["end"]], "word": text[event["trigger"]["start"]: event["trigger"]["end"]]}, "mentions": []}
            for argument in event["arguments"]:
                argument_span = entity_span_dict[argument["entity_id"]]
                one_event["mentions"].append({"role": argument["role"], "span": argument_span, "word": text[argument_span[0]: argument_span[1]], "entity_id": entity_id_dict[argument["entity_id"]]})
            one_event["mentions"].sort(key=lambda x:(x["entity_id"],x["span"][0],x["span"][1]))
            event_list.append(one_event)
        event_list.sort(key=lambda x:(x["trigger"]["span"][0],x["trigger"]["span"][1]))
        example["events"] = event_list
        res.append(example)
    return res


def casie(data):
    res = list()
    for row in data:
        example = OrderedDict()        
        example["doc_id"] = row["id"]
        example["sentence_id"] = row["sentence_id"]
        example["text"] = row["tokens"]
        entities = row["entities"]
        events = row["events"]
        entity_list = list()
        entity_id_dict = dict()
        for entity in entities:
            entity_list.append({"id": entity["id"], "type": entity["type"], "span": [entity["indexes"][0], entity["indexes"][-1]+1], "word": entity["word"]})
        entity_list.sort(key=lambda x:(x["span"][0],x["span"][1]))
        for i,entity in enumerate(entity_list):
            entity_id_dict[entity["id"]] = i
            entity_list[i]["id"] = i
        example["entities"] = entity_list
        event_list = list()
        for event in events:
            one_event = {"type": event["type"], "trigger": {"span": [event["indexes"][0], event["indexes"][-1]+1], "word": event["word"]}, "mentions": []}
            for argument in event["arguments"]:
                one_event["mentions"].append({"role": argument["role"], "span": [argument["indexes"][0], argument["indexes"][-1]+1], "word": argument["word"], "entity_id": entity_id_dict[argument["id"]]})
            one_event["mentions"].sort(key=lambda x:(x["entity_id"],x["span"][0],x["span"][1]))
            event_list.append(one_event)
        event_list.sort(key=lambda x:(x["trigger"]["span"][0],x["trigger"]["span"][1]))
        example["events"] = event_list
        res.append(example)
    return res


def ere(data):
    res = list()
    for row in data:
        example = OrderedDict()        
        example["doc_id"] = row["doc_id"]
        example["sentence_id"] = row["sent_id"]
        example["text"] = row["tokens"]
        text = row["tokens"]
        entities = row["entity_mentions"]
        events = row["event_mentions"]
        entity_list = list()
        entity_span_dict = dict()
        entity_id_dict = dict()
        for entity in entities:
            entity_list.append({"id": entity["id"], "type": ace_entity_mapping[entity["entity_type"]], "span": [entity["start"], entity["end"]], "word": text[entity["start"]: entity["end"]]})
            entity_span_dict[entity["id"]] = [entity["start"], entity["end"]]
        entity_list.sort(key=lambda x:(x["span"][0],x["span"][1]))
        for i,entity in enumerate(entity_list):
            entity_id_dict[entity["id"]] = i
            entity_list[i]["id"] = i
        example["entities"] = entity_list
        event_list = list()
        for event in events:
            one_event = {"type": event["event_type"].replace(":","."), "trigger": {"span": [event["trigger"]["start"], event["trigger"]["end"]], "word": text[event["trigger"]["start"]: event["trigger"]["end"]]}, "mentions": []}
            for argument in event["arguments"]:
                argument_span = entity_span_dict[argument["entity_id"]]
                one_event["mentions"].append({"role": argument["role"], "span": argument_span, "word": text[argument_span[0]: argument_span[1]], "entity_id": entity_id_dict[argument["entity_id"]]})
            one_event["mentions"].sort(key=lambda x:(x["entity_id"],x["span"][0],x["span"][1]))
            event_list.append(one_event)
        event_list.sort(key=lambda x:(x["trigger"]["span"][0],x["trigger"]["span"][1]))
        example["events"] = event_list
        res.append(example)
    return res



if __name__ == "__main__":
    dataset_map_func = {
        "ace": ace,
        "aceplus": aceplus,
        "casie": casie,
        "ere": ere
    }    
    ace_entity_mapping = {
    "LOC": "Location",
    "WEA": "Weapon",
    "PER": "Person",
    "FAC": "Facilities",
    "GPE": "Geo-Political",
    "ORG": "Organization",
    "VEH": "Vehicle"
    }  
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="potefin", help="Dataset Name.")
    parser.add_argument("--raw_data_dir", type=str, default="data/raw_data", help="Directory of raw data.")
    parser.add_argument("--train_file", type=str, default="train.json", help="File name of training set.")
    parser.add_argument("--valid_file", type=str, default="valid.json", help="File name of validation set.")
    parser.add_argument("--test_file", type=str, default="test.json", help="File name of test set.")
    parser.add_argument("--entity_file", type=str, default="ace_entity_map.json", help="File name of entity map.")
    parser.add_argument("--schema_file", type=str, default="schema.json", help="File name of schema.")
    parser.add_argument("--output_dir", type=str,  default="data/new_data", help="Directory of dealed data.")
    args = parser.parse_args()
    assert args.dataset in dataset_map_func, "Invalid dataset name!"
    args.input_path = os.path.join(args.raw_data_dir,args.dataset)
    args.output_path = os.path.join(args.output_dir,args.dataset)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    deal(args)
    