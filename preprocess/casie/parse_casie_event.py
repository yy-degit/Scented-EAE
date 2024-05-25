import argparse
import os
import jsonlines
import json
from collections import defaultdict
import copy


def one_fold(fold, data_dir, output_dir):
    res = list()
    with open(os.path.join(data_dir,fold+".jsonlines"), "r", encoding="utf-8") as fs:
        data = jsonlines.Reader(fs)
        for item in data:
            entity_mentions = defaultdict(list)
            event_mentions = defaultdict(list)
            for event in item["event"]:
                for mention in event["mentions"]:
                    nugget = mention["nugget"]
                    sent_id = nugget["tokens"][0][0]
                    if nugget["tokens"][0][0] != nugget["tokens"][-1][0]:
                        continue
                    event_mention = {"id": mention["id"], "type": mention["subtype"], "trigger": {"indexes": [x[1] for x in nugget["tokens"]],}, "arguments": [],}
                    for argument in mention["arguments"]:
                        if argument["tokens"][0][0] != argument["tokens"][-1][0]:
                            continue
                        arg_sent_id = argument["tokens"][0][0]
                        entity_mention = {"id": argument["id"], "indexes": [x[1] for x in argument["tokens"]], "type": argument["filler_type"],}
                        entity_mentions[arg_sent_id].append(entity_mention)
                        if arg_sent_id == sent_id:
                            event_mention["arguments"].append({"id": argument["id"], "role": argument["role"],})
                    event_mentions[sent_id].append(event_mention)
                    
            for sent_id, sentence in enumerate(item["sentences"]):
                token_list = [token["word"] for token in sentence["tokens"]]
                entity_mention = entity_mentions[sent_id]
                event_mention = event_mentions[sent_id]
                instance = {"id": item["id"], "sentence_id": sent_id, "tokens": token_list}
                entities = dict()
                for entity in entity_mention:
                    indexes = entity["indexes"]
                    tokens = [token_list[id] for id in indexes]
                    entities[entity["id"]] = {"id": entity["id"], "type": entity["type"], "indexes": indexes, "word": tokens}
                events = dict()
                for event in event_mention:
                    indexes = event["trigger"]["indexes"]
                    tokens = [token_list[id] for id in indexes]
                    arguments = list()
                    for argument in event["arguments"]:
                        cor_entity = copy.deepcopy(entities[argument["id"]])
                        cor_entity["role"] = argument["role"]
                        arguments.append(cor_entity)
                    events[event["id"]] = {"id": event["id"], "type": event["type"], "indexes": indexes, "word": tokens, "arguments": arguments}
                instance["entities"] = [entity for entity in entities.values()]
                instance["events"] = [event for event in events.values()]
                res.append(instance)   
                     
    with open(os.path.join(output_dir,fold+".json"), "w", encoding="utf-8") as fs:
        json.dump(res,fs,indent=4,ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CASIE event data.")
    parser.add_argument("--data_dir", default="data/dataset/casie/target" , help="Name for data directory.")
    parser.add_argument("--output_dir", default="data/row_data/casie" , help="Name for output directory.")
    args = parser.parse_args()
    
    for fold in ["train", "valid", "test"]:
        one_fold(fold, args.data_dir, args.output_dir)