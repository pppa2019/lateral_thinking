import sys
import numpy as np
sys.path.append('.')
from utils.metrics import bert_similarity
def load_kg_from_ann(file_path='amr_test_ann/4.ann'):
    span_dict = {}
    relation_dict = {}
    event_dict = {}
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            trunks = line.split()
            if line[0]=='T':
                span_dict[trunks[0]] = trunks[1:]
            elif line[0]=='E':
                event_dict[trunks[0]] = trunks[1:]
            elif line[0]=='R':
                relation_dict[trunks[0]] = trunks[1:]
    return span_dict, relation_dict, event_dict


def convert_KG2PathSent(span_dict, relation_dict, event_dict):
    def abbr_projector(item_name):
        abbr = item_name.split(':')[-1]
        if abbr[0]=='T':
            return span_dict[abbr][-1]
        else:
            return abbr_projector(event_dict[abbr][0])
    path_sents = []
    
    for item in relation_dict.values():
        path_sents.append(abbr_projector(item[-2])+'[SEP]'+abbr_projector(item[-1]))

    for item in event_dict.values():
        path_sents.append('[SEP]'.join([abbr_projector(name) for name in item]))
        
    return path_sents

def path_matcher(kg1, kg2, matcher=bert_similarity, threshold=0.5):
    if isinstance(kg1, list):
        path_list1 = []
        for kg in kg1:
            path_list1.extend(convert_KG2PathSent(*kg))
    else:
        path_list1 = convert_KG2PathSent(*kg1)
    # import ipdb;ipdb.set_trace()
    path_list2 = convert_KG2PathSent(*kg2)
    if len(path_list1)==0 or len(path_list2)==0:
        return []
    similar_paths = []
    score = matcher(path_list1, path_list2)
    match_path1_id, match_path2_id = np.where(score>threshold)
    # import ipdb;ipdb.set_trace()
    for id1, id2 in zip(match_path1_id, match_path2_id):
        similar_paths.append([path_list1[id1], path_list2[id2], str(score[id1][id2])])
        print(similar_paths[-1])
    # for path1 in path_list1:
    #     for path2 in path_list2:
    #         score = matcher(path1, path2)
    #         if score > threshold:
    #             similar_paths.append((path1, path2, score))
    #             print(similar_paths[-1])
    # import ipdb;ipdb.set_trace()
    return similar_paths

    
if __name__=="__main__":
    puzzle_kg = load_kg_from_ann('situation-data/KG_annotation/4.puzzle.ann')
    truth_kg = load_kg_from_ann('situation-data/KG_annotation/4.truth.ann')
