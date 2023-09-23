from collections import defaultdict
import os
import amrlib
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from fuzzywuzzy import fuzz
import spacy
amrlib.setup_spacy_extension()


def load_files(path='assign_annotation/B_folder'):
    examples = {}
    files = os.listdir(path)
    for file in files:
        if '.txt' in file:
            prefix = file.replace('.txt', '')
            with open(os.path.join(path, file), 'r') as f:
                lines = f.readlines()
                examples[prefix] = ' '.join(lines)
    return examples



def merge_multihop_relationship(instance_dict, relation_triples):
    for key, value in instance_dict.items():
        if value=='cause':
            cause_triples = []
            cause_head = None
            cause_end = None
            for idx, triple in enumerate(relation_triples):
                if triple[0]==key:
                    
                    if triple[1]==':ARG0': cause_head=triple[2]
                    if triple[1]==':ARG1': cause_end=triple[2]
                    cause_triples = relation_triples.pop(idx)
            if (cause_end is not None) & (cause_head is not None):
                relation_triples.append([cause_head, ':cause', cause_end])
        if value=='and':
            temporal_triples = []
            op_dict = {}
            temporal_flag = True
            delete_idx = None
            for idx, triple in enumerate(relation_triples):
                if triple[2]==key and triple[1]!=':Arg0':
                    temporal_flag = False
                    head_entity = triple[0]
                    relationship = triple[1]
                    delete_idx = idx
                if triple[0]==key:
                    if ':op' in triple[1]:
                        op_dict[triple[1].split(':op')[-1]] = triple[2]
            if delete_idx is not None:
                relation_triples.pop(delete_idx)
            if temporal_flag:
                for i in range(1, len(op_dict)):
                    temporal_triples.append([op_dict[str(i)], 'temporal', op_dict[str(i+1)]])
                relation_triples.extend(temporal_triples)
            else:
                # import ipdb;ipdb.set_trace()
                for key, value in op_dict.items():
                    relation_triples.append([head_entity, relationship, value])
            
    return relation_triples


def triples2graph(instance_dict, triples, output_path):
    import networkx as nx
    import pydot
    from matplotlib import pyplot as plt
    from networkx.drawing.nx_pydot import graphviz_layout, pydot_layout
    
    G = nx.DiGraph()
    pos = []
    label_dict = {}
    abbv2id = {}
    
    for idx, (abbv, word) in enumerate(instance_dict.items()):
        abbv2id[abbv] = idx
        pos.append(word)
    for triple in triples:
        try:
            G.add_edge(instance_dict[triple[0]], instance_dict[triple[2]], rel=triple[1])
            label_dict[(abbv2id[triple[0]], abbv2id[triple[2]])] = triple[1]
        except:
            pass
    addition_relationship = []
    template_dict = {
        'head': None,
        'tail': None,
        'rel': None
    }
    if 'cause' in G:
        print('May have CAUSE relationship.')
    if 'and' in G:
        if len(G['and'])==2:
            nodes =  [item for item in G.neighbors('and')]
            if ({'rel': ':ARG0'} in G[nodes[0]].values()) and ({'rel': ':ARG0'} in G[nodes[1]].values()):
                if 'after' in G[nodes[0]]:
                    relation_dict = template_dict.copy()
                    relation_dict['head'] = nodes[0]
                    relation_dict['tail'] = nodes[1]
                    relation_dict['rel'] = 'EventTemperal'
                    addition_relationship.append(relation_dict)
                elif 'after' in G[nodes[1]]:
                    relation_dict = template_dict.copy()
                    relation_dict['head'] = nodes[1]
                    relation_dict['tail'] = nodes[0]
                    relation_dict['rel'] = 'EventTemperal'
                    addition_relationship.append(relation_dict)
                else:
                    relation_dict = template_dict.copy()
                    relation_dict['head'] = nodes[1]
                    relation_dict['tail'] = nodes[0]
                    relation_dict['rel'] = 'EventSameTime'
                    addition_relationship.append(relation_dict)


        print('May have PartOf or SameTime relationship')
    if "after" in G:
        print('May have temporal relationship')
    if "domain" in edge_labels.values():
        print("May have partof or same relationship")
    return addition_relationship

def convert_AMR2EventGraph(instance_dict, relation_triples, raw_sent, begin_idx, span_dict, event_dict, relation_dict, recover_lemma=True):
    span_type = ['Head_End', "Att_span", "Event"]
    relationship_type = [
        [span_type[0], "Entity_att", span_type[1]], #mod :ARG0-of
        [span_type[2], "Head", span_type[0]], # ARG0
        [span_type[2], "End", span_type[0]], # ARG1
        [span_type[0], "EntityPartOf", span_type[0]], # domain op op
        [span_type[0], "EntityEqual", span_type[1]], # domain name op
        [span_type[2], "EventSameTime", span_type[2]], # and op1 op2 event, should preprocess the triples and convert.
        [span_type[2], "EventCause", span_type[2]], # cause, argw w
        [span_type[2], "EventTemporal", span_type[2]], # and op op time after
        [span_type[2], "EventEqual", span_type[2]], # and op op
        [span_type[1], "Event_att", span_type[2]], # ARG2
    ]
    amr_converter = {
        ':ARG0': relationship_type[1],
        ':ARG1': relationship_type[2],
        # ':ARG2': relationship_type[0],
        'temporal':relationship_type[-3],
        ':ARG1-of': relationship_type[0],
        ':mod': relationship_type[0],
        ":ARG0-of": relationship_type[0],
        ":cause": relationship_type[-4]
    }
    word_list = raw_sent.split()

    instance2spanid = {}
    instance2eventid = {}
    event_entity_dict = defaultdict(list)

    def fuzzy_match_word(word):
        threashold = 70
        max_score = 0
        best_match = None
        result_len = 0
        wnl = WordNetLemmatizer()
        if word not in word_list:
            for origin_word in word_list:
                lemma_word = wnl.lemmatize(origin_word, 'v')
                lemma_word = wnl.lemmatize(lemma_word, 'n')
                match_score = fuzz.ratio(word, lemma_word)
                if match_score>=threashold:
                    if match_score>max_score:
                        best_match = origin_word  
                        max_score = match_score
            if best_match is not None:
                result_len = len(best_match)
            return best_match, result_len
        return word, len(word)

    for item in relation_triples:
        if  item[1] in amr_converter.keys():
            schema = amr_converter[item[1]]
            span_type0 = schema[0]
            relation = schema[1]
            span_type1 = schema[2]
            try:
                instance0 = instance_dict[item[0]]
                if recover_lemma:
                    recover_word, _ = fuzzy_match_word(instance0)
                    if recover_word is not None:
                        instance0 = recover_word
            except:
                continue
            try:
                instance1 = instance_dict[item[2]]
                if recover_lemma:
                    recover_word, _ = fuzzy_match_word(instance1)
                    if recover_word is not None:
                        instance1 = recover_word
            except:
                continue
            span_dict[f'T{len(span_dict)+1}'] = [span_type0, instance0]
            instance2spanid[instance0] =  f'T{len(span_dict)}'
            span_dict[f'T{len(span_dict)+1}'] = [span_type1, instance1]
            instance2spanid[instance1] =  f'T{len(span_dict)}'
            
            # event triples contain event and event's relationships
            if span_dict[instance2spanid[instance0]][0]==span_type[2] and span_dict[instance2spanid[instance1]][0]!=span_type[2] and ('att' not in schema[1]):
                event = instance0
                if event not in instance2eventid:
                    instance2eventid[event] = f'E{len(event_dict)+1}'
                if instance2eventid[event] not in event_dict:
                    event_dict[instance2eventid[event]] = [f'Event:{instance2spanid[instance0]}']
                if instance2spanid[instance1] not in event_entity_dict[instance2eventid[event]]:
                    event_entity_dict[instance2eventid[event]].append(instance2spanid[instance1])
                    event_dict[instance2eventid[event]].append(f'{relation}:{instance2spanid[instance1]}')
            # event event relationship
            if span_type0==span_type[2] and span_type1==span_type[2]:
                relation_dict[f"R{len(relation_dict)+1}"] = [schema[1], f'Arg1:{instance2spanid[instance1]}', f'Arg2:{instance2spanid[instance0]}']
            # entity entity relationship

            # attribuite recognization
            if 'att' in schema[1]:
                if span_dict[instance2spanid[instance0]][0]==span_type[2]:
                    schema[1] = 'Event_att'
                relation_dict[f"R{len(relation_dict)+1}"] = [schema[1], f'Arg1:{instance2spanid[instance1]}', f'Arg2:{instance2spanid[instance0]}']

    return span_dict, relation_dict, event_dict

def convert_sent2EventGraph(text):
    import amrlib
    amrlib.setup_spacy_extension()
    nlp = spacy.load('en_core_web_sm')

    doc = nlp(text)
    graphs, sent_triples = doc._.to_amr()
    begin_idx = 0
    relation_dict = {}
    span_dict = {}
    event_dict = {}
    sents = []
    idx = 0
    for sent, graph, triples in zip(doc.sents, graphs, sent_triples):
        sent = sent.text
        sent = sent.replace('.', '')
        sent = sent.replace(',', ' ')
        sent = sent.replace('?', ' ')
        sent = sent.lower()
        sents.append(sent)
        relation_triples = []
        instance_dict = {}
        for item in triples:
            if item[1]==':instance':
                instance_dict[item[0]] = item[2].split('-')[0]
            else:
                relation_triples.append(item)
        
        try:
            triples2graph(instance_dict, relation_triples, f'visualize_graph/amr{idx}.svg')
            idx += 1
        except:
            pass
        relation_triples = merge_multihop_relationship(instance_dict, relation_triples)
        span_dict, relation_dict, event_dict = convert_AMR2EventGraph(instance_dict, relation_triples, sent, begin_idx, span_dict, event_dict, relation_dict)
        begin_idx += len(sent)+2
    event_triples = []
    span_name_dict = {}
    for key, value in span_dict.items():
        span_name_dict[key] = value[-1]
    for _, value in event_dict.items():
        event_id = value[0].split(':')[-1]
        for i in range(1, len(value)):
            event_triples.append((event_id, value[i].split(':')[0],value[i].split(':')[1]))
    for key, value in relation_dict.items():
        event_triples.append([value[1].split(':')[-1], value[0], value[2].split(':')[-1]])
    return span_dict, relation_dict, event_dict


if __name__=='__main__':
    test_sent = [
        # 'I ate an apple and a banana.',
        "A man goes into a restaurant, orders albatross, eats one bite, and kills himself."
        ]
    for sent in test_sent:
        print(sent)
        print(convert_sent2EventGraph(sent))
        print('*-'*20)

    