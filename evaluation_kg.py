import os
os.environ["NUM_WORKERS"] = "1"
from utils.convert_prompt import *
from utils.metrics import bert_similarity
from utils.load_kg_ann import load_kg_from_ann, path_matcher
import argparse
import json
import csv
from tqdm import tqdm
from amr2brat import convert_sent2EventGraph
# from conceptnet_test import conceptNet


parser = argparse.ArgumentParser()
# output/infer_output_s277_GPT-3_t4.json
parser.add_argument('--eval_file_path', type=str, default="experiment/gpt3.5-neural_symbolic.json")
parser.add_argument('--is_golden', action='store_true')
parser.add_argument('--use_amr_kg', action='store_true')
args = parser.parse_args()
raw_data = json.load(open(args.eval_file_path))
raw_data = json.load(open(args.eval_file_path))
puzzle_dict = DefaultDict()
solution_ref = DefaultDict()
solution_hypo = DefaultDict()
question_hypo = DefaultDict()
for item in raw_data:
    puzzle_dict[f'puzzle_{len(puzzle_dict)}'] = item['puzzle']
    try:
        solution_ref[f"puzzle_{len(solution_ref)}"] = item["final_answer"]
    except:
        solution_ref[f"puzzle_{len(solution_ref)}"] = item["final_answer"][0]
    # import ipdb;ipdb.set_trace()
    try:
        solution_hypo[f"puzzle_{len(solution_hypo)}"] = item['solution_history'][-1]
    except:
        import ipdb;ipdb.set_trace()
    if 'gen_question_list' in item:
        question_hypo[f"question_{len(question_hypo)}"] = item['gen_question_list']

ann_path = 'situation-data/KG_annotation/{}.ann'

def highlight_path(puzzle_kg, truth_kg, questions, solution_list, index):
    raw_data[index]['question_hit_paths'] = []
    raw_data[index]['solution_infer_hit_path'] = []
    # question_hit_paths = []
    # solution_infer_hit_paths = []
    # if len(questions)>0:
    #     import ipdb;ipdb.set_trace()
    for question in questions:
        question = question.strip()
        question_kg = convert_sent2EventGraph(question)
        simi_puzz_ques_path = path_matcher([puzzle_kg, truth_kg], question_kg)
        print('similar puzzle-question path:', simi_puzz_ques_path)
        # question_hit_paths.extend(simi_puzz_ques_path)
        raw_data[index]['question_hit_paths'].append(simi_puzz_ques_path)
    assert len(raw_data[index]['question_hit_paths'])==len(questions)
    for solution in solution_list:
        solution_hypo_kg = convert_sent2EventGraph(solution)
        similar_puzzle_path = path_matcher([truth_kg], solution_hypo_kg)
        print('similar solution path:', similar_puzzle_path)
        # solution_infer_hit_paths.extend(similar_puzzle_path)
        raw_data[index]['solution_infer_hit_path'].append(similar_puzzle_path)

# import ipdb;ipdb.set_trace()
is_golden = args.is_golden
use_amr_kg = args.use_amr_kg
for i in tqdm(range(len(puzzle_dict))):
    # import ipdb;ipdb.set_trace()
    item = raw_data[i]
    if not use_amr_kg:
        if 'id' in item:
            id = item['id']
            puzzle_kg = load_kg_from_ann(ann_path.format(f'{id}.puzzle'))
            truth_kg = load_kg_from_ann(ann_path.format(f'{id}.truth'))
        else:
            puzzle_kg = load_kg_from_ann(ann_path.format(f'{i+1}.puzzle'))
            truth_kg = load_kg_from_ann(ann_path.format(f'{i+1}.truth'))
    else:
        puzzle_kg = convert_sent2EventGraph(item['puzzle'])
        truth_kg = convert_sent2EventGraph(item['final_answer'])
    if is_golden:
        
        highlight_path(puzzle_kg, truth_kg, item['question_list'], [item['final_answer']], i)
    else:
        if 'gen_question_list' in item:
            highlight_path(puzzle_kg, truth_kg, item['gen_question_list'], item['solution_history'], i)
        else:
            # import ipdb;ipdb.set_trace()
            highlight_path(puzzle_kg, truth_kg, [], item['solution_history'], i)

if use_amr_kg:
    output_suffix = '_AmrKG'
else:
    output_suffix = ''
if is_golden:
    output_path = args.eval_file_path.replace('.json', output_suffix+'_newamr_golden.json')
else:
    output_path = args.eval_file_path.replace('.json', output_suffix+'_newamr_infer.json')
with open(output_path, 'w') as f:
    json.dump(raw_data, f)
# import ipdb;ipdb.set_trace()