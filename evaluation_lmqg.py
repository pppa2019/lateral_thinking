import os
os.environ["NUM_WORKERS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from typing import DefaultDict
from lmqg.automatic_evaluation_tool.bleu.bleu import Bleu
from lmqg.automatic_evaluation_tool.rouge import Rouge
# from lmqg.automatic_evaluation_tool.bertscore import BERTScore
# from lmqg.automatic_evaluation_tool.moverscore import MoverScore
import argparse
import json
import csv
from bleurt import score
import numpy as np
import jieba

parser = argparse.ArgumentParser()
parser.add_argument('--eval_file_path', type=str, default="experiment/chatglm2-neural_symbolic.json")
parser.add_argument('--zh', action='store_true')
args = parser.parse_args()

raw_data = json.load(open(args.eval_file_path))
solution_ref = DefaultDict()
solution_hypo = DefaultDict()
question_ref = DefaultDict()
question_hypo = DefaultDict()
for item in raw_data:
    if 'solution_history' not in item:
        continue
    try:
        if args.zh:
            solution_ref[f"solution_{len(solution_ref)}"] = [' '.join(list(item["final_answer"]))]
        else:
            solution_ref[f"solution_{len(solution_ref)}"] = [item["final_answer"].encode()]
    except:
        if args.zh:
            solution_ref[f"solution_{len(solution_ref)}"] = [' '.join(list(item["final_answer"][0]))]
        else:
            solution_ref[f"solution_{len(solution_ref)}"] = [item["final_answer"][0].encode()]
        
    if args.zh:
        solution_hypo[f"solution_{len(solution_hypo)}"] = [' '.join(list(item['solution_history'][-1]))]
    else:
        try:
            solution_hypo[f"solution_{len(solution_hypo)}"] = [item['solution_history'][-1].split('\n')[0].encode()]
        except:
            # import ipdb;ipdb.set_trace()
            solution_ref.pop(f'solution_{len(solution_hypo)}')
            pass
    
    if 'gen_question_list' in item:
        for gen_question in item['gen_question_list']:
            # for ref_question in item['question_list']:
            if len(item['question_list'])!=0:
                # question_ref[f"question_{len(question_ref)}"] = [ref_question for ref_question in item['question_list']]
                question_ref[f"question_{len(question_ref)}"] = [item['question_list'][0].encode()]
                question_hypo[f"question_{len(question_hypo)}"] = [gen_question.encode()]

# import ipdb;ipdb.set_trace()
output_csv_name = args.eval_file_path.replace('.json', '_metric.csv')
f = open(output_csv_name, 'w')
csv_writer = csv.writer(f, delimiter=',')
csv_writer.writerow(["metric", "solution_score", "question_score"])
for scorer, method in [(Bleu(4), ["Bleu-1", "Bleu-2", "Bleu-3", "Bleu-4"])]:
    puzzle_score, puzzle_scores = scorer.compute_score(solution_ref, solution_hypo)
    if len(question_ref.keys())>0:
        question_score, question_scores = scorer.compute_score(question_ref, question_hypo)
    if isinstance(puzzle_score, list):
        if len(question_ref.keys())>0:
            for m, ps, qs in zip(method, puzzle_score, question_score):
                csv_writer.writerow([m, ps, qs])
        else:
            for m, ps in zip(method, puzzle_score):
                csv_writer.writerow([m, ps])
#     else:
#         if len(question_ref.keys())>0:
#             csv_writer.writerow([method, puzzle_score, question_score])
#         else:
#             csv_writer.writerow([method, puzzle_score])
#     if len(question_ref.keys())>0:
    #     print(f"{method}  puzzle:{puzzle_score} question:{question_score}")
    # else:
    #     print(f"{method}  puzzle:{puzzle_score}")
# import ipdb;ipdb.set_trace()
scorer = score.BleurtScorer('BLEURT-20')
bleurt_solution_score = scorer.score(
    references=[sent[0].decode() for sent in solution_ref.values()],
    candidates=[sent[0].decode() for sent in solution_hypo.values()]
    )

bleurt_question_score = scorer.score(
    references=[sent[0].decode() for sent in question_ref.values()], 
    candidates=[sent[0].decode() for sent in question_hypo.values()]
    )
# import ipdb;ipdb.set_trace()

csv_writer.writerow(['BLEURT', np.mean(bleurt_solution_score), np.mean(bleurt_question_score)])