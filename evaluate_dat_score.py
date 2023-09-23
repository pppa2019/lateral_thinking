import json
from tqdm import tqdm
from utils.metrics import dat_score, solution_dat
import numpy as np
path = 'experiment/gpt3.5-neural_symbolic.json'
eval_data = json.load(open(path))

golden_question_dats = []
infer_question_dats = []
golden_sq_dats = []
infer_sq_dats = []

for item in tqdm(eval_data):
    if len(item['question_list']) > 1:
        golden_question_dats.append(dat_score(item['question_list']))
        infer_question_dats.append(dat_score(item['gen_question_list']))
    if len(item['question_list'])>0:
        golden_sq_dats.append(solution_dat(item['final_answer'], item['question_list']))
        infer_sq_dats.append(solution_dat(item['final_answer'], item['gen_question_list']))

    
print('golden question dat:', np.mean(golden_question_dats))
print('infer question dat:', np.mean(infer_question_dats))
print('golden solution dat:', np.mean(golden_sq_dats))
print('infer solution dat:', np.mean(infer_sq_dats))