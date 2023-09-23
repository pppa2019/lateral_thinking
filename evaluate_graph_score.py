import json
import numpy as np
from utils.metrics import dat_score, solution_dat
from tqdm import tqdm
import argparse
import csv
from bleurt import score
from utils.load_kg_ann import load_kg_from_ann, convert_KG2PathSent
parser = argparse.ArgumentParser()
# output/infer_output_s277_GPT-3_t4.json
parser.add_argument('--eval_file_path', type=str, default="experiment/gpt3.5-neural_symbolic.json")
args = parser.parse_args()
golden_path = args.eval_file_path.replace('.json', '_newamr_golden.json')
infer_path = args.eval_file_path.replace('.json', '_newamr_infer.json')
scorer = score.BleurtScorer('BLEURT-20')
with open(golden_path, 'r') as f:
    golden_data = json.load(f)
with open(infer_path, 'r') as f:
    infer_data = json.load(f)
f = open(args.eval_file_path.replace('.json', '_metric.csv'), 'a+')
csvwriter = csv.writer(f)

if 'question_hit_paths' in infer_data[0]:
    golden_question_hit_paths = [item['question_hit_paths'] for item in golden_data]
    infer_question_hit_paths = [item['question_hit_paths'] for item in infer_data]

golden_solution_hit_paths = [item['solution_infer_hit_path'] for item in golden_data]
infer_solution_hit_paths = [item['solution_infer_hit_path'] for item in infer_data]

mean_gq_len = []
gq_len = []
mean_inferq_len = []
inferq_len = []
infer_question_bleurt = []
if 'question_hit_paths' in golden_data[0]:
    for idx, golden_question_hit, infer_question_hit in zip(range(277), golden_question_hit_paths, infer_question_hit_paths):

        ann_path = 'situation-data/KG_annotation/{}.ann'
        if 'id' in infer_data[idx]:
            id = infer_data[idx]['id']
        else:
            id = idx+1
        item_puzzle_kg = load_kg_from_ann(ann_path.format(f'{id}.puzzle'))
        item_puzzle_sents = convert_KG2PathSent(*item_puzzle_kg)
        item_solution_kg = load_kg_from_ann(ann_path.format(f'{id}.truth'))
        item_solution_sent = convert_KG2PathSent(*item_solution_kg)
        item_event_sents = item_puzzle_sents + item_solution_sent
        event_triple_num = len(set(item_event_sents))
        
        if len(golden_question_hit)==0:
            continue
        else:
            gq_hit_unique = []
            inferq_hit_unique = []
            for hit_q in golden_question_hit:
                hit_q = [item for item in hit_q if float(item[2])>0.5]
                gq_hit_unique.append(len(set([pairs[0] for  pairs in hit_q]))/event_triple_num)
            for hit_q in infer_question_hit:
                hit_q = [item for item in hit_q if float(item[2])>0.5]
                inferq_hit_unique.append(len(set([pairs[0] for pairs in hit_q]))/event_triple_num)
            item_scores = []
            if 'gen_question_list' in golden_data[idx]:
                for sent in golden_data[idx]['gen_question_list']:
                    scores = scorer.score(
                        references=golden_data[idx]['question_list'],
                        candidates=[sent]*len(golden_data[idx]['question_list'])
                    )
                    item_scores.append(max(scores))
                infer_question_bleurt.append(item_scores)
            gq_len.append(gq_hit_unique)
            inferq_len.append(inferq_hit_unique)
            mean_gq_len.append(np.array(gq_hit_unique).mean())
            mean_inferq_len.append(np.array(inferq_hit_unique).mean())
            # for question, bleurt_score, kg_score in zip(golden_data[idx]['gen_question_list'],  item_scores, inferq_hit_unique):
            #     if 1.0+bleurt_score<kg_score and kg_score>0.2:
            #         print(golden_data[idx]['puzzle'])
            #         print(golden_data[idx]['final_answer'])
            #         print(question, bleurt_score, kg_score)
            #         print('=*'*20)
            #         import ipdb;ipdb.set_trace()
mean_gs_hit_len = []
mean_infers_hit_len = []
infers_hit_len = []
infer_solution_bleurt = []
for idx, golden_solution_hit, infer_solution_hit in zip(range(277), golden_solution_hit_paths, infer_solution_hit_paths):
    ann_path = 'situation-data/KG_annotation/{}.ann'
    if 'id' in infer_data[idx]:
        id = infer_data[idx]['id']
    else:
        id = idx+1
    item_puzzle_kg = load_kg_from_ann(ann_path.format(f'{id}.puzzle'))
    item_puzzle_sents = convert_KG2PathSent(*item_puzzle_kg)
    item_solution_kg = load_kg_from_ann(ann_path.format(f'{id}.truth'))
    item_solution_sent = convert_KG2PathSent(*item_solution_kg)
    item_event_sents = item_puzzle_sents + item_solution_sent
    event_triple_num = len(set(item_event_sents))
    solution_triple_num = len(set(item_solution_sent))

    gs_hit_unique = []
    infers_hit_unique = []
    for hit_s in golden_solution_hit:
        hit_s = [item for item in hit_s if float(item[2])>0.6 and item[0] in item_solution_sent]
        gs_hit_unique.append(len(set([pairs[0] for  pairs in hit_s]))/solution_triple_num)
    for hit_s in infer_solution_hit:
        hit_s = [item for item in hit_s if float(item[2])>0.6 and item[0] in item_solution_sent]
        infers_hit_unique.append(len(set([pairs[0] for pairs in hit_s]))/solution_triple_num)
    # mean_gs_hit_len.append(len(golden_solution_hit[0]))
    # mean_infers_hit_len.append(int(np.array([len(hit_s) for hit_s in infer_solution_hit]).mean()))
    item_scores = []
    for sent in golden_data[idx]['solution_history']:
        scores = scorer.score(
                    references=[golden_data[idx]['final_answer']],
                    candidates=[sent]
                )
        item_scores.append(max(scores))
    infer_solution_bleurt.append(item_scores)
    mean_gs_hit_len.append(gs_hit_unique[0])
    mean_infers_hit_len.append(np.array(infers_hit_unique).mean())
    infers_hit_len.append(infers_hit_unique)
    for solution, bleurt_score, kg_score, hit_s in zip(golden_data[idx]['solution_history'],  item_scores, infers_hit_unique, infer_data[idx]['solution_infer_hit_path']):
        if bleurt_score-kg_score>0.3 and kg_score>0.3:
            print('puzzle:', golden_data[idx]['puzzle'])
            print('golden final answer:', golden_data[idx]['final_answer'])
            print('infer solution:',solution)
            print('bluert score:', bleurt_score)
            print('KG coverage:',kg_score)
            print([item for item in hit_s if float(item[2])>0.6])
            print('=*'*20)
            import ipdb;ipdb.set_trace()
x = []
y = []
for bleurt_scores, kg_scores in zip(infer_question_bleurt, inferq_len):
    x.extend(bleurt_scores)
    y.extend(kg_scores)
# import matplotlib.pyplot as plt
# plt.scatter(x, y)
# plt.xlabel('BLEURT score')
# plt.ylabel('KG score')
# plt.savefig('bleurt-score-question.jpg')
# plt.savefig('bleurt-score-question.svg')
x = []
y = []
for bleurt_scores, kg_scores in zip(infer_solution_bleurt, infers_hit_len):
    x.extend(bleurt_scores)
    y.extend(kg_scores)
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x, y)
plt.xlabel('BLEURT score')
plt.ylabel('KG score')
plt.savefig('bleurt-score-solution.jpg')
plt.savefig('bleurt-score-solution.svg')
# import ipdb;ipdb.set_trace()


print('golden question coverage:', np.mean(mean_gq_len))
print('infer question coverage:', np.mean(mean_inferq_len))
print('golden coverage:', np.mean(mean_gs_hit_len))
print(mean_gs_hit_len)
print('infer coverage:', np.mean(mean_infers_hit_len))

golden_question_dats = []
infer_question_dats = []
golden_sq_dats = []
infer_sq_dats = []
if 'gen_question_list' in infer_data[0]:
    for item in tqdm(infer_data):
        if len(item['question_list']) > 1:
            golden_question_dats.append(dat_score(item['question_list']))
            infer_question_dats.append(dat_score(item['gen_question_list']))
        if len(item['question_list'])>0:
            golden_sq_dats.append(solution_dat(item['final_answer'], item['question_list']))
            infer_sq_dats.append(solution_dat(item['final_answer'], item['gen_question_list']))

    
    print('golden question dat:', np.mean(golden_question_dats))
    print('infer question dat:', np.mean(infer_question_dats))

csvwriter.writerows([
    ['golden_coverage', np.mean(mean_gq_len),np.mean(mean_gs_hit_len)],
    ['infer_coverage', np.mean(mean_inferq_len), np.mean(mean_infers_hit_len)],
    ['question_dat', np.mean(golden_question_dats), np.mean(infer_question_dats)],
    ['solution_q_dat', np.mean(golden_sq_dats), np.mean(infer_sq_dats)]
    ]
)