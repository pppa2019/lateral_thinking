import time
import os
from utils.load_kg_ann import load_kg_from_ann
import json
from random import sample
import argparse
from utils.convert_prompt import *
from utils.metrics import *
from get_response import *
import json
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, AutoModel
import importlib

import inspect
import torch


def load_data(data_path):
	with open(data_path, 'r') as f:
		data = json.load(f)
	return data

def standardization_answer(answer):
	answer = answer.lower()
	if 'yes' in answer and 'no' not in answer:
		return 'Yes.'
	if 'no' in answer and 'yes' not in answer:
		return 'No.'
	return 'Irrelevant.'

def abstract_entity_event(span_dict):
	entity_list = []
	event_list = []
	for _, value in span_dict.items():
		if value[0]=='Head_End':
			if len(value)==2:
				entity_list.append(value[1])
			else:
				entity_list.append(' '.join(value[3:]))
		elif value[0]=='Event':
			if len(value)==2:
				entity_list.append(value[1])
			else:
				event_list.append(' '.join(value[3:]))
	return entity_list, event_list

if __name__=="__main__":
	parse = argparse.ArgumentParser()
	parse.add_argument('--max_turn', type=int, default=5)
	parse.add_argument('--example_only_first_turn', action='store_true')
	parse.add_argument('--with_hint', action='store_true')
	parse.add_argument('--threshold', type=float, default=0.4)
	parse.add_argument('--model', type=str, default="Electra")
	parse.add_argument('--without_QA', action='store_true')
	parse.add_argument('--suffix', type=str, default=None)
	parse.add_argument('--KGQA', action='store_true')
	parse.add_argument('--human', action='store_true')
	parse.add_argument('--shuffle', action='store_true')
	parse.add_argument('--golden_QA', action='store_true')
	parse.add_argument('--input_file', type=str)


	args = parse.parse_args()

	

	model_func_dict = {
		"gpt3.5": get_response_from_OpenAI,
		"chatglm2": get_response_from_chatglm,
		"llama": get_response_from_llama,
		"bloom": get_response_from_llama
	}
	# import ipdb;ipdb.set_trace()
	get_response = model_func_dict[args.model]
	if args.model== 'chatglm2' or args.model=='chatglm':
		
		base_model='THUDM/chatglm-6b'
		model_json = os.path.join(base_model, "config.json")
		if os.path.exists(model_json):
			model_json_file = json.load(open(model_json))
			model_name = model_json_file["_name_or_path"]
			tokenizer = AutoTokenizer.from_pretrained(model_name,
														fast_tokenizer=True, trust_remote_code=True)
		model = AutoModel.from_pretrained(
			base_model,
			torch_dtype=torch.float16,
			trust_remote_code=True,
		).cuda()
	elif args.model=='llama':
		base_model= 'meta-llama/Llama-2-7b'
		tokenizer = LlamaTokenizer.from_pretrained(base_model)
		model = AutoModelForCausalLM.from_pretrained(
				base_model,
				torch_dtype=torch.float16,
				device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
				trust_remote_code=True,
			)
	
	else:
		model = None
		tokenizer = None
	if args.KGQA:
		from sentence_transformers import SentenceTransformer
		kgqa_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
	dataset = load_data(args.input_file)
	# check cache and reload
	# import ipdb;ipdb.set_trace()
	output_file = f'experiment/{args.model}-{args.suffix}.json'
	if os.path.exists(output_file):
		result = json.load(open(output_file))
	else:
		result = [ ]
	with_hint = args.with_hint
	path_template = 'situation-data/KG_annotation/{}.ann'
	valid_answer = ["Yes", 'No', 'Irrelevant', "Yes.", 'No.', 'Irrelevant.']
	for index, item in zip(range(len(result), len(dataset)+1), dataset[len(result):]):
	# for index,item in enumerate(result[:len(dataset)]):
		turn_count = 0
		# import ipdb;ipdb.set_trace()
		# turn_count = len(item['gen_question_list'])
		
		item['gen_question_list'] = []
		item['gen_answer_list'] = []
		item['solution_history'] = []
		if args.without_QA:
			prompt = convert_to_baseline_prefix(item['puzzle'])
			generated_solution, gen_raw, _ = get_response(prompt, model, tokenizer)
			item["solution_history"].append(generated_solution)
			result.append(item)
			with open(output_file, 'w+') as f:
				json.dump(result, f)
			continue
		if args.KGQA:
			
			pz_span_dict, _, _ = load_kg_from_ann(path_template.format(f"{index+1}.puzzle"))
			puzzle_entities = list(set(abstract_entity_event(pz_span_dict)[0]))
			print(puzzle_entities)
		while turn_count < args.max_turn:
			keywords = []
			if args.KGQA:
				keywords = sample(puzzle_entities, min(3, len(puzzle_entities)))
				# import ipdb;ipdb.set_trace()
			
			question_generation_prompt = convert_to_question_generation_prefix(item, keywords)
			if args.human:
				print(question_generation_prompt)
				generated_question = input()
				if generated_question=='q':
					break
			else:
				generated_question, gen_raw, _ = get_response(question_generation_prompt, model, tokenizer)
			item['gen_question_list'].append(generated_question)
	

			answer_generation_prompt = convert_to_answer_generation_prefix(item)
			if not args.KGQA and not args.human:
				generated_answer, _, _ = get_response(answer_generation_prompt, model, tokenizer)
				# if generated_answer in valid_answer:
				# TODO: a answer standardization function the same as evaluation part
				item["gen_answer_list"].append(standardization_answer(generated_answer))
			elif args.human:
				print(answer_generation_prompt)
				generated_answer = input()	
				if generated_answer=='q':
					break
				item["gen_answer_list"].append(standardization_answer(generated_answer))
			else:
				
				kg_answer, matched_keywords = get_response_from_KG(
					path_template.format(f'{index+1}.puzzle'),
					path_template.format(f'{index+1}.truth'),
					item["gen_question_list"][-1],
					kgqa_model
					)
				prefix = convert_qa_with_explanation(item, generated_question, matched_keywords)
				_, gen_raw, _ = get_response(prefix, model, tokenizer)
				try:
					generated_answer = json.loads(gen_raw)['judge']
				except:
					generated_answer = "Irrelevant"
				puzzle_entities = list(set(puzzle_entities) - set(matched_keywords))
				print(puzzle_entities)
				item["gen_answer_list"].append(generated_answer)
			
			solution_generation_prompt = convert_to_solution_generation_prefix(item, with_hint)
			generated_solution, _, _ = get_response(solution_generation_prompt, model, tokenizer)
			print(f"turn_{turn_count+1}: ", generated_solution)

			item["solution_history"].append(generated_solution)

			jaccard_score = Jaccard_similarity(generated_solution, item['final_answer'][0])

			if jaccard_score>args.threshold:
				break
			turn_count += 1
		
		result.append(item)
		with open(output_file, 'w+') as f:
			json.dump(result, f)
