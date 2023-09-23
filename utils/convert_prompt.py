from collections import defaultdict
from typing import DefaultDict


example_data = {
	"puzzle":"A man walks into a bar and asks the bartender for a drink of water. The bartender pulls out a qun, points it at the man and cocks it. The man pauses before saying \"Thank you\" and leaving. What happened?",
	"final_answer":"The man had the hiccups. The bartender realized this and chose instead to cure the hiccups by frightening the man with the gun.",
	"question_list":[
		"Could the bartender hear him?",
		"Did the man ask for water in an offensive way?"
	],
	"answer_list": [
		"Yes",
		"No"
	],
	"qa_exp":[
		"The bartender understood what the man said and what happened so he can hear the man.",
		"The bartender help the man so the man didn't ask in an offensive way."
	]
}

def convert_qa_with_explanation(input_item, question, keywords_hint=None):
    if keywords_hint is None or len(keywords_hint)==0:
        extra_text = ''
    else:
        extra_text = f"\nKeywords:{', '.join(keywords_hint)}"
    return f'''You are an intelligent bot that can play as judge in situation puzzles with user. A puzzle is given first, and the user begin to ask a "yes/no" question to ensure details, and you should give "Yes/No" as answer to questions. Then Give an explanation for you answer finally.

Puzzle: {input_item['puzzle']}
Solution: {input_item['final_answer']}
Question: {question}{extra_text}
Answer(return json, {{judge, explanation}}):'''

def convert_to_question_generation_prefix(item_dict, keywords=[]):
	task_description = "I am an intelligent bot that can play situation puzzles with user. A puzzle is given first, and the user begin to ask a \"yes/no\" question to ensure details."

	result_prefix = task_description+'\n\n'+"Puzzle: "+example_data['puzzle']+'\n'
	result_prefix += "Question: "+example_data["question_list"][0]+'\n'
	result_prefix += "Answer: "+example_data["answer_list"][0]+'\n'
	result_prefix += "Question: "+example_data["question_list"][1]+'\n'
	result_prefix += '\n'
	result_prefix += "Puzzle:"+item_dict['puzzle']+'\n'
	if "gen_question_list" in item_dict:
		for question, answer in zip(item_dict['gen_question_list'], item_dict['gen_answer_list']):
			result_prefix += "Question: "+question+'\n'
			result_prefix += "Answer: "+answer+'\n'

	if len(keywords)>0:
		result_prefix += "Keywords: "+', '.join(keywords)+'\n'
	result_prefix += "Question: "
	return result_prefix


def convert_to_answer_generation_prefix(item_dict, zero_shot=True, with_explanation=False):
	task_description = "You are an intelligent bot that can play as judge in situation puzzles with user. A puzzle is given first, and the user begin to ask a \"yes/no\" question to ensure details, and you should give \"Yes/No\" as answer to questions."
	# task_description = "I am an intelligent bot that can play as judge in situation puzzles with user. A puzzle is given first, and the user begin to ask a \"yes/no\" question to ensure details, and I will give \"Yes/No/Irrelevent\" as answer to questions."
	# task_description = "I can answer the question with only yes or no or irrelevant with the \"truth\". "
	result_prefix = task_description+'\n\n'
	if not zero_shot:
		result_prefix = result_prefix+"Puzzle: "+example_data['puzzle']+'\n'
		result_prefix += "Solution: "+example_data["final_answer"]
		result_prefix += "Question: "+example_data["question_list"][0]+'\n'
		result_prefix += "Answer: "+example_data["answer_list"][0]+'\n'
		if with_explanation:
			result_prefix += "Explanation: "+example_data['qa_exp'][0]+'\n'
		result_prefix += "Question: "+example_data["question_list"][1]+'\n'
		result_prefix += "Answer: "+example_data["answer_list"][1]+'\n'
		if with_explanation:
			result_prefix += "Explanation: "+example_data['qa_exp'][1]+'\n'
		result_prefix += '\n'
	result_prefix += "Puzzle: "+item_dict['puzzle']+'\n'
	result_prefix += "Solution: "+item_dict['final_answer']+'\n'
	if "gen_question_list" in item_dict and "gen_answer_list" in item_dict:
		for question, answer in zip(item_dict['gen_question_list'], item_dict['gen_answer_list']):
			result_prefix += "Question: "+question+'\n'
			result_prefix += "Answer: "+answer+'\n'

	result_prefix += "Question: "+item_dict['gen_question_list'][-1]+'\n'
	result_prefix += "Answer: "
	return result_prefix


def convert_to_answer_prefix(item_dict, question_list, answer_list):
	task_description = "You are an intelligent bot that can play as judge in situation puzzles with user. A puzzle is given first, and the user begin to ask a \"yes/no\" question to ensure details, and you should give \"Yes/No/Irrelevent\" as answer to questions. If the question can be judged from the given information, give me an \"Yes/No\" as long as you can."
	# task_description = "I can answer the question with only yes or no or irrelevant with the \"truth\". "
	result_prefix = task_description+'\n\n'+"Puzzle: "+example_data['puzzle']+'\n'
	result_prefix += "Solution: "+example_data["final_answer"]
	result_prefix += "Question: "+example_data["question_list"][0]+'\n'
	result_prefix += "Answer: "+example_data["answer_list"][0]+'\n'
	result_prefix += "Question: "+example_data["question_list"][1]+'\n'
	result_prefix += "Answer: "+example_data["answer_list"][1]+'\n'
	result_prefix += '\n'
	result_prefix += "Puzzle:"+item_dict['puzzle']+'\n'
	
	for question, answer in zip(question_list, answer_list):
		result_prefix += "Question: "+question+'\n'
		result_prefix += "Answer: "+answer+'\n'

	result_prefix += "Question: "+question_list[-1]+'\n'
	result_prefix += "Answer: "
	return result_prefix

def convert_to_baseline_prefix(prefix):
	task_description = "I am an intelligent bot that can play lateral thinking puzzles with user. A puzzle is given first,  you are expected to give an reasonable answer creatively."
	result_prefix = task_description+'\n'+"Puzzle: "+example_data['puzzle']+'\n'
	result_prefix += "Solution: "+example_data['final_answer']+'\n\n'
	result_prefix += "Puzzle: "+prefix+'\n'
	result_prefix += "Solution: "
	return result_prefix 

def convert_to_solution_generation_prefix(item_dict, with_hint, golden_history=False, example_qa=True):
	task_description = "I am an intelligent bot that can play situation puzzles with user. A puzzle is given first, and the user begin to ask \"yes/no\" question to ensure details, then I will give the question a\"yes/no/irrelevent\" answer. Finally user try to give solution for the puzzle."
	# task_description = "I can generate a logical answer to the question based on the puzzle, the follow up questions and the follow up answers. "
	reject_hint = "You should ask \"yes/no\" question only."
	hint_dict = {}
	hint_dict['Yes'] = ', correctly think!'
	hint_dict['Yes.'] = ' Correctly think!'
	hint_dict['No'] = ', reverse thinking!'
	hint_dict['No.'] = ' Reverse thinking!'
	hint_dict['Irrelevant'] = ', think differently!'
	hint_dict['Irrelevant.'] = ' Think differently!'

	result_prefix = task_description+'\n\n'+"Puzzle: "+example_data['puzzle']+'\n'
	if example_qa:
		result_prefix += "Question: "+example_data["question_list"][0]+'\n'
		result_prefix += "Answer: "+example_data["answer_list"][0]+'\n'
		result_prefix += "Question: "+example_data["question_list"][1]+'\n'
		result_prefix += "Answer: "+example_data["answer_list"][1]+'\n'
		result_prefix += "Solution: "+example_data["final_answer"]+'\n'
		result_prefix += '\n'
	
	result_prefix += "Puzzle:"+item_dict['puzzle']+'\n'
	if "gen_question_list" in item_dict and "gen_answer_list" in item_dict and not golden_history:
		for question, answer in zip(item_dict['gen_question_list'], item_dict['gen_answer_list']):
			if with_hint:
				if answer in hint_dict:
					result_prefix += "Question: "+question+'\n'
					result_prefix += "Answer: "+answer+hint_dict[answer]+'\n'
				else:
					result_prefix +=f"Tips: {reject_hint}\n"
				
			else:
				if answer in hint_dict:
					result_prefix += "Question: "+question+'\n'
					result_prefix += "Answer: "+answer+'\n'
	else:
		for question, answer in zip(item_dict['question_list'], item_dict['answer_list']):
			if with_hint:
				if answer in hint_dict:
					result_prefix += "Question: "+question+'\n'
					result_prefix += "Answer: "+answer+hint_dict[answer]+'\n'
				else:
					result_prefix +=f"Tips: {reject_hint}\n"
				
			else:
				if answer in hint_dict:
					result_prefix += "Question: "+question+'\n'
					result_prefix += "Answer: "+answer+'\n'
	result_prefix += f"Solution: "
	return result_prefix