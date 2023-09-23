# lateral_thinking
Code for "Exploring and Quantifying Lateral Thinking in LLMs"
## Setup
- Follow instructions in https://github.com/bjascob/amrlib
- python>=3.8
- torch
- openai
- tqdm
- csv
- bleurt
- sentence_transformer
- fuzzywuzzy
- nltk
- spacy
## Dataset
Our datasets concludes 3 parts: English puzzle dataset, KG annotation of English puzzle dataset, and Chinese puzzle dataset. We seperate the puzzles with example QA process in indepandent files: `data/en_withqa.json`, `data/lateral_zh_withoutQA.json`
## Inference
Our scripts support generate solutions for baseline, neural setting and neural-symbolic setting. If you want to use OpenAI API-base models, please replace your api-key as the envirnoment varible first.
Example usage:
<details>
<summary>Baseline</summary>
<pre><code>
    python3 main.py\
        --input_file situation-data/lateral_data.json\
        --max_turn 4 \
        --withoutQA \
        --suffix baseline \
        --model gpt3.5
</code></pre>
</details>
<details>
<summary>Neural Setting</summary>
<pre><code>
    python3 main.py\
        --input_file situation-data/lateral_data.json\
        --max_turn 4 \
        --suffix neural_setting \
        --model gpt3.5
</code></pre>
</details>
<details>
<summary>Neural-Symbolic Setting</summary>
<pre><code>
    python3 main.py\
        --input_file situation-data/lateral_data.json\
        --max_turn 4 \
        --KGQA \
        --suffix neural_setting \
        --model gpt3.5
</code></pre>
</details>

## Evaluation
Our evaluation for lateral thinking includes 3 dimension: the semantic-based/lexical-based metrics, our proposed KG-based metrics and the DAT score. and the detail usage are as following:
- To calculate BLEU scores and BLEURT scores for generated results, run script `evaluation_lmqg.py`.
- The KG-based evaluation needs to run script `evaluation_kg.py` to generate an autometically eventKG annotation for generated files first, then run script `evaluate_graph_score.py`
- DAT score is a metric to evaluate divergent, and you can run `evaluate_dat_score.py` to get the DAT scores.