

export OPENAI_API_KEY= # your api-key here

python3 main.py\
    --input_file situation-data/lateral_data.json\
    --max_turn 4 \
    --suffix neural_setting \
    --model gpt3.5