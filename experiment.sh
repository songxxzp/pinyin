MODEL_PATH="trigram_sina_weibo_newscrawl_baike_webtext_wiki"

TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector default --storage-top-k-selector default --final-top-k-selector default --batch-size 64
TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector default --storage-top-k-selector default --final-top-k-selector std --batch-size 64
TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector default --storage-top-k-selector default --final-top-k-selector gpt --batch-size 64

TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector std --storage-top-k-selector default --final-top-k-selector default --batch-size 64
TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector std --storage-top-k-selector default --final-top-k-selector std --batch-size 64
TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector std --storage-top-k-selector default --final-top-k-selector gpt --batch-size 64

TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector default --storage-top-k-selector std --final-top-k-selector default --batch-size 64
TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector default --storage-top-k-selector std --final-top-k-selector std --batch-size 64
TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector default --storage-top-k-selector std --final-top-k-selector gpt --batch-size 64

TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector std --storage-top-k-selector std --final-top-k-selector default --batch-size 64
TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector std --storage-top-k-selector std --final-top-k-selector std --batch-size 64
TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector std --storage-top-k-selector std --final-top-k-selector gpt --batch-size 64

TRANSFORMERS_OFFLINE=1 python main.py --model-path $MODEL_PATH --calculate-top-k-selector gpt --storage-top-k-selector gpt --final-top-k-selector gpt --batch-size 64 --top-k-storage 4 --top-k-calculate 1
