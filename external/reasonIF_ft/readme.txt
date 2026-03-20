# To launch the training runs, follow the instructions on https://github.com/huggingface/gpt-oss-recipes/tree/main

# after the environment is set up, upload the train*parquet to your hugging face account and then refer to it in your sft_full.yaml file

# run the following script in your environment
accelerate launch --config_file configs/zero3.yaml sft.py --config configs/sft_full.yaml --attn_implementation kernels-community/vllm-flash-attn3

