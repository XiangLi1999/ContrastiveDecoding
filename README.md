# Contrastive Decoding

Contrastive Decoding: Open-ended Text Generation as Optimization


Arxiv Link: https://arxiv.org/abs/2210.15097 
 
-------------
## Setup 
```python
pip install -e transformers 
```
-------------
##  Run contrastive decoding on a specified prompt:  
```python
cd text-generation; 

python run_generation.py --model_name_or_path gpt2-xl --model_type gpt2 --length 256 --prompt "<|endoftext|> A version of Sonic the Hedgehog was developed by Ancient and released in 1991" --student_name_or_path gpt2 --st_coef 1.0   --student_temperature 0.5  --outfile outputs/temp_out.json    --ignore_prefix no
```
--------------

##  Run contrastive decoding on dataset (see submit_decoding.py for detail):  
```python
python run_generation.py --model_name_or_path gpt2-xl --model_type gpt2 --length 256 --prompt_file wikitext --student_name_or_path gpt2 --st_coef 1.0   --student_temperature 0.5  --outfile outputs/temp_out.json    --ignore_prefix no
```

---------------
This code is used for producing all results in the paper. We will release a cleaner version of the code soon;  
