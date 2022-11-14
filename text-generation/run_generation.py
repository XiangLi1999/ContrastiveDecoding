#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging
import sys 
import numpy as np
import torch
from tqdm import tqdm 
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import json 
from collections import Counter 

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    OPTModel,
    OPTForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    # OPTLMHeadModel,
)

class NgramModel(torch.nn.Module):
    def __init__(self, n, vocab_size):
        super().__init__()
        self.n = n 
        self.alpha = 0.1 
        self.vocab_size = vocab_size

        
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        return {
            "input_ids": input_ids,
        }
    
    
    @staticmethod
    def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
        return generated_ngrams

    @staticmethod
    def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - ngram_size
        ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
        return banned_ngrams.get(ngram_idx, [])

    @staticmethod
    def _calc_banned_ngram_tokens(
        ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
    ):
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]

        generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)

        banned_tokens = [
            _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
            for hypo_idx in range(num_hypos)
        ]
        return banned_token

    def forward(self, input_ids, **kwargs):
        # print(input_ids)
        generated_ngrams = self._get_ngrams(self.n, input_ids, len(input_ids))
        base_lst = []
        for hypo_idx in range(len(input_ids)):
            ngram_cands = self._get_generated_ngrams(generated_ngrams[hypo_idx], input_ids[hypo_idx], self.n, input_ids.size(1))
            ngram_count = Counter(ngram_cands)
            # print(ngram_count, 'the number of occurences for different indices. ') 
            k_lst = list(ngram_count.keys())
            v_lst = list(ngram_count.values())
            k_lst = torch.LongTensor(k_lst).cuda()
            v_lst = torch.Tensor(v_lst).cuda()
            base = torch.ones(self.vocab_size).cuda() * self.alpha 
            base.scatter_add_(-1, k_lst, v_lst) 
            base_lst.append(base) 
        base_lst = torch.stack(base_lst, dim=0)
        # print(base_lst.shape)
        # normalize. 
        base_lst = base_lst / base_lst.sum(dim=-1).unsqueeze(-1)
        base_lst = base_lst.log().unsqueeze(1)
        # print(base_lst.shape, base_lst)
        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=base_lst,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None, 
        )

def analysis(model, generated, prompt_len):
    output = model(generated, labels=generated)
    print(output.loss)
    return 

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "opt": (OPTForCausalLM, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
    "gptj": (AutoModelForCausalLM, AutoTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text

def opt_prepare_inputs_for_generation(input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
    if kwargs.get('useprompt', None):
        kwargs['useprompt'] = False
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }

    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    if past:
        input_ids = input_ids[:, -1:]
    # first step, decoder_cached_states are empty
    return {
        "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": use_cache,
    }

# get_len = 31 
def ignore_prefix_opt_prepare_inputs_for_generation(input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
    if past is None:
        input_ids = input_ids[:, -1:]
    else:
        # print(past[0][0].shape) 
        genlen = past[0][0].shape[2] 
        input_ids = input_ids[:, -(genlen + 1):]
    # print(input_ids.shape) 

    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)


    input_ids = input_ids[:, -1:]
    # print(attention_mask.shape, input_ids.shape, 'ignore_prefix') 
    # first step, decoder_cached_states are empty
    return {
        "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": use_cache,
    }

def ignore_prefix_prepare_inputs_for_generation(input_ids, past=None, **kwargs):
            
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    input_ids = input_ids[:, -1].unsqueeze(-1)
    if token_type_ids is not None:
        token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

def our_prepare_inputs_for_generation(input_ids, past=None, **kwargs):
    
    if kwargs.get('useprompt', None):
        kwargs['useprompt'] = False
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        token_type_ids = kwargs.get("token_type_ids", None)
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None
    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def out_file(outfile_path, generation_lst):
    with open(outfile_path, 'w') as f:
        for kk in generation_lst:
            print(json.dumps(kk), file=f) 

    print(f'written to {outfile_path}')
    return 

def format_out(generated_text, prompt, generated_tokens, gold_ref=None):
    output = {
                'ended'      : False,
                'tokens'     : generated_tokens,
                'prompt'     : prompt,
                'gen_text'   : generated_text, 
                'len'        : 0,
                'nll4tok'    : [],
                'ppl4tok'    : [],
                'ppl'        : 0,
                'gold_ref'   : gold_ref, 
            } 
            
    return output 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--student_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--revision",
        default='checkpoint-200000',
        type=str,
    )
    parser.add_argument("--contrastive_decoding", type=str, default="student")
    parser.add_argument("--contrastive_prompt", type=str, default="I love repetitive text! Here is my writing:")
    parser.add_argument("--st_coef", type=float, default=0.5)

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt_file", type=str, default="")
    parser.add_argument("--do_sample", type=str, default="no")
    parser.add_argument("--outfile", type=str, default="outfile.jsonl")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--num_beam", type=int, default=5)

    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--min_prob", type=float, default=0.0)

    parser.add_argument("--student_min_prob", type=float, default=0.0)
    parser.add_argument("--student_temperature", type=float, default=1.0)
    parser.add_argument("--use_cap_student", type=str, default='no')
    parser.add_argument("--ignore_prefix", type=str, default='yes') # IMPORTANT
    parser.add_argument("--use_switch", type=str, default='no')
    

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    return args 

def main(args):
    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        # gpt2_model_class, gpt2_tokenizer_class =  MODEL_CLASSES['gpt2']
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    # if args.contrastive_decoding == 'train_model_reg':
    #     sys.path.append('train/')
    #     from mode_reg import Mode_Reg_Trainer, ModeRegGPT2
    #     gpt2 = model_class.from_pretrained('gpt2-medium')
    #     model = ModeRegGPT2.from_pretrained(args.model_name_or_path, gpt2=gpt2, )
    #     model = model.gpt2
    if args.do_sample == 'contrastive_search_baseline':
        if 'gpt' in  args.model_name_or_path:
            from simctg.simctggpt import SimCTGGPT
            model_name = args.model_name_or_path
            model = SimCTGGPT(model_name)
            model.eval()
            tokenizer = model.tokenizer
            eos_token_id = tokenizer.eos_token_id
        elif 'opt' in args.model_name_or_path:
            from simctg.simctgopt import SimCTGOPT
            model_name = args.model_name_or_path
            print(model_name) 
            model = SimCTGOPT(model_name)
            tokenizer = model.tokenizer
            model.eval()
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id
        else:
            raise NotImplemented
    else:
        print(model_class)
        model = model_class.from_pretrained(args.model_name_or_path)
        if args.model_name_or_path == 'EleutherAI/gpt-j-6B':
            tokenizer_gpt2 = tokenizer_class.from_pretrained('gpt2')
            model.resize_token_embeddings(len(tokenizer_gpt2))
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    if args.fp16:
        model.half()
    model.to(args.device)
    if args.contrastive_decoding == 'student':
        assert args.student_name_or_path is not None
        student_lm = AutoModelForCausalLM.from_pretrained(args.student_name_or_path) 
        if args.fp16:
            student_lm.half()
        student_lm.to(args.device)
        if args.ignore_prefix == 'yes':
            if 'gpt' in args.model_name_or_path:
                student_lm.prepare_inputs_for_generation = ignore_prefix_prepare_inputs_for_generation
            elif 'opt' in args.model_name_or_path:
                student_lm.prepare_inputs_for_generation = ignore_prefix_opt_prepare_inputs_for_generation
    elif args.contrastive_decoding == 'ngram':
        print('using student_lm')
        student_lm = NgramModel(3, len(tokenizer))
    elif args.contrastive_decoding == 'earlystop':
        assert args.student_name_or_path is not None and args.revision is not None 
        print(f'loading from earlystop checkpoints of mistral: {args.student_name_or_path},  {args.revision}')
        student_lm = model_class.from_pretrained(args.student_name_or_path, revision=args.revision) 
        student_lm.to(args.device)
    elif args.contrastive_decoding == 'prompt':
        assert args.contrastive_prompt is not None
        student_lm = model 
        contrastive_ids = tokenizer(args.contrastive_prompt, return_tensors='pt')
        print(contrastive_ids['input_ids'])
        contrastive_out = student_lm(contrastive_ids['input_ids'].to(model.device))
        print(contrastive_out.logits, len(contrastive_out.past_key_values))
        student_lm_past = contrastive_out.past_key_values
        student_lm_past = [[l1.expand(5, -1, -1, -1) for l1 in l2] for l2 in student_lm_past]
        model.prepare_inputs_for_generation = our_prepare_inputs_for_generation
    elif args.contrastive_decoding == 'beam_prefix':
        if 'gpt' in args.model_name_or_path:
            sys.path.append('/private/home/xlisali/decoding/PrefixTuning/gpt2')
            from train_control import PrefixTuning
            config = AutoConfig.from_pretrained(args.student_name_or_path)
            print('loading from PrefixTuning.', args.student_name_or_path, )
            gpt2 = AutoModelForCausalLM.from_pretrained('gpt2-medium').cuda()
            prefix_model = PrefixTuning.from_pretrained(
                    args.student_name_or_path,
                    config=config,
                    model_gpt2=model, optim_prefix=True, preseqlen=10,
                    use_infix=False
                ).cuda()
            prompt = prefix_model.get_prompt(None, gpt2=gpt2, bsz=1) #src, control_code=None, gpt2=None, bsz=None, attn_mask=None
            prompt = [x.expand(-1, 5 , -1, -1, -1) for x in prompt]
            if args.ignore_prefix == 'no':
                model.prepare_inputs_for_generation = our_prepare_inputs_for_generation #TODAY DEBUG
        elif 'opt' in args.model_name_or_path:
            sys.path.append('/private/home/xlisali/decoding/text-generation/train')
            from PrefixTuning import PrefixTuning
            config = AutoConfig.from_pretrained(args.student_name_or_path)
            print('loading from PrefixTuning.', args.student_name_or_path, )
            gpt2 = AutoModelForCausalLM.from_pretrained('facebook/opt-350m').cuda()
            prefix_model = PrefixTuning.from_pretrained(
                    args.student_name_or_path,
                    config=config,
                    model_gpt2=model, optim_prefix=True, preseqlen=10,
                    use_infix=False
                ).cuda()
            prompt = prefix_model.get_prompt(None, gpt2=gpt2, bsz=1) #src, control_code=None, gpt2=None, bsz=None, attn_mask=None
            prompt = [x.expand(-1, 5 , -1, -1, -1) for x in prompt]
            model.prepare_inputs_for_generation = opt_prepare_inputs_for_generation # TODAY DEBUG
    elif args.contrastive_decoding == 'train_contra_reg':
        assert args.student_name_or_path is not None
        student_lm = model_class.from_pretrained(args.student_name_or_path) 
        student_lm.to(args.device)
    else:
        student_lm = None 

    if args.do_sample != 'contrastive_search_baseline':
        args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    if not args.prompt_file:
        prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
        prompt_lst = [prompt_text]
        ref_lst = [(0, None)] 
    elif args.prompt_file == 'wikitext' or args.prompt_file == 'cc_news':
        # load wikitext. 
        from datasets import load_dataset, concatenate_datasets
        if args.prompt_file == 'wikitext':
            datasets_val = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
            datasets_test = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
            datasets = concatenate_datasets([datasets_val, datasets_test])
            # datasets['train'] = datasets['train'][:100]
            print(datasets)
        elif args.prompt_file == 'cc_news':
            datasets = load_dataset('cc_news', split='train[0:5000]')
            print(datasets)
            print(datasets.column_names)
        column_names = datasets.column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            # print('tokenize_func', len(examples[text_column_name]))
            # print(examples[text_column_name][:1])
            examples[text_column_name] = [x.replace(' <newline>', '\n') for x in examples[text_column_name]]
            examples[text_column_name] = [tokenizer.bos_token + x for x in examples[text_column_name] if len(x) > 0]

            result_dict = tokenizer(examples[text_column_name], add_special_tokens=False) 
            # use the first 50 words as the prompt, and generate next 128 words. 
            # input_ids_lst = [x[:50] for x in result_dict['input_ids'] if len(x) >= 150 ]
            # gold_lst = [x for x in result_dict['input_ids'] if len(x) >= 150 ]
            input_ids_lst = [x[:32] for x in result_dict['input_ids'] if len(x) >= 160 ]
            gold_lst = [x for x in result_dict['input_ids'] if len(x) >= 160 ]
            result_dict2 = {'input_ids':input_ids_lst, 'gold':gold_lst}
            return result_dict2

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
            load_from_cache_file=True,
        )
        print(tokenized_datasets)
        if args.prompt_file == 'wikitext':
            prompt_ids = tokenized_datasets[:2000]['input_ids'] 
            ref_lst = tokenized_datasets[:2000]['gold'] 
            # prompt_ids = tokenized_datasets['test'][:2000]['input_ids'] 
            # ref_lst = tokenized_datasets['test'][:2000]['gold'] 
            ref_lst = tokenizer.batch_decode(ref_lst)
            ref_lst = [(0, x) for x in ref_lst]
        elif args.prompt_file == 'cc_news':
            prompt_ids = tokenized_datasets[:2000]['input_ids'] 
            ref_lst = tokenized_datasets[:2000]['gold'] 
            ref_lst = tokenizer.batch_decode(ref_lst)
            ref_lst = [(0, x) for x in ref_lst]

        prompt_lst = tokenizer.batch_decode(prompt_ids)
        print(len(prompt_lst), prompt_lst[:20])

        if args.do_sample == 'gold':
            fout = open(args.outfile, 'w')
            prompt_cont = tokenized_datasets['test'][:2000]['gold'] 
            prompt_cont_lst = tokenizer.batch_decode(prompt_cont)
            for (xx, yy) in zip(prompt_lst, prompt_cont_lst):
                print(json.dumps({"gen_text":yy, 'prompt':xx}), file=fout) 
            fout.close() 

    else:
        def load_prompts_simple(dataset_path):
            ref_lst = []
            prefix_lst = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    j = json.loads(line.strip())
                    if 'trunc_gen_text' in j:
                        trunc_gen_text = j['trunc_gen_text']
                    else:
                        trunc_gen_text = j['text'] 
                    trunc_gen_text = trunc_gen_text.encode("ascii", "ignore")
                    trunc_gen_text = trunc_gen_text.decode()
                    trunc_gen_text_ref_ids = tokenizer(trunc_gen_text)['input_ids']
                    if len(trunc_gen_text_ref_ids) < 150:
                         continue
                    ref_lst.append((trunc_gen_text_ref_ids, trunc_gen_text))
                    prefix_ids = trunc_gen_text_ref_ids[:32]
                    prefix_words = tokenizer.batch_decode([prefix_ids])[0]
                    prefix_lst.append((prefix_ids, prefix_words))
            return prefix_lst, ref_lst

        def load_prompts(dataset_path, batch_size, device, bs=False):
            """ Loads data from a jsonl file with "tokens" attribute """
            dataset, count, tokens, ends, last_len = [], 0, [], [], None
            with open(dataset_path, encoding='utf_8') as f:
                for line in tqdm(f):
                    j = json.loads(line.strip())
                    cur_len = len(j['tokens'])
                    # beam search batches must only contain contexts of the same length
                    if not bs:
                        tokens.append(j['tokens'])
                        end = cur_len-1
                        ends.append(end)
                        count += 1
                        if count == batch_size:
                            max_len = max(ends)
                            data = torch.zeros(batch_size, max_len+1).long()
                            for b, (toks, end) in enumerate(zip(tokens, ends)):
                                data[b, :end+1] = torch.Tensor(toks)
                            data = data.to(device)
                            dataset.append((data, ends))
                            tokens, ends = [], []
                            count = 0
                    else:
                        if last_len is None:
                            last_len = cur_len
                        elif last_len != cur_len  or count == batch_size:
                            data = torch.zeros(count, last_len).long()
                            for b, (toks, end) in enumerate(zip(tokens, ends)):
                                data[b, :last_len] = torch.Tensor(toks)
                            data = data.to(device)
                            dataset.append((data, ends))
                            tokens, ends = [], []
                            count = 0
                            last_len = cur_len
                        tokens.append(j['tokens'])
                        ends.append(cur_len-1)
                        count += 1
            if bs and len(tokens) > 0:
                data = torch.zeros(count, last_len).long()
                for b, (toks, end) in enumerate(zip(tokens, ends)):
                    data[b, :last_len] = torch.Tensor(toks)
                data = data.to(device)
                dataset.append((data, ends))

            return dataset
            
        # load prompts 
        prefix_lst, ref_lst = load_prompts_simple(args.prompt_file)
        prompt_lst = [x[1] for x in prefix_lst]
        print('loaded prompts', len(prompt_lst)) 
        print(prompt_lst[:2])
        prompt_lst = prompt_lst[:2000] 
        print('loaded prompts', len(prompt_lst)) 
        # prompt_json_lst = load_prompts(args.prompt_file, 1, model.device)
        # print(len(prompt_json_lst))
        # prompt_lst = []
        # for x in prompt_json_lst:
        #     print(x) 
        #     tokens, length = x 
        #     print(tokens)
        #     x_sent = tokenizer.batch_decode(tokens)
        #     print(x_sent)
        #     prompt_lst.append(x_sent[0])

    generation_lst = []
    
    # LISA 
    for iidx, prompt_text in enumerate(prompt_lst[:2000]):
        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        print(len(encoded_prompt[0]), input_ids.shape) 
        if args.do_sample == 'no' and (args.contrastive_decoding == 'student' or args.contrastive_decoding == 'earlystop' or args.contrastive_decoding == 'ngram') :
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=args.length + len(encoded_prompt[0]),
                min_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                min_prob=args.min_prob,
                repetition_penalty=args.repetition_penalty,
                do_sample=False,
                num_beams=args.num_beam,
                num_return_sequences=args.num_return_sequences,
                student_lm=student_lm,
                teacher_student=True,
                model_kwargs_student={}, 
                st_coef=args.st_coef,
                tokenizer=tokenizer, # analysis
                student_min_prob=args.student_min_prob,
                student_temperature=args.student_temperature,
                use_cap_student=(args.use_cap_student=='yes'), #cap student debug
                use_switch=(args.use_switch == 'yes')
            )
            print('student=smaller model')

        elif args.do_sample == 'no' and args.contrastive_decoding == 'prompt':
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=args.length + len(encoded_prompt[0]),
                min_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                min_prob=args.min_prob,
                repetition_penalty=args.repetition_penalty,
                do_sample=False,
                num_beams=args.num_beam,
                num_return_sequences=args.num_return_sequences,
                student_lm=student_lm,
                teacher_student=True,
                model_kwargs_student={"past":student_lm_past, "useprompt":True}, 
                st_coef=args.st_coef,
                student_min_prob=args.student_min_prob,

            )
            print('student=prompt')

        elif args.do_sample == 'no' and args.contrastive_decoding == 'beam_prefix':
            assert prompt is not None
            print(prompt[0].shape, prompt[0].device)  
            # src = torch.LongTensor(src).to(model.device).unsqueeze(0)
            # print(input_ids)
            # prompt = prefix_model.get_prompt(src, None, gpt2=gpt2, bsz=1) #src, control_code=None, gpt2=None, bsz=None, attn_mask=None
            # prompt = [x.expand(-1, args.num_return_sequences , -1, -1, -1) for x in prompt]
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=args.length + len(encoded_prompt[0]),
                min_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                min_prob=args.min_prob,
                repetition_penalty=args.repetition_penalty,
                do_sample=False,
                num_beams=args.num_beam,
                num_return_sequences=args.num_return_sequences,
                student_lm=gpt2,
                teacher_student=True,
                model_kwargs_student={"past":prompt, "useprompt":True}, 
                st_coef=args.st_coef,
                student_min_prob=args.student_min_prob,
            )
            print('student=beam_prefix')

        elif args.contrastive_decoding == 'train_contra_reg':
            print('beam+train_contra')
            output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            min_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=False,
            num_beams=5,
            num_return_sequences=args.num_return_sequences,
            student_lm=student_lm,
            teacher_student=True,
            model_kwargs_student={}, 
            st_coef=args.st_coef,
            train_contra_reg=True,
            )
            print('beam, contra_reg')
        elif args.do_sample=='greedy' and args.contrastive_decoding == 'none':
            output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            min_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=False,
            num_beams=1,
            num_return_sequences=args.num_return_sequences,
            student_lm=student_lm,
            teacher_student=False,
            model_kwargs_student={}, 
            st_coef=args.st_coef)
            print('greedy')
        elif args.do_sample=='beam':
            output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            min_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=False,
            num_beams=args.num_beam,
            num_return_sequences=args.num_return_sequences,
            teacher_student=False,
            model_kwargs_student={},)
            print('beam search')

        elif args.do_sample == 'typical':
            output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            min_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            typical_p=0.95,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_beams=1,
            num_return_sequences=args.num_return_sequences,
            student_lm=student_lm,
            teacher_student=True,
            model_kwargs_student={}, 
            st_coef=args.st_coef)
            print('typical sampling')
        elif args.do_sample == 'contrastive_search_baseline':
            if 'gpt2' in args.model_name_or_path:
                prefix_text = prompt_text
                tokens = tokenizer.tokenize(prefix_text)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.LongTensor(input_ids).view(1,-1).to(model.model.device)
                beam_width, alpha, decoding_len = 4, 0.6, 256
            else:
                prefix_text = prompt_text
                tokens = tokenizer.tokenize(prefix_text)
                input_ids = tokenizer.convert_tokens_to_ids(tokens) # adds </s> to the beginning of every prompt
                input_ids = torch.LongTensor(input_ids).view(1,-1).to(model.model.device)
                beam_width, alpha, decoding_len = 5, 0.6, 256
            print(model.model.device, input_ids.device)
            # print(input_ids)
            output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, 
                                                alpha=alpha, decoding_len=decoding_len,
                                                end_of_sequence_token_id = eos_token_id, early_stop = False) 
            print("Output:\n" + 100 * '-')
            print(tokenizer.decode(output))
            print("" + 100 * '-')
            output_sequences = torch.tensor([output]).to(input_ids.device) 
            print('contrastive search baseline')
        elif args.do_sample == 'yes' and args.contrastive_decoding == 'student':
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=args.length + len(encoded_prompt[0]),
                min_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                min_prob=args.min_prob,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_beams=1,
                num_return_sequences=args.num_return_sequences,
                student_lm=student_lm,
                teacher_student=True,
                model_kwargs_student={}, 
                st_coef=args.st_coef,
                tokenizer=tokenizer, # analysis
                student_min_prob=args.student_min_prob,
                student_temperature=args.student_temperature,
                use_cap_student=(args.use_cap_student=='yes'), #cap student debug
                use_switch=(args.use_switch == 'yes')
            )
            print('contrastive sampling: student=smaller model')

        elif args.do_sample == 'greedy' and args.contrastive_decoding == 'student':
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=args.length + len(encoded_prompt[0]),
                min_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                min_prob=args.min_prob,
                repetition_penalty=args.repetition_penalty,
                do_sample=False,
                num_beams=1,
                num_return_sequences=args.num_return_sequences,
                student_lm=student_lm,
                teacher_student=True,
                model_kwargs_student={}, 
                st_coef=args.st_coef,
                tokenizer=tokenizer, # analysis
                student_min_prob=args.student_min_prob,
                student_temperature=args.student_temperature,
                use_cap_student=(args.use_cap_student=='yes'), #cap student debug
                use_switch=(args.use_switch == 'yes')
            )
            print('contrastive greedy search: student=smaller model')

        else:
            output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            min_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_beams=1,
            num_return_sequences=args.num_return_sequences,
            student_lm=student_lm,
            teacher_student=True,
            model_kwargs_student={}, 
            st_coef=args.st_coef)
            print('sampling')

        print('analysis', output_sequences.shape, 'output.shape', input_ids.shape, 'input_ids.shape')
        # analysis(model, output_sequences, -1)
        # print(output_sequences)
        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            # print(tokenizer.batch_decode(generated_sequence))
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_dict = format_out(total_sequence, prompt_text, generated_sequence, gold_ref=ref_lst[iidx][1])
            generated_sequences.append(generated_dict)
            print(total_sequence)
            # print(generated_dict)


        generation_lst.append(generated_sequences)

    out_file(args.outfile, generation_lst)
    return generation_lst
    
    
if __name__ == "__main__":
    args = get_args() 
    # with torch.cuda.amp.autocast():
    main(args) 

    import submitit, copy 
    jobs = []
    log_folder = "log_test/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    
    # executor.update_parameters(timeout_min=1440, slurm_partition="devlab", gpus_per_node=1, 
    #                             cpus_per_task=10, constraint='volta32gb')
    executor.update_parameters(timeout_min=1440, slurm_partition="devlab", gpus_per_node=1, 
                                cpus_per_task=10, constraint='volta32gb')

    with executor.batch():
        job = executor.submit(main, args) 
        jobs.append(job)

    
