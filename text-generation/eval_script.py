# parse the generated results into a list of text
from dis import disco
import json, sys, os 
import numpy as np 
our_file = sys.argv[1]

rewrite=False  
mauve = True 
coherence = True 
ppl=True 
entity_f1 = True 
disco_coh= False  

# rewrite=False 
# mauve = False 
# coherence = False 
# ppl=True 
# entity_f1 = False 
# disco_coh= False  
 

if rewrite:
    with open(our_file, 'r') as f1,  open(our_file[:-6]+'_gold.jsonl', 'w') as f2:
        examples = [json.loads(l.strip()) for l in f1]
        for x in examples:
            x[0]['gen_text'] = x[0]['gold_ref']
            print(json.dumps(x), file=f2)
    our_file = our_file[:-6]+'_gold.jsonl'

print(our_file)
cumulative_stats = {}
sys.path.insert(0, '/private/home/xlisali/decoding/text-generation/baselines/SimCTG')
def get_2lst_repl(our_file):
    '''/private/home/xlisali/decoding/text-generation/baselines/simctg_contrasive.json'''
    text_list = []
    with open(our_file) as f:
        item_list = json.load(f)
    for item in item_list:
        text = item['generated_result']['0']['continuation']
        text_list.append(text)
    text_list2 = []
    with open(our_file) as f:
        item_list = json.load(f)
    for item in item_list:
        text = item['reference_continuation_text']
        text_list2.append(text)
    return text_list, text_list2 


def load_file_full(our_file):
    text_list = []
    if our_file.endswith('json'):
        with open(our_file) as f:
            item_list = json.load(f)
        for item in item_list:
            text = item['generated_result']['0']['continuation']
        #     text = item['generated_result']['0']['full_text']
            text_list.append(text)
    else: # for jsonl files. 
        with open(our_file, "r") as fin:
            examples = [json.loads(l.strip()) for l in fin]
            for example in examples:
                if isinstance(example, list):
                    example = example[0]
                if 'gen_text' in example:
                    text = example['gen_text']
                    text_list.append(text)
                else:
                    text = example['trunc_gen_text']
                    text_list.append(text)
    return text_list


def load_file_ref_ours(tokenizer, our_file, cont_only=True):
    text_list = []
    text_ref_lst = []
    if our_file.endswith('json'):
        with open(our_file) as f:
            item_list = json.load(f)
        for item in item_list:
            text = item['generated_result']['0']['continuation']
        #     text = item['generated_result']['0']['full_text']
            text_list.append(text)
    else: # for jsonl files. 
        with open(our_file, "r") as fin:
            examples = [json.loads(l.strip()) for l in fin]
            for example in examples:
                if isinstance(example, list):
                    example = example[0]
                if 'gen_text' in example:
                    text = example['gen_text']
                    text_ref = example['gold_ref']
                    if False:
                        encoded_prompt = tokenizer(example['prompt'])['input_ids']
                        encoded_gen = tokenizer(text)['input_ids']
                        encoded_ref = tokenizer(text_ref)['input_ids']
                        # print(len(example['tokens']), len(encoded_prompt), len(encoded_gen), len(encoded_ref))
                        encoded_gen = example['tokens'][len(encoded_prompt):]
                        encoded_ref = encoded_ref[len(encoded_prompt):]
                        text = tokenizer.decode(encoded_gen)
                        text_ref = tokenizer.decode(encoded_ref) 
                    if cont_only:
                        text_prompt = example['prompt']
                        if text_prompt.startswith('</s>'):
                            text_prompt = text_prompt.lstrip('</s>')
                        if text_ref.startswith('</s>'):
                            text_ref = text_ref.lstrip('</s>')
                        if text.startswith('</s>'):
                            text = text.lstrip('</s>')
                        try:
                            # print(text_prompt)
                            # print(text[:len(text_prompt)])
                            # print(text_prompt == text[:len(text_prompt)])
                            # print(text_prompt == text_ref[:len(text_prompt)])
                            # print( text_ref[:len(text_prompt)])
                            assert text_prompt == text[:len(text_prompt)]
                            assert text_prompt == text_ref[:len(text_prompt)]
                        except:
                            continue
                        text = text[len(text_prompt):]
                        text_ref = text_ref[len(text_prompt):]
                    text_list.append(text)
                    text_ref_lst.append(text_ref)
                else:
                    assert False, 'invalid formatting'
    return text_ref_lst, text_list


def load_file_(our_file):
    text_list = []
    if our_file.endswith('json'):
        with open(our_file) as f:
            item_list = json.load(f)
        for item in item_list:
            text = item['generated_result']['0']['continuation']
        #     text = item['generated_result']['0']['full_text']
            text_list.append(text)
    else: # for jsonl files. 
        with open(our_file, "r") as fin:
            examples = [json.loads(l.strip()) for l in fin]
            for example in examples:
                if isinstance(example, list):
                    example = example[0]
                if 'gen_text' in example:
                    text = example['gen_text']
                    text_prefix = example['prompt']
                    # print(text[:len(text_prefix)])
                    # print(text_prefix)
                    # assert text[:len(text_prefix)] == text_prefix 
                    if len(text_prefix) >= len(text):
                        continue 
                    text_list.append(text[len(text_prefix):])
                else:
                    text = example['trunc_gen_text']
                    text_list.append(text)
    return text_list

def load_file_pair(our_file):
    text_list = []
    if our_file.endswith('json'):
        with open(our_file) as f:
            item_list = json.load(f)
        for item in item_list:
            text = item['generated_result']['0']['continuation']
        #     text = item['generated_result']['0']['full_text']
            prefix = item['prefix_text']
            text_list.append((prefix, text))
    else: # for jsonl files. 
        with open(our_file, "r") as fin:
            examples = [json.loads(l.strip()) for l in fin]
            for example in examples:
                if isinstance(example, list):
                    example = example[0]
                if 'gen_text' in example:
                    text = example['gen_text']
                    text_prefix = example['prompt']
                    # assert text[:len(text_prefix)] == text_prefix 
                    if len(text_prefix) >= len(text):
                        continue 
                    text_list.append((text_prefix, text[len(text_prefix):]))
                else:
                    text = example['trunc_gen_text']
                    try:
                        text_prefix = example['prompt']
                    except:
                        text_prefix = example['prefix']
                    text_list.append((text_prefix,text))
    return text_list

def process_text(text_lst):
    for i, text in enumerate(text_lst):
        temp_sent = text.replace(' @', '').replace('@ ', '') # remove space 
        from nltk.tokenize.treebank import TreebankWordDetokenizer
        temp_sent = TreebankWordDetokenizer().detokenize(temp_sent.split())
        # print(temp_sent) 
        text_lst[i] = temp_sent
    return text_lst 

text_list = load_file_(our_file)
# compute the evaluation results
from simctg.evaluation import measure_repetition_and_diversity
rep_2, rep_3, rep_4, diversity = measure_repetition_and_diversity(text_list)
print(diversity)
print ('{} rep-2 is {}, rep-3 is {}, rep-4 is {}, and diversity is {}'.format(our_file, rep_2, rep_3, rep_4, round(diversity,2)))
cumulative_stats['name'] = our_file
cumulative_stats['rep-2'] = rep_2
cumulative_stats['rep-3'] = rep_3
cumulative_stats['rep-4'] = rep_4
cumulative_stats['diversity'] = diversity #round(diversity,2)

'''
   The result of rep-2 is 3.93, rep-3 is 0.78, rep-4 is 0.31, and diversity is 0.95
'''
if mauve:
    # p_text, q_text = get_2lst_repl(our_file)
    # print('Evaluating MAUVE score')

    # p_text = load_file_full(sys.argv[1])
    # q_text = load_file_full(sys.argv[2])

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    p_text, q_text = load_file_ref_ours(tokenizer, our_file, cont_only=True)


    print(len(p_text), len(q_text))
    # tokenize by GPT2 first. 
    tgt_len = 128
    # xx = tokenizer(q_text)['input_ids']
    x = tokenizer(p_text, truncation=True, max_length=tgt_len)['input_ids']
    y = tokenizer(q_text, truncation=True, max_length=tgt_len)['input_ids']
    # xxx = [xx for xx in x if len(xx) == tgt_len]
    # print(len(xxx))
    # y = [xx for xx in y if len(xx) == tgt_len]
    xxyy = [(xx, yy) for (xx, yy) in zip(x, y) if len(xx) == tgt_len and len(yy) == tgt_len]
    x, y = zip(*xxyy)
    # print([len(xx) for xx in x], [len(xx) for xx in y])
    # map back to texts. 
    p_text = tokenizer.batch_decode(x)#[:target_num]
    q_text = tokenizer.batch_decode(y)#[:target_num]
    print(len(p_text), len(q_text))
    
    import mauve 
    ref_list = p_text
    pred_list = q_text 
    # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
    # out = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0, max_text_length=tgt_len, verbose=True) 
    out = mauve.compute_mauve(p_text=ref_list, q_text=pred_list, device_id=0, max_text_length=256, 
        verbose=False, featurize_model_name='gpt2')
    # print(out)
    print(out.mauve) # prints 0.9917, 
    cumulative_stats['mauve'] = out.mauve
    

if coherence:
    print('Evaluating coherence score')

    from simcse import SimCSE
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    sent_lst = load_file_pair(our_file)
    full_sim_lst = []
    pp_lst, yy_lst = zip(*sent_lst)
    pp_lst = list(pp_lst)
    yy_lst = list(yy_lst) 
    print(len(pp_lst), len(yy_lst))
    similarities = model.similarity(pp_lst, yy_lst)
    similarities = np.array(similarities)
    coherence_score = similarities.trace() / len(similarities) 
    cumulative_stats['coherence'] = coherence_score
    print(round(coherence_score, 2))
    # for (pp,yy) in sent_lst:
    #     similarities = model.similarity([pp], [yy])
    #     full_sim_lst.append(similarities[0][0])
    # full_sim_lst = np.array(full_sim_lst)
    # print(full_sim_lst.mean(), full_sim_lst.std())

if entity_f1:
    import stanza 
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # our_file = sys.argv[1]
    print(our_file) 
    p_text, q_text = load_file_ref_ours(tokenizer, our_file, cont_only=True)

    q_text=process_text(q_text) 
    p_text = process_text(p_text)
    print(len(q_text), len(p_text)) 

    overlap = []
    for sent_q, sent_p in zip(q_text, p_text):
        docs_q = nlp(sent_q)
        docs_p = nlp(sent_p)
        p_ent = set([ent.text.lower() for ent in docs_p.ents])
        q_ent = set([ent.text.lower() for ent in docs_q.ents])
        # print(p_ent, q_ent)
        union = p_ent.union(q_ent)
        intersection = p_ent.intersection(q_ent)
        # print(union, intersection)
        if len(union) == 0 or len(q_ent) == 0:
            continue
        if len(p_ent) == 0:
            continue 
        recall = len(intersection) / len(p_ent)
        precision = len(intersection) / len(q_ent)
        if recall + precision != 0:
                f1 = 2 * (recall * precision) / (recall + precision)
        else: 
                f1 = 0 

        overlap.append([recall, precision, f1])
    overlap = np.array(overlap) 
    result_f1 = overlap.mean(axis=0)
    cumulative_stats['recall'] = result_f1[0]
    cumulative_stats['precision'] = result_f1[1]
    cumulative_stats['f1'] = result_f1[2]

if disco_coh:
    from disco_score import DiscoScorer
    import tqdm 
    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    disco_scorer = DiscoScorer(device='cuda:0', model_name='bert-base-uncased')
    p_text, q_text = load_file_ref_ours(tokenizer, our_file, cont_only=True)
    print(len(q_text), len(p_text))
    if True:
        tgt_len = 128
        # xx = tokenizer(q_text)['input_ids']
        x = tokenizer(p_text, truncation=True, max_length=tgt_len)['input_ids']
        y = tokenizer(q_text, truncation=True, max_length=tgt_len)['input_ids']
        # xxx = [xx for xx in x if len(xx) == tgt_len]
        # print(len(xxx))
        # y = [xx for xx in y if len(xx) == tgt_len]
        xxyy = [(xx, yy) for (xx, yy) in zip(x, y) if len(xx) == tgt_len and len(yy) == tgt_len]
        x, y = zip(*xxyy)
        # print([len(xx) for xx in x], [len(xx) for xx in y])
        # map back to texts. 
        p_text = tokenizer.batch_decode(x)#[:target_num]
        q_text = tokenizer.batch_decode(y)#[:target_num]
        
    # q_text = q_text[:100] 
    # p_text = p_text[:100] 

    EntityGraph_lst = []
    LexicalChain_lst = []
    RC_lst = []
    LC_lst = []
    DS_Focus_NN_lst = []
    DS_SENT_NN_lst = []
    for s, refs in tqdm.tqdm(zip(q_text, p_text)):
        try:
            s = s.lower()
            refs = [refs.lower()]
            EntityGraph_lst.append(disco_scorer.EntityGraph(s, refs))
            LexicalChain_lst.append(disco_scorer.LexicalChain(s, refs))
            RC_lst.append(disco_scorer.RC(s, refs))    
            LC_lst.append(disco_scorer.LC(s, refs)) 
            DS_Focus_NN_lst.append(disco_scorer.DS_Focus_NN(s, refs)) # FocusDiff 
            DS_SENT_NN_lst.append(disco_scorer.DS_SENT_NN(s, refs)) # SentGraph
        except:
            continue 
        print('-', end=' ')

    import numpy as np 
    EntityGraph_score = np.array(EntityGraph_lst).mean()
    LexicalChain_score = np.array(LexicalChain_lst).mean()
    RC_score = np.array(RC_lst).mean()
    LC_score = np.array(LC_lst).mean()
    DS_Focus_NN_score = np.array(DS_Focus_NN_lst).mean()
    DS_SENT_NN_score = np.array(DS_SENT_NN_lst).mean()

    cumulative_stats['EntityGraph_score'] = EntityGraph_score
    cumulative_stats['LexicalChain_score'] = LexicalChain_score
    cumulative_stats['RC_score'] = RC_score
    cumulative_stats['LC_score'] = LC_score
    cumulative_stats['DS_Focus_NN_score'] = DS_Focus_NN_score
    cumulative_stats['DS_SENT_NN_score'] = DS_SENT_NN_score

if ppl:
    print('Evaluating perplexity score.')
    from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
    import torch 
    model_name = "EleutherAI/gpt-j-6B"
    # model_name = "gpt2-medium"


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    lm = AutoModelForCausalLM.from_pretrained(model_name).half().cuda()
    # lm.resize_token_embeddings(len(tokenizer))

    bsz_size = 20 
    text_list = load_file_(our_file)
    text_list = process_text(text_list)
    score_lst = []
    for i in range(len(text_list)//bsz_size):
        text_list_i = text_list[i*bsz_size:(i+1) * bsz_size]
        inputs = tokenizer(text_list_i, return_tensors='pt', truncation=True, padding=True, max_length=100)
        with torch.no_grad():
            labels = inputs['input_ids'].cuda() 
            labels[labels== tokenizer.pad_token] = -100 
            out = lm(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), 
                     labels=labels)
            # print(out.loss) 
            score_lst.append(out.loss) 
    score_lst = torch.tensor(score_lst)
    ppl = np.e ** score_lst.mean()
    cumulative_stats['ppl'] = ppl.item()

print(cumulative_stats)
str_head = ''
str_ = ''
for k, v in cumulative_stats.items():
    str_head += f"\t{k}&"
    if isinstance(v, float):
        str_ += f'\t{round(v, 2)}&'
    else:
        str_ += f'\t{v}&'
print(str_head + '\\\\')
print(str_ + '\\\\') 