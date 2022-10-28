import os 
from itertools import product 
command_lst = []
# task_lst = ['wp']
task_lst = ['wikitext'] #['wikitext', 'wp']
# task_lst = ['owt'] #['cc_news']#['owt'] #['wikitext', 'wp']

model_lst = ['xl']
folder = 'outputs_ablation_256'
length = 256 

# teacher_min_prob_lst = [0.0001, ]
# student_min_prob_lst = [0.0001, ]
teacher_min_prob_lst = [0.0, ]
student_min_prob_lst = [0.0, ]
num_beam_lst = [2, 10, 15, 20, 30]

st_coef_lst = [1.0]
student_temperature_lst = [1.0] #[0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
# notes = ' --k 5'
notes = ' --fp16 '
for (task, teacher, st_coef, teacher_min_prob, student_min_prob, num_beam, student_temperature) in product(task_lst, model_lst, st_coef_lst, teacher_min_prob_lst, student_min_prob_lst, num_beam_lst, student_temperature_lst):
    if task =='wp': task2 = 'dataset/wp/wp_valid_ref_encoded200.jsonl'
    elif task =='wp2': task2 = 'dataset/wp/wp_valid_ref_encoded200.jsonl'
    elif task =='owt': task2 = 'dataset/webtext/webtext.valid..proc.jsonl'
    elif task =='cc_news': task2 = task
    elif task == 'wikitext': task2 = task
    # student=gpt2 v.s. teacher
    COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
              f"--outfile {folder}/{task}_gpt2-{st_coef}_{teacher}_{length}_b={num_beam}_t={student_temperature}.jsonl  --num_beam {num_beam} --student_temperature {student_temperature}  {notes}"
    command_lst.append(COMMAND)

    # # student=beamprefix
    # if task == 'owt':
    #     COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
    #           f"--outfile {folder}/{task}_beamprefix-{st_coef}_{teacher}_{length}_t={teacher_min_prob}_s={student_min_prob}_b={num_beam}.jsonl  --min_prob {teacher_min_prob} --student_min_prob {student_min_prob} --num_beam {num_beam} " \
    #           f" --student_name_or_path /private/home/xlisali/decoding/PrefixTuning/gpt2/save_owt_models/owt_prefixtune_y_10_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 " \
    #           f" --contrastive_decoding beam_prefix  {notes}"
    #     command_lst.append(COMMAND)
    #     print(COMMAND)
    # else:
    
    #     COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
    #             f"--outfile {folder}/{task}_beamprefix-{st_coef}_{teacher}_{length}_t={teacher_min_prob}_s={student_min_prob}_b={num_beam}.jsonl  --min_prob {teacher_min_prob} --student_min_prob {student_min_prob} --num_beam {num_beam} "\
    #             f" --student_name_or_path /private/home/xlisali/decoding/PrefixTuning/gpt2/save_e2e_models_convcheck/data2textprefixtune_y_10_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 " \
    #             f" --contrastive_decoding beam_prefix  {notes}"
    #     command_lst.append(COMMAND)
    #     print(COMMAND)



    # early stopping
#     COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length {length} --prompt_file {task2} "\
#               f"--student_name_or_path stanford-crfm/caprica-gpt2-small-x81 --st_coef {st_coef}  " \
#               f"--outfile {folder}/{task}_early200k-{st_coef}_{teacher}_{length}.jsonl --contrastive_decoding earlystop  {notes}"
#     command_lst.append(COMMAND)

#     COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length {length} --prompt_file {task2} "\
#               f"--student_name_or_path stanford-crfm/caprica-gpt2-small-x81 --st_coef {st_coef}  " \
#               f"--outfile {folder}/{task}_ngram-{st_coef}_{teacher}_{length}.jsonl --contrastive_decoding ngram  {notes}"
#     command_lst.append(COMMAND)


# ------------------------------------- other ----------------------------- 
# for (task, teacher) in product(task_lst, model_lst):
#     if task =='wp': task2 = 'dataset/wp/wp_valid_ref_encoded200.jsonl'
#     elif task == 'wikitext': task2 = task
#     # student=gpt2 v.s. teacher
#     COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef 1.0  " \
#               f"--outfile {folder}/{task}_gpt2-1.0_{teacher}.jsonl"
#     command_lst.append(COMMAND)

    # # student=beamprefix
    # COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef 1.0  " \
    #           f"--outfile {folder}/{task}_beamprefix-1.0_{teacher}.jsonl  --student_name_or_path /private/home/xlisali/decoding/PrefixTuning/gpt2/save_e2e_models_convcheck/data2textprefixtune_y_10_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 " \
    #           f" --contrastive_decoding beam_prefix"
    # command_lst.append(COMMAND)

    # early stopping
    # COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length 100 --prompt_file {task2} "\
    #           f"--student_name_or_path stanford-crfm/caprica-gpt2-small-x81 --st_coef {st_coef}  " \
    #           f"--outfile {folder}/{task}_early200k-{st_coef}_{teacher}.jsonl --contrastive_decoding earlystop"
    # command_lst.append(COMMAND)

    # # greedy 
    # COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
    #           f"--outfile {folder}/{task}_p-0.95_{teacher}.jsonl  --do_sample yes --p 0.95 "
    # command_lst.append(COMMAND)

    # COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
    #           f"--outfile {folder}/{task}_k-50_{teacher}.jsonl  --do_sample yes --k 50 --p 1.0 "
    # command_lst.append(COMMAND)

    # COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
    #         f"--outfile {folder}/{task}_k-10_{teacher}.jsonl  --do_sample yes --k 10 --p 1.0 "
    # command_lst.append(COMMAND)

    # COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
    #           f"--outfile {folder}/{task}_greedy_{teacher}.jsonl  --do_sample greedy "
    # command_lst.append(COMMAND)


# ------------------------------------- finetune ----------------------------- 

# task = 'wikitext'
# task2 = task
# st_coef = 1.0 
# model_path_ft = 'train/models/ft_e=5_b=40_m=gpt2_large_wiki' 
# teacher_name = 'ft-large'

# COMMAND = f"python run_generation.py --model_name_or_path {model_path_ft} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
#           f"--outfile {folder}/{task}_gpt2-{st_coef}_{teacher_name}.jsonl"
# command_lst.append(COMMAND)

# # student=beamprefix
# COMMAND = f"python run_generation.py --model_name_or_path {model_path_ft} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
#             f"--outfile {folder}/{task}_beamprefix-{st_coef}_{teacher_name}.jsonl  --student_name_or_path /private/home/xlisali/decoding/PrefixTuning/gpt2/save_e2e_models_convcheck/data2textprefixtune_y_10_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 " \
#             f" --contrastive_decoding beam_prefix"
# command_lst.append(COMMAND)

# COMMAND = f"python run_generation.py --model_name_or_path {model_path_ft} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
#           f"--outfile {folder}/{task}_p-0.95_{teacher_name}.jsonl  --do_sample yes --p 0.95 "
# command_lst.append(COMMAND)

# COMMAND = f"python run_generation.py --model_name_or_path {model_path_ft} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
#           f"--outfile {folder}/{task}_k-50_{teacher_name}.jsonl  --do_sample yes --k 50 --p 1.0 "
# command_lst.append(COMMAND)

# COMMAND = f"python run_generation.py --model_name_or_path {model_path_ft} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
#         f"--outfile {folder}/{task}_k-10_{teacher_name}.jsonl  --do_sample yes --k 10 --p 1.0 "
# command_lst.append(COMMAND)

# COMMAND = f"python run_generation.py --model_name_or_path {model_path_ft} --model_type gpt2 --length 100 --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
#           f"--outfile {folder}/{task}_greedy_{teacher_name}.jsonl  --do_sample greedy "
# command_lst.append(COMMAND)

# 


# # ---------------------- OPT ----------------------------
# command_lst=[]
# task_lst = ['wikitext'] #['wikitext', 'wp']
# # task_lst = ['owt'] #['cc_news']#['owt'] #['wikitext', 'wp']

# # model_lst = ['EleutherAI/gpt-j-6B']
# model_lst = ['facebook/opt-1.3b']#['facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-1.3b']
# folder = 'outputs_opt_256'
# length = 256 
# teacher_type = 'opt'

# st_coef_lst = [0.8, 1.0 ]
# # notes = ' --k 5'
# notes = ' --fp16 '
# for (task, teacher, st_coef) in product(task_lst, model_lst, st_coef_lst):
#     if task =='wp': task2 = 'dataset/wp/wp_valid_ref_encoded200.jsonl'
#     elif task =='wp2': task2 = 'dataset/wp/wp_valid_ref_encoded200.jsonl'
#     elif task =='owt': task2 = 'dataset/webtext/webtext.valid..proc.jsonl'
#     elif task =='cc_news': task2 = task
#     elif task == 'wikitext': task2 = task

#     teacher_name = os.path.basename(teacher) 

#     # student=gpt2 v.s. teacher
#     COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} " \
#               f"--student_name_or_path facebook/opt-125m --st_coef {st_coef}  " \
#               f"--outfile {folder}/{task}_opt-{st_coef}_{teacher_name}_{length}.jsonl  {notes}"
#     command_lst.append(COMMAND)

#     if False: 
#         COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
#                 f"--outfile {folder}/{task}_beamprefix-{st_coef}_{teacher_name}_{length}.jsonl  --student_name_or_path /private/home/xlisali/decoding/PrefixTuning/gpt2/save_e2e_models_convcheck/data2textprefixtune_y_10_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 " \
#                 f" --contrastive_decoding beam_prefix  {notes}"
#         command_lst.append(COMMAND)
#         print(COMMAND)

# #     COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} "\
# #               f"--student_name_or_path stanford-crfm/caprica-gpt2-small-x81 --st_coef {st_coef}  " \
# #               f"--outfile {folder}/{task}_ngram-{st_coef}_{teacher_name}_{length}.jsonl --contrastive_decoding ngram  {notes}"
# #     command_lst.append(COMMAND)

#     # typical decoding 
#     if st_coef == 1.0:
#         COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
#                 f"--outfile {folder}/{task}_typical-0.95_{teacher_name}_{length}.jsonl  --do_sample typical --p 1.0 "
#         command_lst.append(COMMAND)

#          #greedy 
#         COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
#                 f"--outfile {folder}/{task}_p-0.95_{teacher_name}_{length}.jsonl  --do_sample yes --p 0.95 "
#         command_lst.append(COMMAND)

#         COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
#                 f"--outfile {folder}/{task}_k-50_{teacher_name}_{length}.jsonl  --do_sample yes --k 50 --p 1.0 "
#         command_lst.append(COMMAND)

#         COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
#                 f"--outfile {folder}/{task}_k-10_{teacher_name}_{length}.jsonl  --do_sample yes --k 10 --p 1.0 "
#         command_lst.append(COMMAND)

#         COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
#                 f"--outfile {folder}/{task}_greedy_{teacher_name}_{length}.jsonl  --do_sample greedy "
#         command_lst.append(COMMAND)



for x in command_lst:
    print(x) 
    os.system(x) 
print(len(command_lst))

