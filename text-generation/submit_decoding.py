import os 
from itertools import product 
command_lst = []
# task_lst = ['wp']
task_lst = ['wikitext', 'book', 'wikinews'] #['wikinews'] #['book'] #['wikitext'] #['wikinews'] #['book'] #['cc_news'] #['wikitext'] #['cc_news'] #['wikitext'] #['wikitext', 'wp']
# task_lst = ['owt'] #['cc_news']#['owt'] #['wikitext', 'wp']

# model_lst = ['xl']
model_lst = ['gpt2-xl'] #['gpt2-medium', 'gpt2']

# folder = 'outputs_ignorePrefix_ccnews_256'
# folder = 'outputs_ablation_containPrefix_256'
folder = 'outputs_ablation_256'


length = 256

st_coef_lst = [0.8, 1.0] #[0.8, 1.0, 1.5]
# student_model_name_lst = ['gpt2', 'gpt2-xl']

student_model_name_lst = ['gpt2'] #['gpt2-medium', 'gpt2-large', 'gpt2', 'gpt2-xl']
# st_coef_lst = []
# notes = ' --k 5'
# notes = ' --use_cap_student yes '
notes = '  --ignore_prefix yes '
student_temperature = 0.5 #1.0
for (task, teacher, st_coef, student_model_name) in product(task_lst, model_lst, st_coef_lst, student_model_name_lst):
    if task =='wp': task2 = 'dataset/wp/wp_test_ref_encoded400.jsonl' #'dataset/wp/wp_valid_ref_encoded200.jsonl'
    elif task =='wp2': task2 = 'dataset/wp/wp_valid_ref_encoded200.jsonl'
    elif task =='owt': task2 = 'dataset/webtext/webtext.valid..proc.jsonl'
    elif task == 'book': task2= '/private/home/xlisali/decoding/text-generation/bookcorpus/books1/ref_200.jsonl' 
    elif task == 'wikinews': task2 = '/private/home/xlisali/decoding/webscrape/wikinews_text.jsonl'
    elif task =='cc_news': task2 = task
    elif task == 'wikitext': task2 = task
#     student=gpt2 v.s. teacher

    if False: # contrastive search. 
        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} " \
                f"--prompt_file {task2} --student_name_or_path {student_model_name} --st_coef {st_coef}   --student_temperature {student_temperature}  " \
                f"--outfile {folder}/{task}_{student_model_name}-{st_coef}-t{student_temperature}_{teacher}_{length}.jsonl  {notes}"
        command_lst.append(COMMAND)
        print(COMMAND)

    if False: # contrastive sampling. 
        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} " \
                f"--prompt_file {task2} --student_name_or_path {student_model_name} --st_coef {st_coef} --do_sample yes   --student_temperature {student_temperature}  " \
                f"--outfile {folder}/{task}_sample_{student_model_name}-{st_coef}-t{student_temperature}_{teacher}_{length}.jsonl  {notes}"
        command_lst.append(COMMAND)
        print(COMMAND)

    if True: # contrastive greedy. 
        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} " \
                f"--prompt_file {task2} --student_name_or_path {student_model_name} --st_coef {st_coef}   --student_temperature {student_temperature}  " \
                f"--outfile {folder}/{task}_greedy_{student_model_name}-{st_coef}-t{student_temperature}_{teacher}_{length}.jsonl  --do_sample greedy --contrastive_decoding student  {notes}"
        command_lst.append(COMMAND)
        print(COMMAND)

    if False:
        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} " \
                f"--prompt_file {task2} --contrastive_decoding ngram --st_coef {st_coef}   --student_temperature {student_temperature}  " \
                f"--outfile {folder}/{task}_ngram-{st_coef}-t{student_temperature}_{teacher}_{length}.jsonl  {notes}"
        command_lst.append(COMMAND)
        print(COMMAND)

#     student=beamprefix
    if False:
        # if task == 'owt':
        #         COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
        #         f"--outfile {folder}/{task}_beamprefix-{st_coef}_{teacher}_{length}.jsonl  --student_name_or_path /private/home/xlisali/decoding/PrefixTuning/gpt2/save_owt_models/owt_prefixtune_y_10_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 " \
        #         f" --contrastive_decoding beam_prefix  {notes}"
        #         command_lst.append(COMMAND)
        #         print(COMMAND)
        # elif task == 'cc_news':
        #         temp_path = '/private/home/xlisali/decoding/PrefixTuning/gpt2/save_ccnews_models/ccnews_prefixtune_y_10_act_cat_b=10-e=2_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1'
        #         COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
        #         f"--outfile {folder}/{task}_beamprefix-{st_coef}_{teacher}_{length}_ccnews.jsonl  --student_name_or_path {temp_path} " \
        #         f" --contrastive_decoding beam_prefix  {notes}"
        #         command_lst.append(COMMAND)
        if True:
        
                COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
                        f"--outfile {folder}/{task}_beamprefix-{st_coef}_{teacher}_{length}.jsonl  --student_name_or_path /private/home/xlisali/decoding/PrefixTuning/gpt2/save_e2e_models_convcheck/data2textprefixtune_y_10_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 " \
                        f" --contrastive_decoding beam_prefix  {notes}"
                command_lst.append(COMMAND)
                print(COMMAND)



    # early stopping
#     COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length {length} --prompt_file {task2} "\
#               f"--student_name_or_path stanford-crfm/caprica-gpt2-small-x81 --st_coef {st_coef}  " \
#               f"--outfile {folder}/{task}_early200k-{st_coef}_{teacher}_{length}.jsonl --contrastive_decoding earlystop  {notes}"
#     command_lst.append(COMMAND)

#     COMMAND = f"python run_generation.py --model_name_or_path gpt2-{teacher} --model_type gpt2 --length {length} --prompt_file {task2} "\
#               f"--student_name_or_path stanford-crfm/caprica-gpt2-small-x81 --st_coef {st_coef}  " \
#               f"--outfile {folder}/{task}_ngram-{st_coef}_{teacher}_{length}.jsonl --contrastive_decoding ngram  {notes}"
#     command_lst.append(COMMAND)

    # typical decding 
    if False and st_coef == 1.0:
        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
                f"--outfile {folder}/{task}_typical-0.95_{teacher}_{length}.jsonl  --do_sample typical "
        command_lst.append(COMMAND)

         #greedy 
        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
                f"--outfile {folder}/{task}_p-0.95_{teacher}_{length}.jsonl  --do_sample yes --p 0.95 "
        command_lst.append(COMMAND)

        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
                f"--outfile {folder}/{task}_k-50_{teacher}_{length}.jsonl  --do_sample yes --k 50 --p 1.0 "
        command_lst.append(COMMAND)

        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
                f"--outfile {folder}/{task}_k-10_{teacher}_{length}.jsonl  --do_sample yes --k 10 --p 1.0 "
        command_lst.append(COMMAND)

        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
                f"--outfile {folder}/{task}_greedy_{teacher}_{length}.jsonl  --do_sample greedy "
        command_lst.append(COMMAND)

        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type gpt2 --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
                f"--outfile {folder}/{task}_contrastive_{teacher}_{length}.jsonl  --do_sample contrastive_search_baseline "
        command_lst.append(COMMAND)




for x in command_lst:
        print(x) 
        print('lol')
        os.system(x) 
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

task = 'wikitext'
task2 = task
st_coef = 1.0 
model_path_ft = 'train/models/ft_e=5_b=40_m=gpt2_large_wiki' 
teacher_name = 'ft-large'

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


# ---------------------- OPT ----------------------------
if False: 
        command_lst=[]
        task_lst = ['wikitext', 'book', 'wikinews'] #['wikitext', 'book', 'wikinews'] #['wikinews']#['book'] #['cc_news'] #['wikitext'] #['wikitext', 'wp']
        # task_lst = ['owt'] #['cc_news']#['owt'] #['wikitext', 'wp']

        # model_lst = ['EleutherAI/gpt-j-6B']
        # model_lst =  ['facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-125m','facebook/opt-350m' ]

        model_lst =  ['facebook/opt-6.7b', 'facebook/opt-13b'] #['facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b']
        # folder = 'outputs_opt_256'
        # folder = 'outputs_ignorePrefix_256'
        # folder = 'outputs_opt_tune'
        folder = 'outputs_ignorePrefix_ccnews_256'
        # folder = 'outputs_ablation_256'

        length = 256 
        teacher_type = 'opt'

        student_temperature=1.0
        # task_lst = []

        # st_coef_lst = [1.0, 1.5, 0.8]
        st_coef_lst = [1.0, 0.5, 0.8, 1.2]
        student_lm_lst = ['opt-125m']
        # student_lm_lst = ['opt-6.7b']
        # student_lm_lst = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b']

        # notes = ' --k 5'
        notes = ' --fp16 '
        for (task, teacher, st_coef, student_lm) in product(task_lst, model_lst, st_coef_lst, student_lm_lst):
                if task =='wp': task2 = 'dataset/wp/wp_test_ref_encoded400.jsonl'
                elif task =='wp2': task2 = 'dataset/wp/wp_valid_ref_encoded200.jsonl'
                elif task =='owt': task2 = 'dataset/webtext/webtext.valid..proc.jsonl'
                elif task == 'book': task2= '/private/home/xlisali/decoding/text-generation/bookcorpus/books1/ref_200.jsonl' 
                elif task == 'wikinews': task2 = '/private/home/xlisali/decoding/webscrape/wikinews_text.jsonl'
                elif task =='cc_news': task2 = task
                elif task == 'wikitext': task2 = task

                teacher_name = os.path.basename(teacher) 

                # student=gpt2 v.s. teacher
                if False:
                        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} " \
                                f"--student_name_or_path facebook/{student_lm} --st_coef {st_coef} --student_temperature {student_temperature} " \
                                f"--outfile {folder}/{task}_{student_lm}-t{student_temperature}-r0.1-{st_coef}_{teacher_name}_{length}.jsonl  {notes}"
                        command_lst.append(COMMAND)
        
                if True: # contrastive sampling. 
                        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} " \
                                f"--student_name_or_path facebook/{student_lm} --st_coef {st_coef} --do_sample yes   --student_temperature {student_temperature} " \
                                f"--outfile {folder}/{task}_sample_{student_lm}-t{student_temperature}-r0.1-{st_coef}_{teacher_name}_{length}.jsonl  {notes}"
                        command_lst.append(COMMAND)
                        print(COMMAND)

                if False: #ngram LM 
                        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} " \
                                f"--student_name_or_path facebook/{student_lm} --st_coef {st_coef} --contrastive_decoding ngram " \
                                f"--outfile {folder}/{task}_ngram-r0.1-{st_coef}_{teacher_name}_{length}.jsonl  {notes}"
                        command_lst.append(COMMAND)

                if False: 
                        prefix_path = '/private/home/xlisali/decoding/text-generation/train/models/prefix_e=5_b=10_m=opt-250m_beam'
                        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
                                f"--outfile {folder}/{task}_beamprefix-{st_coef}_{teacher_name}_{length}.jsonl  --student_name_or_path {prefix_path} " \
                                f" --contrastive_decoding beam_prefix  {notes}"
                        command_lst.append(COMMAND)
                        print(COMMAND)

                if False: 
                        prefix_path = '/private/home/xlisali/decoding/text-generation/train/models/ft_e=5_b=10_m=opt-250m_beam'

                        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} " \
                                f"--student_name_or_path {prefix_path} --st_coef {st_coef}  " \
                                f"--outfile {folder}/{task}_beamft-{st_coef}_{teacher_name}_{length}.jsonl  {notes}"
                        command_lst.append(COMMAND)
                        print(COMMAND)

                #     COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} "\
                #               f"--student_name_or_path stanford-crfm/caprica-gpt2-small-x81 --st_coef {st_coef}  " \
                #               f"--outfile {folder}/{task}_ngram-{st_coef}_{teacher_name}_{length}.jsonl --contrastive_decoding ngram  {notes}"
                #     command_lst.append(COMMAND)

                # typical decoding 
                if False and st_coef == 1.0:
                        # COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef {st_coef}  " \
                        #         f"--outfile {folder}/{task}_typical-0.95_{teacher_name}_{length}.jsonl  --do_sample typical --p 1.0 {notes} "
                        # command_lst.append(COMMAND)

                        # #greedy 
                        # COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
                        #         f"--outfile {folder}/{task}_p-0.95_{teacher_name}_{length}.jsonl  --do_sample yes --p 0.95 {notes} "
                        # command_lst.append(COMMAND)

                        # COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
                        #         f"--outfile {folder}/{task}_k-50_{teacher_name}_{length}.jsonl  --do_sample yes --k 50 --p 1.0 {notes}"
                        # command_lst.append(COMMAND)

                        # COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
                        #         f"--outfile {folder}/{task}_k-10_{teacher_name}_{length}.jsonl  --do_sample yes --k 10 --p 1.0 {notes}"
                        # command_lst.append(COMMAND)

                        # COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
                        #         f"--outfile {folder}/{task}_greedy_{teacher_name}_{length}.jsonl  --do_sample greedy {notes}"
                        # command_lst.append(COMMAND)

                        COMMAND = f"python run_generation.py --model_name_or_path {teacher} --model_type {teacher_type} --length {length} --prompt_file {task2} --student_name_or_path gpt2 --st_coef 0.8  " \
                                f"--outfile {folder}/{task}_contrastive_{teacher_name}_{length}.jsonl  --do_sample contrastive_search_baseline {notes} "
                        command_lst.append(COMMAND)



        for x in command_lst:
                print('-'*100)
                print(x) 
                os.system(x) 

