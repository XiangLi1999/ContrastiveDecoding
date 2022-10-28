import argparse, csv, emoji, random, csv, json 
import numpy as np 

# Mechanical Turk can't handle emojis
def remove_emojis(s):
  from cleantext import remove_emoji
  return remove_emoji(s)
  return ''.join(filter(lambda c: c not in emoji.UNICODE_EMOJI, s))

def gather_files(file_lst):
  file_handle_lst = []
  full_result_lst = []
  for file in file_lst:
    ff = open(file, 'r')
    file_handle_lst.append(ff) 
  for pkgs in zip(*file_handle_lst):
    # print(pkgs, len(pkgs))
    pkgs = [json.loads(x)[0] for x in pkgs]
    temp_dict = {}
    # print(pkgs)
    temp_dict['prompt'] = pkgs[0]['prompt']
    for (x,y) in zip(pkgs, file_lst):
      temp_dict[y] = x['gen_text']
      assert temp_dict['prompt'] == x['prompt'] 
    for k, v in temp_dict.items():
      # temp_dict[k] = remove_emojis(v.strip().replace('<|endoftext|>', '')).replace('@', '')
      temp_dict[k] = remove_emojis(v.strip().replace('<|endoftext|>', '')).replace('@', '').replace('\n', '<br/>')


    full_result_lst.append(temp_dict) 
  return full_result_lst

def subsample(full_lst, init, end):
  random.shuffle(full_lst)
  return full_lst[init:end] #[-300:-200]

def construct_pairs(full_result_lst, ours_field):
  full_pair_lst = []
  for prompt_dict in full_result_lst:
    pp = prompt_dict.pop('prompt')
    ours = prompt_dict.pop(ours_field)
    for (k, v) in prompt_dict.items():
      full_pair_lst.append({'prompt':pp, 'ours':ours, k:v})
  return full_pair_lst 

def construct_ablation_pairs(full_result_lst, ours_field):
  full_pair_lst = []
  for prompt_dict in full_result_lst:
    pp = prompt_dict.pop('prompt')
    ours = prompt_dict.pop(ours_field)
    for (k, v) in prompt_dict.items():
      full_pair_lst.append({'prompt':pp, ours_field:ours, k:v})
  return full_pair_lst 

def get_simple_pair_lst():
  full_lst = []
  dict1 = {'prompt':'New York City Ballet principal dancer Rebecca Krohn will take her final bow with the company this Saturday night. Krohn joined NYCB as an apprentice in', 'nucleus':"New York City Ballet principal dancer Rebecca Krohn will take her final bow with the company this Saturday night. Krohn joined NYCB as an apprentice in 2009. She was trained at the Philadelphia Ballet's dance conservatory and the London Youth Ballet.<br><br>Before joining NYCB, Krohn was a member of San Antonio Ballet's Orchestra and U.S. Army Brass Band. She performed at the first Gala Ball of New York City Ballet in 2010 and has shared the stage with Louise Aderin-Pocock and Margot Atadero. The ballet recently celebrated its 10-year anniversary at The Metropolitan Opera House.<br><br>Admission is free for all performances. Below are a couple videos from shows that Krohn performed as an apprentice.", 'ours': "New York City Ballet principal dancer Rebecca Krohn will take her final bow with the company this Saturday night. Krohn joined NYCB as an apprentice in the fall of 1998 and slowly rose through the ranks, becoming a principal in 2012. Though Krohn is best known for her flawless execution of classic Balanchine leotard ballets, her repertoire is vast, spanning Jerome Robbins to Justin Peck. After dancing Stravinsky Violin Concerto with Amar Ramasar on Saturday, Krohn will return to the NYCB studios on Monday in a new role: ballet master. We had the chance to talk to the thoughtful and eloquent dancer about her time with the company and goals for the future.<br>Was New York City Ballet always your dream company?<br>As soon as I knew I wanted to be a professional dancer, I knew that I wanted to be in New York City Ballet."}
  dict2 = {'prompt':'Erik Wiese, a member of the SpongeBob SquarePants crew, considers \" Sailor Mouth \" to be his favorite episode, mainly due to its', 'ours':"Erik Wiese, a member of the SpongeBob SquarePants crew, considers \" Sailor Mouth \" to be his favorite episode, mainly due to its random and satirical nature, saying \" Sometimes SpongeBob just catches me off @-@ guard. \" Nancy Basile of About.com ranked the episode at number two for her list of the Top 10 SpongeBob SquarePants Episodes. She said \"'Sailor Mouth'just barely missed being in the number one slot. \" Basile praised the episode's plot and called it \" genius because children can relate to the forbidden thrill of using curse words, and adults can laugh at the parody of TV censorship. \" In an interview with Paul Tibbitt, one of the episode's writers, he told that \" Sailor Mouth \" is his second favorite SpongeBob episode. <br>", "nucleus":"Erik Wiese, a member of the SpongeBob SquarePants crew, considers \" Sailor Mouth \" to be his favorite episode, mainly due to its ability to sell tie-in merchandise by no means being a bad episode. However, he ended up seeing an episode that he considered more consistently good than \"moody scruffy\" (SBSP, \"The Stench of the Sea\"), in spite of having a good eye for quality. Despite in extenuation, he stopped \"discovering\" shows this season on his own.<br><br>\" to be his favorite episode, mainly due to its ability to sell tie-in merchandise by no means being a bad episode. However, he ended up seeing an episode that he considered more consistently good than \"moody scruffy\" (SBSP, \"The Stench of the Sea\"), in spite of having a good eye for quality. Despite in extenuation, he stopped \"discovering\" shows this season on his own. In response to Sailor Moon Crystal, Murphy also mentioned in an interview that \" The Great Teacher Onizuka,\" which is similar to the fictional teacher in DokiDoki Universe and the character Bubblegum, based on Hirohiko Araki's character, served as inspiration for his show.<br><br>, Murphy also mentioned in an interview that \",\" which is similar to the fictional teacher in DokiDoki"}
  dict3 = {'prompt':'Off - site, the new brake wheel and fantail were made. The original brake wheel was too rotten to repair, and showed evidence that it', 'nucleus':"Off - site, the new brake wheel and fantail were made. The original brake wheel was too rotten to repair, and showed evidence that it had been abused, such as multiple fissures. A pic here show the crud that gave way in to the frame along the ends of the crank, and thus was patched with brazing thread. A smidge larger than a normal bidirectional coupler is needed, using a 7/16\" joint. Beveled pilot jig to make the coupler as strong as possible. A -@ showed an 'A' gauge master wrench w/ a 12-32. Think you need to source these for some job? This was like a vacation in buying brake parts. 2-3 A-@ helped figure out various brake coupling types, and stuff. An A-@ handheld said, \"For these things, ANY of these are fine.\" Don't forget to send in any questions or comments to ameniotec@gmail.com.<br><br>http://pubs.usenet.com/usenet/ARTICLE1.HTML Translation: \"The magazine's a little leaky. We actually ran into a leak. Somebody figured out how to dump from the gaskets that hold the engine together,\" was along the lines of saying (gasp), \"We might be good now.\" I found it funny when the auto engineer", 'ours':"Off - site, the new brake wheel and fantail were made. The original brake wheel was too rotten to repair, and showed evidence that it had been altered from the original one installed in 1819. The original brake wheel was 6 feet ( 1 . 83 m ) diameter to allow the Common sails to run at their optimum speed. When the mill was modernised in 1832 it was necessary to alter the gear ratios within the mill, as Patent sails run at a slower speed than Common sails. The great spur wheel was increased in diameter and the stone nuts reduced in diameter. The brake wheel was also rebuilt, with a cast iron segment ring fitted in place of the original cogs, resulting in a wheel 7 feet 2 inches ( 2 . 18 m ) in diameter. The opinion of professional millwrights was sought, and it was decided that a scaled @-@ up version of the original brake wheel would be made, but retaining the cast iron teeth segments. The remains of the original brake wheel were retained as an exhibit in the mill. The original iron segments were all broken, so a pattern was made and new segments were cast in heat - treated malleable cast iron. When the brake wheel was completed it was dismantled and transported to the mill ready for reassembly."}
  full_lst = [dict1, dict2, dict3]
  return full_lst 

def write_to_csv(full_pair_lst,output_path, ours_name='ours'): 
  # construct pairs. 
  random.shuffle(full_pair_lst)
  cols = []
  with open(output_path, 'w') as output_file:
    writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
    writer.writerow('story0_prompt,story0_cont0,story0_cont1,story0_note0,story0_note1,story1_prompt,story1_cont0,story1_cont1,story1_note0,story1_note1,story2_prompt,story2_cont0,story2_cont1,story2_note0,story2_note1'.split(','))
    for idx, x in enumerate(full_pair_lst):
      if len(cols) == 3:
        cols = sum(cols, [])
        writer.writerow(cols)
        cols = []
      # print(x.keys())
      pp = x.pop('prompt') 
      ours = x.pop(ours_name)
      their_labels = list(x.keys())[0]
      theirs = x.pop(their_labels) 

      assert ours[:len(pp)] == pp
      assert theirs[:len(pp)] == pp 

      # left = "\x1B[3m" 
      # right = "\x1B[0m"
      left = "<em>" 
      right = "</em>"
      left="<span class=\"prompt\">"
      right = '</span>' 
      # ours = f'<em>{ours[:len(pp)]}</em>{ours[len(pp):]}'
      # theirs = f'<em>{theirs[:len(pp)]}</em>{theirs[len(pp):]}'
      ours = f'{left}{ours[:len(pp)]}{right}{ours[len(pp):]}'
      theirs = f'{left}{theirs[:len(pp)]}{right}{theirs[len(pp):]}'


      rand = np.random.rand(1)
      if rand > 0.5:
        # print('11')
        cols.append([pp, ours, theirs, ours_name, their_labels ])
      else:
        # print('22')
        cols.append([pp, theirs, ours, their_labels, ours_name])
    if len(cols) == 3:
        cols = sum(cols, [])
        writer.writerow(cols)
        cols = []
  pass 




def instantiate(full_pair_lst):
  random.shuffle(full_pair_lst)
  cols = []
  with open(output_path, 'w') as output_file:
    writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
    writer.writerow('story0_prompt,story0_cont0,story0_cont1,story0_note0,story0_note1,story1_prompt,story1_cont0,story1_cont1,story1_note0,story1_note1,story2_prompt,story2_cont0,story2_cont1,story2_note0,story2_note1'.split(','))
    for idx, x in enumerate(full_pair_lst):
      pp = x.pop('prompt') 
      ours = x.pop('ours')
      theirs = list(x.values())[0] 
      cols.append([pp, ours, theirs, 'ours', list(x.keys())])
      assert ours[:len(pp)] == pp
      assert theirs[:len(pp)] == pp 

      left = "\x1B[3m" 
      right = "\x1B[0m"
      # ours = f'<em>{ours[:len(pp)]}</em>{ours[len(pp):]}'
      # theirs = f'<em>{theirs[:len(pp)]}</em>{theirs[len(pp):]}'
      ours = f'{left}{ours[:len(pp)]}{right}{ours[len(pp):]}'
      theirs = f'{left}{theirs[:len(pp)]}{right}{theirs[len(pp):]}'
      print(pp)
      print('+'*100)
      print(ours)
      print('-'*100)
      print(theirs) 
      print('='*100)
      print()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str,
                        help="one-article per line file")
    parser.add_argument("output_path", type=str,
                        help="file to write output CSV to")
    parser.add_argument('-b', type=int, default=25,
                        help='chunk size')
    args = parser.parse_args()
    print(args)

    with open(args.input_path, 'r') as input_file, open(args.output_path, 'w') as output_file:
        writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
        writer.writerow([ 'OUT%d' % i for i in range(args.b)])
        cols = []
        i = 0
        for line in input_file:
            if i > 0 and (i % args.b) == 0:
                writer.writerow(cols)
                cols = []
            cols.append(remove_emojis(line.strip().replace('<|endoftext|>', '')))
            i += 1
        if i > 0 and (i % args.b) == 0:
            writer.writerow(cols)
            cols = []

if __name__ == '__main__':
    import random 
    random.seed(101)
    # random.seed(109)

    # ours_field = 'wp_ours_0.8_xl.jsonl'
    # ours_field = 'outputs_temp/wikitext_temp-0.95_xl.jsonl'
    # ours_field = 'outputs_temp/wikitext_temp-0.95_xl-4e-4-10.jsonl'
    # ours_field = 'outputs_temp/wikitext_temp-0.95_xl-4e-4.jsonl'
    # file_lst = [ours_field, 'outputs_temp/wikitext_sample-0.95_xl.jsonl']

    # ours_field = 'outputs_temp/wikitext_gpt2-0.9_xl-t=s=4e-4-b=5-long512.jsonl'
    # file_lst = [ours_field, 'outputs_temp/wikitext_p-0.95_xl-long-512.jsonl']


    # ours_field = 'outputs_v3_256/wikitext_beamprefix-0.8_xl_256.jsonl'
    # ours_field = 'outputs_v3_256/wikitext_greedy_xl_256_gold.jsonl'
    # ours_field = 'outputs_v3_256/wikitext_gpt2-0.8_xl_256.jsonl'
    # file_lst = [ours_field,  'outputs_v3_256/wikitext_typical-0.95_xl_256.jsonl'] #outputs_v3_256/wikitext_p-0.95_xl_256.jsonl'] #'outputs_v3_256/wikitext_gpt2-0.8_xl_256.jsonl']

    ours_field = 'outputs_ignorePrefix_256/wikitext_gpt2-0.8_xl_256.jsonl'
    file_lst = [ours_field,  'outputs_v3_256/wikitext_p-0.95_xl_256.jsonl']
    output_path = 'ours-gpt2-ignore0.8_nucleus0.95_mturk_100_idx2.csv'

    # ours_field = 'outputs_nop-filters_256/wikitext_gpt2-1.0_xl_256_t=0.0_s=0.0_b=5_t=0.7.jsonl'
    # ours_field = 'outputs_nop-filters_256/wikitext_gpt2-1.0_xl_256_t=0.0001_s=0.0001_b=5_t=1.0.jsonl'
    # ours_field = 'outputs_nop-filters_256/wikitext_gpt2-1.0_xl_256_t=0.0_s=0.0_b=5_t=0.5.jsonl'
    ours_field = 'outputs_nop-filters_256/wikitext_gpt2-1.0_xl_256_t=top0.3_b=5_t=0.5.jsonl'
    file_lst = [ours_field,  'outputs_v3_256/wikitext_p-0.95_xl_256.jsonl']
    output_path = 'ours-gpt2-ignore1.0-t0.5-r0.3_nucleus0.95_mturk_100_idx4.csv'

    ours_field = 'outputs_nop-filters_256/pre-qua'
    file_lst = [ours_field, 'outputs_nop-filters_256/pre-qual-p0.95.jsonl']
    output_path = 'pre-qualification2.csv'


    # ours_field = 'outputs_ignorePrefix_256/wikitext_opt-0.8_opt-6.7b_256.jsonl'
    # file_lst = [ours_field,  'outputs_opt_256/wikitext_p-0.95_opt-6.7b_256.jsonl']
    # output_path = 'ours-opt-ignore0.8_nucleus0.95_mturk_100_idx2.csv'

    ours_field = 'outputs_nop-filters_256/wikitext_opt-1.0_xl_256_t=top0.3_b=5_t=0.5.jsonl'
    file_lst = [ours_field,  'outputs_ignorePrefix_256/wikitext_p-0.95_opt-13b_256.jsonl']
    output_path = 'ours-opt13b-ignore1.0-t0.5-r0.3_nucleus0.95_mturk_100_idx2.csv'

    # -------------------------------------------------
    # wikitext 




    # ours_field = 'outputs_nop-filters_256/wikitext_gpt2-1.0_xl_256_t=top0.1_b=5_t=0.5.jsonl'
    # file_lst = [ours_field,  'outputs_v3_256/wikitext_p-0.95_xl_256.jsonl']
    # output_path = 'mturk/wikitext_gpt2xl_nucleus0.95_idx200.csv'

    # ours_field = 'outputs_nop-filters_256/wikitext_gpt2-1.0_xl_256_t=top0.1_b=5_t=0.5.jsonl'
    # file_lst = [ours_field,  'outputs_v3_256/wikitext_typical-0.95_xl_256.jsonl']
    # output_path = 'mturk/wikitext_gpt2xl_typical0.95_idx300.csv'


    # ours_field = 'outputs_ignorePrefix_ccnews_256/wikitext_opt-125m-t1.0-r0.1-1.0_opt-13b_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/wikitext_p-0.95_opt-13b_256.jsonl']
    # output_path = 'mturk/wikitext_opt13b_nucleus0.95_idx0.csv'

    # ours_field = 'outputs_ignorePrefix_ccnews_256/wikitext_opt-125m-t1.0-r0.1-1.0_opt-13b_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/wikitext_typical-0.95_opt-13b_256.jsonl']
    # output_path = 'mturk/wikitext_opt13b_typical0.95_idx100.csv'

    # ours_field = 'outputs_v3_256/wikitext_beamprefix-0.8_xl_256.jsonl'
    # file_lst = [ours_field,  'outputs_v3_256/wikitext_p-0.95_xl_256.jsonl']
    # output_path = 'mturk/wikitext_beamprefix_nucleus0.95_idx200.csv'

    # ours_field = 'outputs_nop-filters_256/wikitext_opt-1.0_xl_256_t=top0.1_b=5_t=0.5.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_256/wikitext_p-0.95_opt-13b_256.jsonl']
    # output_path = 'mturk/wikitext_opt13b_nucleus0.95_idx0.csv'

    # -------------------------------------------------
    # # opt wikinews (13b) v.s. nucleus --> looks quite fluent to me. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/wikinews_opt-125m-t1.0-r0.1-1.0_opt-13b_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/wikinews_p-0.95_opt-13b_256.jsonl']
    # output_path = 'mturk/wikinews_opt13b_nucleus0.95_idx200.csv'

    # opt wikinews (6b) v.s. nucleus --> looks quite fluent to me. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/wikinews_opt-125m-t1.0-r0.1-1.0_opt-6.7b_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/wikinews_p-0.95_opt-6.7b_256.jsonl']
    # output_path = 'mturk/wikinews_opt6.7b_nucleus0.95_idx0.csv'

    # # opt wikinews (13b) v.s. typical --> looks quite fluent to me. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/wikinews_opt-125m-t1.0-r0.1-1.0_opt-13b_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/wikinews_typical-0.95_opt-13b_256.jsonl']
    # output_path = 'mturk/wikinews_opt13b_typical0.95_idx0.csv'

    # # gpt2 wikinews v.s. nucleus --> looks good to me. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/wikinews_gpt2-1.0-t0.5_gpt2-xl_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/wikinews_p-0.95_gpt2-xl_256.jsonl']
    # output_path = 'mturk/wikinews_gpt2xl_nucleus0.95_idx0.csv'

    # # gpt2 wikinews v.s. typical --> looks good to me. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/wikinews_gpt2-1.0-t0.5_gpt2-xl_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/wikinews_typical-0.95_gpt2-xl_256.jsonl']
    # output_path = 'mturk/wikinews_gpt2xl_typical0.95_idx0.csv'

    # beam prefix v.s. nucleus decoding. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/wikinews_beamprefix-0.8_gpt2-xl_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/wikinews_p-0.95_gpt2-xl_256.jsonl']
    # output_path = 'mturk/wikinews_beamprefix_nucleus0.95_idx0.csv'
    # -------------------------------------------------


    # ------------ book corpus -------------------------------------
    # # opt wikinews (13b) v.s. nucleus --> looks quite fluent to me. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/book_opt-125m-t1.0-r0.1-1.0_opt-13b_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/book_p-0.95_opt-13b_256.jsonl']
    # output_path = 'mturk/book_opt13b_nucleus0.95_idx0.csv'

    # # opt wikinews (13b) v.s. typical --> looks quite fluent to me. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/book_opt-125m-t1.0-r0.1-1.0_opt-13b_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/book_typical-0.95_opt-13b_256.jsonl']
    # output_path = 'mturk/book_opt13b_typical0.95_idx0.csv'

    # # gpt2 book v.s. nucleus --> looks good to me.  *** worst results *** 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/book_gpt2-1.0-t0.5_gpt2-xl_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/book_p-0.95_gpt2-xl_256.jsonl']
    # output_path = 'mturk/book_gpt2xl_nucleus0.95_idx300.csv'

    # important!!! 

    ours_field = 'outputs_ignorePrefix_ccnews_256/book_gpt2-0.8-t1.0_gpt2-xl_256_noprefix.jsonl'
    file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/book_p-0.95_gpt2-xl_256.jsonl']
    output_path = 'mturk/book_gpt2xl_nucleus0.95_idx200.csv'

    # ours_field = 'outputs_ignorePrefix_ccnews_256/book_gpt2-1.0-t1.0_gpt2-xl_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/book_p-0.95_gpt2-xl_256.jsonl']
    # output_path = 'mturk/book_gpt2xl_nucleus0.95_idx500.csv'


    # # gpt2 wikinews v.s. typical --> looks good to me. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/book_gpt2-1.0-t0.5_gpt2-xl_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/book_typical-0.95_gpt2-xl_256.jsonl']
    # output_path = 'mturk/book_gpt2xl_typical0.95_idx0.csv'

    # beam prefix v.s. nucleus decoding. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/book_beamprefix-0.8_gpt2-xl_256.jsonl'
    # file_lst = [ours_field,  'outputs_ignorePrefix_ccnews_256/book_p-0.95_gpt2-xl_256.jsonl']
    # output_path = 'mturk/book_beamprefix_nucleus0.95_idx0.csv'
    # -------------------------------------------------


    # # gpt2 book --> no obvious errors, but sometimes the meta-repetitiveness makes the content less interesting. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/book_gpt2-1.0-t0.5_gpt2-xl_256.jsonl'
    # 'outputs_ignorePrefix_ccnews_256/book_beamprefix-0.8_gpt2-xl_256.jsonl' #
    # file_lst = [ours_field, 'outputs_ignorePrefix_ccnews_256/book_p-0.95_gpt2-xl_256.jsonl'] 
    # output_path = 'ours-opt6b-ignore1.0-t0.5-r0.1_nucleus0.95_mturk_100_idx2.csv'

    # # dialog 
    # ours_field = 'dialog_outputs/blenderbot-1b-400M.jsonl'
    # file_lst = [ours_field, 'dialog_outputs/blenderbot-1b-p0.95.jsonl']#'dialog_outputs/blenderbot-1b-greedy.jsonl'] 
    # output_path = 'ours-opt6b-ignore1.0-t0.5-r0.1_nucleus0.95_mturk_100_idx2.csv'

    # opt book --> mostly fluent, saw an instance that generates sequence of consecutive words without space. 
    # ours_field = 'outputs_ignorePrefix_ccnews_256/book_opt-125m-t1.0-r0.1-1.0_opt-13b_256.jsonl'
    # file_lst = [ours_field, 'outputs_ignorePrefix_ccnews_256/book_p-0.95_opt-13b_256.jsonl'] 
    # output_path = 'ours-opt6b-ignore1.0-t0.5-r0.1_nucleus0.95_mturk_100_idx2.csv'


    # ours_field = 'prequal/pre-qua-ours.jsonl'
    # file_lst = [ours_field, 'prequal/pre-qual-p0.95.jsonl'] 
    # output_path = 'prequal/gpt2.csv'

    # ----------------- ablation with temperature ----------------
    # ours_field = 'outputs_ablation_256/wikitext_gpt2-1.0_xl_256_b=5_t=0.5.jsonl'
    # file_lst = [ours_field, 
    #            'outputs_ablation_256/wikitext_gpt2-1.0_xl_256_b=5_t=0.1.jsonl', 
    #            'outputs_ablation_256/wikitext_gpt2-1.0_xl_256_b=5_t=1.0.jsonl', 
    #            'outputs_ablation_256/wikitext_gpt2-1.0_xl_256_b=5_t=1.5.jsonl']
    # output_path = 'mturk/ablation_temperature_wikitext_gpt2.csv'

    # ----------------- ablation with lambda  ----------------
    # ours_field = 'outputs_ablation_256/wikitext_gpt2-1.0_xl_256_b=5_t=0.5.jsonl'
    # file_lst = [ours_field, 
    #            'outputs_ablation_256/wikitext_gpt2-0.5_xl_256_b=5_t=0.5.jsonl', 
    #            'outputs_ablation_256/wikitext_gpt2-1.2_xl_256_b=5_t=0.5.jsonl', 
    #            'outputs_ablation_256/wikitext_gpt2-0.8_xl_256_b=5_t=0.5.jsonl']
    # output_path = 'mturk/ablation_lambda_wikitext_gpt2.csv'

    # --------------- ablation with sampling v.s. search -------
    # ours_field = 'outputs_ignorePrefix_ccnews_256/wikitext_results/wikitext_gpt2-1.0-t0.5_gpt2-xl_256.jsonl'
    # file_lst = [ours_field, 
    #            'outputs_ablation_256/wikitext_sample_gpt2-1.0-t0.5_gpt2-xl_256.jsonl']
    # output_path = 'mturk/ablation_search_sample_wikitext_gpt2.csv'

    ours_field = 'outputs_ignorePrefix_ccnews_256/wikitext_results/wikitext_opt-125m-t1.0-r0.1-1.0_opt-13b_256.jsonl'
    file_lst = [ours_field, 
               'outputs_ignorePrefix_ccnews_256/wikitext_sample_opt-125m-t1.0-r0.1-1.2_opt-13b_256.jsonl']
    output_path = 'mturk/ablation_search_sample_wikitext_opt.csv'

    # ----------------- inclusion of the prefix -----------------
    
    ours_field = 'outputs_ignorePrefix_ccnews_256/wikitext_results/wikitext_gpt2-1.0-t0.5_gpt2-xl_256.jsonl'
    file_lst = [ours_field, 
               'outputs_ablation_containPrefix_256/wikitext_gpt2-1.0-t0.5_gpt2-xl_256.jsonl']
    output_path = 'mturk/ablation_prefix_inclusion_wikitext_gpt2_200.csv'

    # ours_field = 'outputs_ignorePrefix_ccnews_256/book_gpt2-1.0-t0.5_gpt2-xl_256.jsonl'
    # file_lst = [ours_field, 
    #            'outputs_ablation_containPrefix_256/book_gpt2-1.0-t0.5_gpt2-xl_256.jsonl']
    # output_path = 'mturk/ablation_prefix_inclusion_book_gpt2.csv'





    # file_lst = ['wp_ours_0.8_xl.jsonl', 'wp_sample_p0.95_xl.jsonl', 'wp_sample_k10_xl.jsonl', 'wp_greedy_xl.jsonl']
    full_result_lst = gather_files(file_lst)
    print(len(full_result_lst))
    full_result_lst = subsample(full_result_lst, 200, 300)
    print(len(full_result_lst))
    pair_lst = construct_ablation_pairs(full_result_lst, ours_field)
    write_to_csv(pair_lst,output_path, ours_name=ours_field)
    # pair_lst = construct_pairs(full_result_lst, ours_field)
    # write_to_csv(pair_lst,output_path)
    print(len(pair_lst))
    # pair_lst = get_simple_pair_lst()

    # instantiate(pair_lst) 

    # main()