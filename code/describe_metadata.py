
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
  
def describe_metadata_bydata(data):
    df = pd.DataFrame(
        np.zeros((4,3),dtype=int),
        index = ['wrong_word','lack_word','adjacent_swap','original'],
        columns = ['medical','video','ecom']
    )
    label2row = {'wrong_word':0, 'lack_word':1,
                 'adjacent_swap':3, 'original':4}
    cate2col  = {'medical':0,'video':1, 'ecom':2}
    # print(f"初始df:\n{df}")
    for item in data:
        metadata = item['metadata']
        label = metadata['label']
        cate = metadata['category']
        row = label2row[label]
        col = cate2col[cate]
        df[cate][label] += 1
    print(f"数据集长度：{len(data)} 统计结果如下\n{df}")
    if 'rollouts' in data[0].keys():
        easy_count = [1 if item['rollouts']['difficulty']=='easy' else 0 for item in data]
        hard_count = [1 if item['rollouts']['difficulty']=='hard' else 0 for item in data]
        same_count = [1 if item['rollouts']['difficulty']=='same' else 0 for item in data]
        print(f"数据集长度为{len(data)} same:{sum(same_count)} hard:{sum(hard_count)} easy:{sum(easy_count)}")

    parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda",type=int,default=1)
    parser.add_argument("--file",type=str,default="./json/qwen15base/ecom_random_2000_filtered.json")
    #v parser.add_argument("--output_file" ,type=str,default="aresults/GRPO_class_train.txt")
    args = parser.parse_args()
    # detail_eval_byfile(file = args.file,output_file = args.output_file)
    describe_metadata_byfile_difficulty(file = args.file)