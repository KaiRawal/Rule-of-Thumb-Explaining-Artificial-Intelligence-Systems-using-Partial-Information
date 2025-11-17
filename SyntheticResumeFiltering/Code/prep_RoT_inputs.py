from transformers import BertTokenizer, BertModel
from transformers import LongformerTokenizer, LongformerModel
from transformers import BigBirdTokenizer, BigBirdModel
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import numpy as np




def filter_df(infile='./Dataset/classification.csv', outfile='./Dataset/rot_input_sample.csv'):
    df = pd.read_csv(infile)
    # print(f"FINAL { (df['Prediction'] == 'Yes').sum() =}")
    # print(f"FINAL { len((df['Prediction'] == 'Yes')) =}")
    df['index'] = df.index
    filtered_df = df[['index', 'Summary', 'Race', 'Gender', 'Political_orientation', 'Prediction', 'Ground_truth']]
    # filtered_df = filtered_df[['essay_id_comp', 'full_text', 'gender', 'race_ethnicity', 'binarised_gpt_score', 'binarised_holistic_score']]
    filtered_df.columns = ['resume_id', 'summary', 'race', 'gender', 'party', 'llm_prediction', 'human_label']
    filtered_df.to_csv(outfile, index=False)




# tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
# model = AutoModel.from_pretrained('answerdotai/ModernBERT-base')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', force_download=True)
# model = LongformerModel.from_pretrained('allenai/longformer-base-4096', force_download=True)
# tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base', force_download=True)
# model = BigBirdModel.from_pretrained('google/bigbird-roberta-base', force_download=True)

model.eval()




def embed_and_save(input_passage_raw='', output_file_name='empty'):
    input_passage = input_passage_raw.replace('*', '')
    inputs = tokenizer(input_passage, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    if len(tokens) >= 512:
        return None
    # assert len(tokens) < 8192 # ModernBERT length
    # # print(f'Number of tokens embedded: {len(tokens)}')  # Print the length of tokens embedded
    # # print(f'String length: {len(input_passage)}')  # Print the string length
    # print(f'{len(tokens)=} \t {len(input_passage)=}')
    # if len(tokens) == 512:
    #     return len(input_passage)
    # else:
    #     return np.nan
    embeddings_list = embeddings[0].cpu().numpy()
    if np.isnan(embeddings_list).any():
        print(f'NANs in {output_file_name}')
    clean_tokens = [tokenizer.convert_tokens_to_string([t]).strip() for t in tokens]

    df = pd.DataFrame(embeddings_list, index=clean_tokens)
    heads = [f'Dim_{i+1}' for i in range(df.shape[1])]
    df.columns = heads
    
    df.to_csv(f'./DATA/EMBEDDINGS/{output_file_name}.csv', index=True)

    return df.mean().values


def main():
    # all_toks = []
    filter_df()
    input_df = pd.read_csv('./Dataset/rot_input_sample.csv')
    count = 0
    all_data = []
    for index, resume_text, gen, race, party, targ, orig in zip(list(input_df['resume_id']), list(input_df['summary']), list(input_df['gender']), list(input_df['race']), list(input_df['party']), list(input_df['llm_prediction']), list(input_df['human_label'])):
        count += 1
        # race = 0 if race == 'Black/African American' else 1
        gen = 'woman' if gen == 'Female' else 'man'
        targ = 1 if targ == 'Yes' else 0
        embeddings = embed_and_save(f'[ Resume belongs to a {race} {gen} that votes for the {party} party ]\n\n{resume_text}', index)
        if embeddings is None:
            print(f'Skipping {index}')
            continue
        # all_toks.append(embeddings)
        # all_data.append([index])
        # # all_data[-1].extend(embeddings)
        # # all_data[-1].extend([gen, race, targ])
        # # all_data[-1].extend([race])
        # all_data[-1].extend(targ)
        all_data.append([index, targ, orig])
        if count % 20 == 1:
            print(f'embedding generation: iteration {count} of {len(input_df)}')

    # all_df = pd.DataFrame(all_data, columns=['index'] + [f'dim_{i}' for i in range(len(embeddings))] + ['gender', 'race', 'target'])
    all_df = pd.DataFrame(all_data, columns=['index', 'llm_prediction', 'human_label'])
    all_df.to_csv('./DATA/rot_input_embeddings.csv')
    # print(f'{np.nanmin(all_toks)}')
    # print(f'{np.nanmean(all_toks)}')
    # print(f'{np.nanmax(all_toks)}')

if __name__ == '__main__':
    main()