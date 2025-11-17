
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read and aggregate the CSV files
def aggregate_weights(file_paths):
    aggregate_dict = {}

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            # print(row)
            token = str(row['Unnamed: 0']).lower()
            if len(token) < 5:
                continue
            weight = row['importance']
            if token in aggregate_dict:
                aggregate_dict[token].append(weight)
            else:
                aggregate_dict[token] = [weight]
                
    # return {word: weight for word, weight in aggregate_dict.items() if len(word) > 0}
    
    # return {word: weight for word, weight in aggregate_dict.items() if (len(word) > 3) and (word not in ['[CLS]', '[SEP]'])}
    return {word: weight for word, weight in aggregate_dict.items()}
    

stencil = 'DATA/TOKEN_EXPS/{}.csv'
rimps = pd.read_csv('DATA/meta.csv')

negs = sorted(list(rimps[rimps['rot_prediction'] == 0]['index']))
poss = sorted(list(rimps[rimps['rot_prediction'] == 1]['index']))
# print(f'{len(negs)=}')
# print(f'{len(poss)=}')

tok2list_n = aggregate_weights([stencil.format(fname) for fname in negs])
tok2list_p = aggregate_weights([stencil.format(fname) for fname in poss])

result_dict = {}
for word, weight in tok2list_n.items():
    result_dict[word] = {'negs': weight, 'poss': []}

for word, weight in tok2list_p.items():
    if word in result_dict:
        result_dict[word]['poss'] = weight
    else:
        result_dict[word] = {'negs': [], 'poss': weight}
    
result_list = []
for word in result_dict:
    result_list.append({'token': word, 'pos_weights': result_dict[word]['poss'], 'neg_weights': result_dict[word]['negs']})


result_df = pd.DataFrame(result_list) # , columns=['token', 'pos_weights', 'neg_weights'])

for index, row in result_df.iterrows():
    token = row['token']
    demographics = ['white', 'man', 'woman', 'african', 'republican', 'democratic']
    if str(token) not in ['infrastructure', 'hardware', 'educational', 'document', 'sales', 'financial', 'technology'] + demographics:
        pass
        # continue
    pos_data = row['pos_weights']
    neg_data = row['neg_weights']
    if len(pos_data) + len(neg_data) < 20:
        continue
    # print(pos_data)
    # print(neg_data)

    # plt.hist([pos_data, neg_data], bins=10, stacked=True, color=['green', 'red'], label=['IT Worker', 'Other'], alpha=0.5)

    bins = np.histogram_bin_edges(np.concatenate([pos_data, neg_data]), bins=10)
    plt.hist(pos_data, bins=bins, color='green', alpha=0.5, label='IT Worker')
    plt.hist(neg_data, bins=bins, color='red', alpha=0.5, label='Other')
    # plt.hist([pos_data], bins=10, stacked=True, color='green', label='IT Worker', alpha=0.5)
    # plt.hist([neg_data], bins=10, stacked=True, color='red', label='Other', alpha=0.5)
    
    _mean = (sum(pos_data) + sum(neg_data)) / (len(pos_data) + len(neg_data))
    
    plt.axvline(_mean, color='black', linestyle=':', linewidth=1)
    plt.text(_mean, plt.gca().get_ylim()[1] * 0.85, f'Mean RoT\nImportance: {_mean:.2f}', color='black', ha='center', fontsize=16)
    plt.axvline(0, color='black', linestyle='solid', linewidth=1)
    plt.text(0, plt.gca().get_ylim()[1] * 0.05, 'Neutral\nImportance', color='black', ha='center', fontsize=16)

    plt.title(f'Importances for "{token}" across Resume Summary RoT Explanations')
    plt.xlabel('Token Importance Weight')
    plt.ylabel('Frequency Count')
    plt.legend(title='LLM Prediction', fontsize=14, title_fontsize=14, loc="upper left")
    if token in ['financial', 'infrastructure', 'network', 'professional', 'teaching', 'construction']:
        if token in ['infrastructure', 'network']:
            plt.legend(title='LLM Prediction', fontsize=14, title_fontsize=14, loc="upper left")
        else:
            plt.legend(title='LLM Prediction', fontsize=14, title_fontsize=14, loc="upper right")
        plt.xlabel('Token Importance', fontsize=20)
        plt.ylabel('Frequency Count', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./PLOTS/hists/{token}.pdf', dpi=300)
        print(f'./PLOTS/hists/{token}.pdf')
    plt.savefig(f'./PLOTS/distplots/{token}.png')
    if token == 'teacher':
        plt.title('')
        plt.xlabel('Token Importance', fontsize=20)
        plt.ylabel('Frequency Count', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./PLOTS/dists/{token}.pdf', dpi=300)
    elif token == 'data':
        plt.title('')
        plt.xlabel('Token Importance', fontsize=20)
        plt.ylabel('Frequency Count', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./PLOTS/dists/{token}.pdf', dpi=300)
    else:
        plt.title('')
        plt.xlabel('Token Importance', fontsize=20)
        plt.ylabel('Frequency Count', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./PLOTS/dists/{token}.pdf', dpi=300)
        pass
    plt.close()
    

result_df.to_csv('DATA/tokens.csv', index=False)


