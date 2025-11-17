import pandas as pd
import glob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='Process directory for CSV files.')
parser.add_argument('--directory', type=str, default='./DATA/TOKEN_EXPS/', help='Directory containing CSV files')
args = parser.parse_args()


# Step 1: Read and aggregate the CSV files
def aggregate_weights(file_paths, mean=True):
    aggregate_dict = {}
    lengths = []
    weights = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        
        # third_last_row = df.iloc[-3]
        # token = third_last_row['Unnamed: 0']
        # weight = third_last_row['importance']
        # lengths.append(float(token))
        # weights.append(float(weight))
        # print(file_path)
        # print(token)
        # print(weight)
        # print()

        for index, row in df.iloc[:-3].iterrows():
            token = str(row['Unnamed: 0']).lower()
            if not token.isalpha():
                continue
            weight = row['importance']
            if token in aggregate_dict:
                aggregate_dict[token].append(weight)
            else:
                aggregate_dict[token] = [weight]
    
    # avg_resumes_per_occupation = len(file_paths) / 7

    # for token in aggregate_dict:
    #     if len(aggregate_dict[token]) < avg_resumes_per_occupation * 0.5: # word occurs less times than 10% of one occupation = outlier - drop
    #         aggregate_dict[token] = [0]

    # Calculate the mean or sum weight for each token based on the mean flag
    if mean:
        aggregate_dict = {token: sum(weights) / len(weights) for token, weights in aggregate_dict.items()}
    else:
        aggregate_dict = {token: sum(weights) for token, weights in aggregate_dict.items()}

    # # print(f'{max(lengths)=}, {min(lengths)=}')
    # # print(f'{lengths=}')
    # plt.figure(figsize=(10, 6))
    # plt.scatter(lengths, weights, alpha=0.5)
    # plt.title('Scatterplot of Length vs Weight')
    # plt.xlabel('Length')
    # plt.ylabel('Weight')
    # plt.grid(True)
    # plt.savefig(f'{args.directory}scatterplot_length_vs_weight.pdf', dpi=300)
    # plt.close()
    # print(f'{args.directory}scatterplot_length_vs_weight.pdf')

    # return {word: weight for word, weight in aggregate_dict.items() if len(word) > 0}
    
    # words_to_print = ['white', 'black', 'hispanic', 'latino', 'african', 'american', 'asian', 'pacific', 'islander', 'races', 'man', 'woman', 'thousand', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'characters', 'written']
    words_to_print = []
    for word in words_to_print:
        found = False
        for key in aggregate_dict.keys():
            key = str(key)
            if key.lower() == word.lower():
                print(f"Weight for '{word}': {aggregate_dict[key]}")
                found = True
                break
        if not found:
            print(f"'{word}' not found in aggregate_dict")
    return {str(word): weight for word, weight in aggregate_dict.items()}


# Step 2: Create and save the word clouds
def create_and_save_wordclouds(aggregate_dict, pos_output_path, neg_output_path, mean=True):
    strid='SUM'
    if mean:
        strid='MEAN'
    # Separate positive and negative weights for color mapping
    pos_words = {word: weight for word, weight in aggregate_dict.items() if weight > 0}
    neg_words = {word: -weight for word, weight in aggregate_dict.items() if weight < 0}
    print(f'{len(pos_words)=}')
    print(f'{len(neg_words)=}')
    print(f'{sum([pos_words[w] for w in pos_words])=}')
    print(f'{sum([neg_words[w] for w in neg_words])=}')
    
    all_words = {word: (weight if weight > 0 else -weight) for word, weight in aggregate_dict.items()}

    stopwords = STOPWORDS
    # stopwords.add('[CLS]')
    # stopwords.add('[SEP]')
    # stopwords.add('[cls]')
    # stopwords.add('[sep]')
    # stopwords.add('<s>')
    # stopwords.add('</s>')
    large_words = {word[0]: word[1] for word in sorted(all_words.items(), key=lambda item: item[1], reverse=True)[:500] if word[0] not in stopwords and '##' not in word[0] and len(word[0])>3}

    # pos_large_words = {word[0]: word[1] for word in sorted(pos_words.items(), key=lambda item: item[1], reverse=True)[:500] if word[0] not in stopwords and '##' not in word[0] and len(word[0])>3}
    # neg_large_words = {word[0]: word[1] for word in sorted(neg_words.items(), key=lambda item: item[1], reverse=True)[:500] if word[0] not in stopwords and '##' not in word[0] and len(word[0])>3}
    all_pos_large_words = {word: large_words[word] for word in pos_words if word in large_words}
    all_neg_large_words = {word: large_words[word] for word in neg_words if word in large_words}
    words_to_plot = min([len(all_pos_large_words), len(all_neg_large_words)])
    pos_large_words = {word[0]: word[1] for word in sorted(all_pos_large_words.items(), key=lambda item: item[1], reverse=True)[:words_to_plot]}
    neg_large_words = {word[0]: word[1] for word in sorted(all_neg_large_words.items(), key=lambda item: item[1], reverse=True)[:words_to_plot]}
    # pos_large_words = all_pos_large_words
    # neg_large_words = all_neg_large_words


    print(f'{len(pos_large_words)=}')
    print(f'{len(neg_large_words)=}')
    print(f'{sum([pos_large_words[w] for w in pos_large_words])=}')
    print(f'{sum([neg_large_words[w] for w in neg_large_words])=}')
    # print(f'{len(pos_large_words)=}')
    # print(f'{len(neg_large_words)=}')
    # print(f'{pos_large_words=}')

    pos_sum = sum([pos_large_words[k] for k in pos_large_words])
    neg_sum = sum([neg_large_words[k] for k in neg_large_words])
    normalized_words = {w: large_words[w]/pos_sum if w in pos_large_words else large_words[w]/neg_sum for w in large_words}
    # normalized_words = large_words
    # normalized_words = {**pos_large_words, **neg_large_words}
    # print(f'{len(normalized_words)=}')
    # normalized_words = large_words
    # lambda_factor = 1 # inspired from wc default for visualisation
    # normalized_words = {w: normalized_words[w] + lambda_factor * large_words[w] for w in normalized_words}

    # Create word clouds
    wordcloud_pos = WordCloud(color_func=lambda *args, **kwargs: "green", width=800, height=400, random_state=1, background_color='white', relative_scaling=0.5).generate_from_frequencies(pos_large_words)
    wordcloud_neg = WordCloud(color_func=lambda *args, **kwargs: "red", width=800, height=400, random_state=1, background_color='white', relative_scaling=0.5).generate_from_frequencies(neg_large_words)

    wordcloud_combined = WordCloud(stopwords=stopwords, random_state=1, background_color='white', relative_scaling=0.5, width=2800, height=400, color_func=lambda word, *args, **kwargs: 'green' if word in pos_words else 'red').generate_from_frequencies(normalized_words)

    wordcloud_combined.to_file(f'./PLOTS/clouds/all_wordcloud_{strid}.pdf')
    # plt.figure()
    # plt.imshow(wordcloud_combined, interpolation='bilinear')
    # plt.axis('off')
    # plt.show()
    # plt.savefig(f'./wordcloud_{strid}.png', format='png', dpi=300)
    # plt.close()

    # Save word clouds to disk
    wordcloud_pos.to_file(f'{pos_output_path}_{strid}.pdf')
    wordcloud_neg.to_file(f'{neg_output_path}_{strid}.pdf')

# print(args.directory)
file_paths = []
file_paths.extend(glob.glob(f'{args.directory}*.csv'))


for i in range(2):
    for j in range(2):
        for k in range(2):
            file_paths.extend(glob.glob(f'{args.directory}{i}_{j}_{k}/*.csv'))
# file_paths = glob.glob('./DATA/TOKEN_EXPS_200_1024_0.05/0_1_0/*.csv')  # Adjust the path to where your CSV files are located
# print(file_paths)
# aggregate_dict_mean = aggregate_weights(file_paths, mean=True)
aggregate_dict_sum = aggregate_weights(file_paths, mean=False)

pos_output_path = f'./PLOTS/clouds/positive_weights_wordcloud'
neg_output_path = f'./PLOTS/clouds/negative_weights_wordcloud'

# create_and_save_wordclouds(aggregate_dict_mean, pos_output_path, neg_output_path, mean=True)
create_and_save_wordclouds(aggregate_dict_sum, pos_output_path, neg_output_path, mean=False)

# print(pos_output_path)
# print(neg_output_path)


# values = list(aggregate_dict_mean.values())
# plt.figure(figsize=(10, 6))
# sns.histplot(values, kde=True, bins=30, color='blue')
# plt.title('per token importances')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.savefig(f'{args.directory}token_importance_distribution.pdf', dpi=300)
# plt.close()
