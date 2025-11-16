import pandas as pd
import re

df_original = pd.read_csv('DATA/PredEx/test_original.csv')

oracle_list = []
with open('DATA/PredEx/oracle_selected_cases.txt', 'r+') as file:
	for line in file:
		line = line.strip()
		if not line.endswith('.csv'):
			continue
		line = line.split('.')[0]
		line = line.split('__')[1]
		oracle_list.append(line)

# print(oracle_list)

flags = []
for count, (uid, text) in enumerate(zip(df_original['Case Name'], df_original['Input']), 1):
    words = re.findall(r'[a-zA-Z0-9]+', uid)
    camel_cased = ''.join(word.capitalize() for word in words)
    if camel_cased in oracle_list:
    	flags.append(True)
    else:
    	flags.append(False)
    filename = f'{count-1}__{camel_cased}.csv'

# print(sum(flags))
# print(len(flags))

# df_original['flags'] = flags
new_test = df_original[flags]

new_test.to_csv('./DATA/PredEx/test.csv', index=False)

