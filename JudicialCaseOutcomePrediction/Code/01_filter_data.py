import pandas as pd

def find_largest_prefix_match(s1: str, s2: str, min_length: int = 75):
    max_len = min(len(s1), len(s2))
    for length in range(max_len, min_length - 1, -1):
        prefix = s2[:length]
        index = s1.find(prefix)
        if index != -1:
            return index, length
    return -1, -1

def match_bitstring(s1: str, s2: str, min_length: int = 75):
    if type(s1) != type(''):
        print(str(s1))
        print(len(str(s1)))
        print(s1)
        print(type(s1))
    bitstring = ['0'] * len(s1)
    s2 = s2[5:]
    while len(s2) >= min_length:
        index, length = find_largest_prefix_match(s1, s2, min_length)
        if index == -1:
            break
        for i in range(index, index + length):
            bitstring[i] = '1'
        s2 = s2[length:]
    return ''.join(bitstring), s2

def apply_bitstring_matching(df: pd.DataFrame, min_length: int = 75) -> pd.DataFrame:
    bitstrings = []
    unmatched_outputs = []

    for idx, row in df.iterrows():
        print(f"{idx}. {row['Case Name']}")
        if len(str(row['Input'])) <= 3:
            bitstrings.append('ERROR: EMPTY INPUT')
            unmatched_outputs.append(row['Output'])
        elif len(str(row['Output'])) <= 3:
            bitstrings.append('ERROR: EMPTY OUTPUT')
            unmatched_outputs.append(row['Output'])
        elif len(str(row['Input'])) < len(str(row['Output'])):
            bitstrings.append('ERROR: INPUT SMALLER THAN OUTPUT')
            unmatched_outputs.append(row['Output'])
        else:
            bitstring, unmatched = match_bitstring(row['Input'], row['Output'], min_length)
            bitstrings.append(bitstring)
            unmatched_outputs.append(unmatched)

    df['bitstring'] = bitstrings
    df['unmatched'] = unmatched_outputs
    return df

# Load data
df = pd.read_csv('DATA/PredEx/test.csv')

# Apply matching
df = apply_bitstring_matching(df, min_length=12)

# Save the result
df.to_csv('DATA/PREP/filter_all.csv', index=False)



# import pandas as pd

# def find_largest_prefix_match(s1: str, s2: str, min_length: int = 75):
#     max_len = min(len(s1), len(s2))
#     for length in range(max_len, min_length - 1, -1):  # only consider prefixes ≥ min_length
#         prefix = s2[:length]
#         index = s1.find(prefix)
#         if index != -1:
#             return index, length
#     return -1, -1

# def match_bitstring(s1: str, s2: str, min_length: int = 75):
#     print(s1)
#     print(s2)
#     print()
#     bitstring = ['0'] * len(s1)

#     while len(s2) >= min_length:
#         index, length = find_largest_prefix_match(s1, s2, min_length)
#         if index == -1:
#             break
#         for i in range(index, index + length):
#             bitstring[i] = '1'
#         s2 = s2[length:]  # remove the matched prefix

#     return ''.join(bitstring), s2

# def apply_bitstring_matching(df: pd.DataFrame, min_length: int = 75) -> pd.DataFrame:
#     df[['bitstring', 'unmatched']] = df.apply(
#         lambda row: pd.Series(match_bitstring(row['Input'], row['Output'], min_length)),
#         axis=1
#     )
#     return df

# # Load data
# df = pd.read_csv('DATA/test.csv')

# # Apply matching
# df = apply_bitstring_matching(df, min_length=75)

# # Save the result
# df.to_csv('DATA/filter_all.csv', index=False)



# # import re
# # import pandas as pd


# # # def find_largest_prefix_match(s1: str, s2: str, min_prefix_len: int = 15):
# # #     if len(s2) < min_prefix_len:
# # #         return -1, -1

# # #     # Use regex to find all valid prefix candidates ending at whitespace boundaries
# # #     for match in re.finditer(r'\s+', s2):
# # #         end_idx = match.end()
# # #         if end_idx < min_prefix_len:
# # #             continue
# # #         prefix = s2[:end_idx]
# # #         index = s1.find(prefix)
# # #         if index != -1:
# # #             return index, len(prefix)
    
# # #     # Final check if the entire s2 is valid (if it ends without whitespace)
# # #     if len(s2) >= min_prefix_len:
# # #         index = s1.find(s2)
# # #         if index != -1:
# # #             return index, len(s2)

# # #     return -1, -1


# # def find_largest_prefix_match(s1: str, s2: str):
# #     max_len = min(len(s1), len(s2))
# #     for length in range(max_len, 74, -1):  # only consider prefixes of length ≥ 15
# #         prefix = s2[:length]
# #         index = s1.find(prefix)
# #         if index != -1:
# #             return index, length
# #     return -1, -1

# # def match_bitstring(s1: str, s2: str):
# #     bitstring = ['0'] * len(s1)

# #     while len(s2) >= 75:
# #         index, length = find_largest_prefix_match(s1, s2)
# #         if index == -1:
# #             break
# #         for i in range(index, index + length):
# #             bitstring[i] = '1'
# #         s2 = s2[length:]  # remove the matched prefix

# #     return ''.join(bitstring), s2

# # def apply_bitstring_matching(df: pd.DataFrame) -> pd.DataFrame:
# #     df[['bitstring', 'unmatched']] = df.apply(
# #         lambda row: pd.Series(match_bitstring(row['input'], row['output'])),
# #         axis=1
# #     )
# #     return df


# # df = pd.DataFrame({
# #     'input': ["the quick brown fox jumps over the lazy dog"] * 2,
# #     'output': ["quick brown fox jumps over the moon", "lazy dog sleeps forever"]
# # })

# # updated_df = apply_bitstring_matching(df)
# # print(updated_df[['input', 'output', 'bitstring', 'unmatched']])
