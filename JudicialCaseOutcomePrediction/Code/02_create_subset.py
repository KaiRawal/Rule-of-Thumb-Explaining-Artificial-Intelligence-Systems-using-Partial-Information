import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("DATA/PREP/filter_all.csv")

print(f"total data: {len(df)=}")

# Define a helper to check if bitstring is valid (only contains 0 or 1)
def is_valid_bitstring(s):
    return isinstance(s, str) and set(s).issubset({'0', '1'})

# Filter valid bitstrings
valid_mask = df['bitstring'].apply(is_valid_bitstring)
valid_df = df[valid_mask]

# Count valid values
num_valid = valid_mask.sum()
print(f"Number of valid bitstrings: {num_valid}")

# Extract non-valid values and get unique counts
non_valid_values = df.loc[~valid_mask, 'bitstring']
non_valid_counts = non_valid_values.value_counts()
print("\nUnique counts for non-valid values:")
print(non_valid_counts)

# Calculate fraction of valid bitstrings that are all 0 or all 1
all_zeros_mask = valid_df['bitstring'].apply(lambda x: set(x) == {'0'})
all_ones_mask = valid_df['bitstring'].apply(lambda x: set(x) == {'1'})

fraction_all_zeros = all_zeros_mask.mean()
fraction_all_ones = all_ones_mask.mean()

print(f"\nFraction of valid bitstrings that are all 0s: {fraction_all_zeros:.4f}")
print(f"Fraction of valid bitstrings that are all 1s: {fraction_all_ones:.4f}")

# Compute histogram of bitstring lengths (10 bins)
lengths = valid_df['bitstring'].apply(len)
length_counts, length_bins = np.histogram(lengths, bins=10)
print("\nHistogram of bitstring lengths (10 bins):")
for count, edge_left, edge_right in zip(length_counts, length_bins[:-1], length_bins[1:]):
    print(f"[{edge_left:.1f}, {edge_right:.1f}): {count}")

# Compute fraction of 1s in each valid bitstring
def one_fraction(s):
    return s.count('1') / len(s) if len(s) > 0 else 0

valid_df['one_fraction'] = valid_df['bitstring'].apply(one_fraction)

# Histogram of 1s fraction (10 bins from 0 to 1)
frac_counts, frac_bins = np.histogram(valid_df['one_fraction'], bins=10, range=(0, 1))
print("\nHistogram of 1s fraction in bitstrings (10 bins):")
for count, edge_left, edge_right in zip(frac_counts, frac_bins[:-1], frac_bins[1:]):
    print(f"[{edge_left:.2f}, {edge_right:.2f}): {count}")

# Compute histogram of string lengths in 'unmatched' column
unmatched_lengths = df['unmatched'].dropna().astype(str).apply(len)
if not unmatched_lengths.empty:
    unmatched_counts, unmatched_bins = np.histogram(unmatched_lengths, bins=10)
    print("\nHistogram of string lengths in 'unmatched' column (10 bins):")
    for count, edge_left, edge_right in zip(unmatched_counts, unmatched_bins[:-1], unmatched_bins[1:]):
        print(f"[{edge_left:.1f}, {edge_right:.1f}): {count}")
else:
    print("\nNo non-empty values found in 'unmatched' column.")

# Final filter: keep only rows with valid bitstrings, not all-0 or all-1, and unmatched is empty or NaN
final_subset = valid_df[~(all_zeros_mask | all_ones_mask)].copy()
final_subset = final_subset[df.loc[final_subset.index, 'unmatched'].isna()]

# Print final row count and save
print(f"\nNumber of rows in output subset: {len(final_subset)}")
final_subset.to_csv("DATA/PREP/subset.csv", index=False)
