import pandas as pd

# Read the first 10001 lines of the CSV file
df = pd.read_csv('sf_mtc_od.csv', nrows=10001)

# Save to a new file
output_file = 'sf_mtc_od_sample.csv'
df.to_csv(output_file, index=False)

print(f"Successfully saved first {len(df)} lines to {output_file}") 