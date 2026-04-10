import pandas as pd
import pyarrow.parquet as pq
import os

# Find parquet files
output_dir = "output"
files = os.listdir(output_dir)
print("Files in output:", files)

# Check text units for embedding dimensions
for f in files:
    if f.endswith('.parquet'):
        filepath = os.path.join(output_dir, f)
        df = pd.read_parquet(filepath)
        print(f"\n{f}:")
        print(f"  Columns: {list(df.columns)}")
        # Check for embedding columns
        for col in df.columns:
            if 'embed' in col.lower() or 'vector' in col.lower():
                sample = df[col].iloc[0]
                if hasattr(sample, '__len__'):
                    print(f"  {col} dimension: {len(sample)}")