import pandas as pd
import os

output_dir = "output"


# Find the latest run folder
runs = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
if not runs:
	print("No run folders found in the output directory.")
	exit(1)
latest = sorted(runs)[-1]
path = f"{output_dir}/{latest}/artifacts"

# Load and inspect
entities = pd.read_parquet(f"{path}/create_final_entities.parquet")
relationships = pd.read_parquet(f"{path}/create_final_relationships.parquet")
communities = pd.read_parquet(f"{path}/create_final_communities.parquet")

print(f"Total entities extracted: {len(entities)}")
print(f"Total relationships found: {len(relationships)}")
print(f"Total communities detected: {len(communities)}")
print("\n--- Sample Entities ---")
print(entities[['title', 'type', 'description']].head(10))
print("\n--- Sample Relationships ---")
print(relationships[['source', 'target', 'description']].head(10))