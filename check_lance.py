

import lancedb

db = lancedb.connect("output/lancedb")
table = db.open_table("entity_description")

# Print every single field
schema = table.schema
print(f"Total fields: {len(schema)}")
for i, field in enumerate(schema):
    print(f"{i}: {field.name}: {field.type}")