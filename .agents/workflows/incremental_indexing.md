---
description: How to safely add more cases to the GraphRAG index without losing existing progress
---

This workflow outlines the exact steps to incrementally add more judgment files to your LexGraph project. By following this carefully, you ensure that GraphRAG skips processing the files it has already seen (saving you hours of Ollama GPU time) while intelligently integrating the new documents.

### Step 1: Add New Files to Input
Copy your desired additional `.txt` cases from the `legal_corpus\` directory into your `input\` directory. 
*Important:* Do NOT delete the existing 40 files in the `input\` directory. GraphRAG needs to see the entire combined dataset.

### Step 2: Wipe the Output Directory
You must clear the previously generated graph outputs to force GraphRAG to recalculate the communities (since new documents will create new connections).

*CRITICAL:* We are only deleting `output\`. **DO NOT delete the `cache\` directory.** The `cache\` folder is where all the expensive 15-second LLM responses are permanently saved!

// turbo
```bash
rmdir /s /q output || Remove-Item -Recurse -Force output
```

### Step 3: Run the Indexer
Run the GraphRAG index command. Because your `cache\` folder is intact, GraphRAG will near-instantly fly through the original 40 documents and automatically slow down only when it encounters the newly added documents.

// turbo
```bash
graphrag index --root .
```

### Step 4: Verify the New Index
Once finished, you can run a local or global search to confirm the new files are successfully woven into the Knowledge Graph.
