import json

# Charger les résultats
with open('facts_agent1_stay.jsonl', 'r') as f:
    stays = [json.loads(line) for line in f]

# Stats rapides
print(f"Total stays: {len(stays)}")
print(f"LLM success: {sum(1 for s in stays if s['_extraction_mode'] == 'llm')}")
print(f"Fallback: {sum(1 for s in stays if s['_extraction_mode'] == 'regex_fallback')}")

# Check moyenne RAG scores
import numpy as np
rag_scores = [s['_rag_meta']['rag_score_max'] for s in stays]
print(f"RAG score mean: {np.mean(rag_scores):.2f}")
print(f"RAG score max: {np.max(rag_scores):.2f}")