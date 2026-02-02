Système Multi-Agents de Phénotypage de la Polyarthrite Rhumatoïde (PR)
📋 Vue d'ensemble
Système multi-agents pour le phénotypage automatique de la Polyarthrite Rhumatoïde (PR/RA) à partir de dossiers patients informatisés (DPI) non structurés.
Architecture : 3 agents spécialisés + RAG multi-source + Guidelines ACR-EULAR 2010
Dataset : 25 patients fantômes annotés (15 RA+, 10 RA−)
Performance actuelle : F1 = 0.800 (Agent 2 LLM) | F1 = 0.875 (Agent 1 combiné)

🏗️ Architecture du système
DPI texte brut (.txt)
         ↓
    [Agent 1] ← RAG-EHR + RAG-KB
         ↓
  facts.json (structured extraction with evidence IDs)
         ↓
    [Agent 2] ← RAG-Guidelines (ACR-EULAR 2010)
         ↓
criteria_assessment.json
         ↓
    [Agent 3] ← Orchestration + Guardrails (À IMPLÉMENTER)
         ↓
final_decision.json
Composants implémentés :

✅ Agent 1 : Extraction KB-guidée + RAG
✅ Agent 2 : Raisonnement LLM guideline-grounded (3 versions)
✅ RAG-Guidelines : ACR-EULAR PDF → BM25
✅ Évaluation comparative
🚧 Agent 3 : Design validé, implémentation en attente


📂 Fichiers principaux
Code source
src/
├── agent1_kb_guided_extraction.py      # Agent 1 (PRODUCTION)
├── agent2_llm_strict.py                # Agent 2 LLM balanced (PRODUCTION)
├── evaluate_agents.py                  # Évaluation
└── eda_dpi_clinical_notes.py          # EDA données réelles

Data/
├── facts_agent1_patient.jsonl          # Output Agent 1
├── criteria_assessments_llm/           # Output Agent 2
├── phantom_annotations_minimal.txt     # Ground truth
└── evaluation_results.json             # Résultats

🤖 Agent 1 : Extraction KB-guidée
Pipeline

Prefiltering : Anchors (PR, RF, anti-CCP, DMARDs) + contexte
Chunking : Fenêtres glissantes (6 lignes, stride 3)
BM25 + Reranking : Sélection chunks pertinents
LLM Extraction : Qwen2.5:3b + KB-guided prompt
Backfill v3 : Fallback regex

Performance
ApprocheF1AccPrecRecExtraction score0.8570.800.751.00Combined0.8750.840.820.93

🧠 Agent 2 : Raisonnement Clinique
Évolution

Rule-based v1 : F1 = 0.333 (trop strict, recall 0.20)
LLM laxiste : F1 = 0.769 (9 FP/10 RA−)
LLM strict : F1 = 0.690 (overcorrection, 5 FN)
LLM BALANCED ✅ : F1 = 0.800 (5 FP, 1 FN)

Stratégie BALANCED (production)
Classification logic:
- Score ≥7/10 → RA+
- Score 6/10 + DMARD → RA+
- Score 5/10 + biologic/JAKi → RA+
- Score <5/10 → RA−
Insight clé : "Established RA under treatment may have lower scores due to disease control"

📊 Résultats comparatifs
ApprocheAccPrecRecF1TPFPFNTNAgent 1 (Combined)0.8400.8240.9330.87514317Agent 2 (LLM Balanced)0.7200.7000.9330.80014516
Ground Truth : 15 RA+, 10 RA−

🔍 RAG Multi-Source
RAG-EHR (Agent 1)

BM25 + semantic reranking sur timeline patient
Sélection chunks pertinents pour LLM

RAG-KB (Agent 1)

pr_phenotype_spec_v2.json (10 slots)
Guidance du prompt d'extraction

RAG-Guidelines (Agent 2)

ACR-EULAR 2010 PDF (Aletaha et al., 2010)
BM25 retrieval de sections pertinentes
Contexte dynamique pour raisonnement LLM

Example retrieval :
Query: "serology RF anti-CCP scoring"
Retrieved:
  1. Category B: Serology (0-3 points)
  2. Classification Criteria
  3. Clinical Context (Treatment)

📈 Évaluation
Métriques

Accuracy, Precision, Recall, F1, Specificity
Confusion matrix (TP, FP, FN, TN)
Threshold optimization (Agent 1)
Error analysis (FP/FN avec justifications)

Usage
bashpython evaluate_agents.py
# Output: console + evaluation_results.json

🛠️ Technologies

LLM : Qwen2.5:3b-instruct (Ollama local)
Retrieval : BM25 (rank-bm25), chunking custom
Évaluation : scikit-learn
Dépendances : rank-bm25 requests scikit-learn numpy


🚀 Utilisation
1. Agent 1 : Extraction
bashpython agent1_kb_guided_extraction.py
# Output: facts_agent1_patient.jsonl
2. Agent 2 : Raisonnement
bashpython agent2_llm_strict.py  # Version BALANCED
# Output: criteria_assessments_llm/criteria_assessment_*.json
3. Évaluation
bashpython evaluate_agents.py
# Output: console + evaluation_results.json

🎯 Prochaines étapes
Court terme

Finaliser Agent 2 : F1 ≥ 0.85
Implémenter Agent 3 : Orchestrateur + guardrails
Évaluation Agent 3 vs agents individuels

Moyen terme

Données réelles : EDA + adaptation pipeline
Ablation studies : Impact KB, RAG, backfill
Grid search Agent 3 : Optimisation thresholds

Long terme

Packaging : Module phenorag pip-installable
Fine-tuning : LLM spécialisé RA (si n ≥ 500)
Déploiement : API REST (FastAPI)
Certification : Validation clinique


📝 Référence
Guidelines ACR-EULAR 2010 :
Aletaha D, Neogi T, Silman AJ, et al. 2010 Rheumatoid arthritis classification criteria. Arthritis Rheum. 2010;62(9):2569-2581.