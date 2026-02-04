#!/usr/bin/env python3
"""
Debug Agent 1 Patient-Level Aggregation
Analyser les problèmes d'agrégation multi-séjours
"""

import json
from pathlib import Path
from collections import defaultdict

# Paths
FACTS_PATIENT_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\facts_agent1_patient.jsonl"
FACTS_STAY_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\facts_agent1_stay.jsonl"

def analyze_patient_aggregation():
    """Analyser la qualité de l'agrégation patient"""
    
    print("=" * 80)
    print("AGENT 1 AGGREGATION DEBUG")
    print("=" * 80)
    
    # Load patient-level facts
    print(f"\nLoading patient-level facts from {FACTS_PATIENT_PATH}")
    with open(FACTS_PATIENT_PATH, 'r', encoding='utf-8') as f:
        patients = [json.loads(line) for line in f if line.strip()]
    
    print(f"✓ Loaded {len(patients)} patients\n")
    
    # Analyze each patient
    issues_found = 0
    
    for patient in patients:
        patient_id = patient['patient_id']
        n_stays = patient.get('n_stays', 0)
        
        # Check evidence IDs
        evidence_issues = []
        
        # Check disease mentions
        for dm in patient.get('disease_mentions', []):
            ev = dm.get('evidence', {})
            stay_id = ev.get('stay_id')
            line_no = ev.get('line_no')
            
            if stay_id in [None, '', 'null']:
                evidence_issues.append(f"disease_mention: stay_id empty")
            if line_no in [None, '', 'null']:
                evidence_issues.append(f"disease_mention: line_no empty")
        
        # Check labs
        for lab in patient.get('labs', []):
            ev = lab.get('evidence', {})
            stay_id = ev.get('stay_id')
            line_no = ev.get('line_no')
            
            if stay_id in [None, '', 'null']:
                evidence_issues.append(f"lab ({lab.get('test')}): stay_id empty")
            if line_no in [None, '', 'null']:
                evidence_issues.append(f"lab ({lab.get('test')}): line_no empty")
        
        # Check drugs
        for drug in patient.get('drugs', []):
            ev = drug.get('evidence', {})
            stay_id = ev.get('stay_id')
            line_no = ev.get('line_no')
            
            if stay_id in [None, '', 'null']:
                evidence_issues.append(f"drug ({drug.get('name')}): stay_id empty")
            if line_no in [None, '', 'null']:
                evidence_issues.append(f"drug ({drug.get('name')}): line_no empty")
        
        # Check for contradictions in labs
        lab_contradictions = []
        lab_by_test = defaultdict(list)
        
        for lab in patient.get('labs', []):
            test = lab.get('test', '')
            if test:
                lab_by_test[test].append(lab)
        
        for test_name, results in lab_by_test.items():
            if len(results) < 2:
                continue
            
            polarities = [r.get('polarity', '').lower() for r in results if r.get('polarity')]
            
            if 'positive' in polarities and 'negative' in polarities:
                lab_contradictions.append(f"{test_name}: pos/neg found")
            if 'positif' in polarities and 'négatif' in polarities:
                lab_contradictions.append(f"{test_name}: pos/neg found (FR)")
        
        # Report issues
        if evidence_issues or lab_contradictions:
            issues_found += 1
            print(f"Patient {patient_id} ({n_stays} stays):")
            
            if evidence_issues:
                print(f"  Evidence ID issues: {len(evidence_issues)}")
                for issue in evidence_issues[:5]:  # Show first 5
                    print(f"    - {issue}")
                if len(evidence_issues) > 5:
                    print(f"    ... and {len(evidence_issues) - 5} more")
            
            if lab_contradictions:
                print(f"  Lab contradictions: {len(lab_contradictions)}")
                for contradiction in lab_contradictions:
                    print(f"    - {contradiction}")
            
            print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total patients analyzed: {len(patients)}")
    print(f"Patients with issues: {issues_found} ({100*issues_found/len(patients):.1f}%)")
    print(f"Clean patients: {len(patients) - issues_found}")
    
    return patients

def analyze_stay_level():
    """Analyser les facts au niveau séjour pour comparaison"""
    
    print("\n" + "=" * 80)
    print("STAY-LEVEL FACTS ANALYSIS (for comparison)")
    print("=" * 80)
    
    try:
        with open(FACTS_STAY_PATH, 'r', encoding='utf-8') as f:
            stays = [json.loads(line) for line in f if line.strip()]
        
        print(f"\n✓ Loaded {len(stays)} stay-level facts\n")
        
        # Count stays per patient
        stays_per_patient = defaultdict(list)
        for stay in stays:
            patient_id = stay.get('patient_id')
            stay_id = stay.get('stay_id')
            if patient_id:
                stays_per_patient[patient_id].append(stay_id)
        
        print(f"Patients in stay-level: {len(stays_per_patient)}")
        print(f"Avg stays per patient: {len(stays) / len(stays_per_patient):.1f}")
        
        # Show some examples
        print("\nExample stay IDs per patient:")
        for patient_id in list(stays_per_patient.keys())[:5]:
            stay_ids = stays_per_patient[patient_id]
            print(f"  {patient_id}: {len(stay_ids)} stays → {stay_ids}")
        
    except FileNotFoundError:
        print(f"\n⚠️  Stay-level file not found: {FACTS_STAY_PATH}")

def suggest_fix():
    """Suggérer corrections pour l'agrégation"""
    
    print("\n" + "=" * 80)
    print("SUGGESTED FIX FOR AGGREGATION")
    print("=" * 80)
    
    print("""
PROBLEM: Evidence IDs (stay_id, line_no) lost during patient-level aggregation

ROOT CAUSE:
When deduplicating facts across multiple stays, the aggregation code likely:
1. Groups facts by (entity, test, drug) key
2. Merges duplicates
3. But LOSES or CORRUPTS evidence IDs in the process

SOLUTION: Preserve ALL evidence IDs during deduplication

Example fix in Agent 1 code:

# BEFORE (losing evidence)
deduplicated_diseases = list({d['entity']: d for d in all_diseases}.values())

# AFTER (preserving evidence)
deduplicated_diseases = []
seen = {}
for disease in all_diseases:
    key = disease['entity']
    if key not in seen:
        deduplicated_diseases.append(disease)
        seen[key] = disease
    else:
        # Merge evidence from multiple stays
        existing = seen[key]
        if 'evidence' not in existing:
            existing['evidence'] = []
        if isinstance(disease.get('evidence'), dict):
            existing['evidence'].append(disease['evidence'])

SPECIFIC LOCATIONS TO CHECK:
1. Disease mention deduplication
2. Lab result deduplication (key: test name)
3. Drug deduplication (key: drug name)

VALIDATION:
After fix, re-run Agent 1 and verify:
- No stay_id = "" or null
- No line_no = "" or null
- Evidence list contains all source stays
""")

if __name__ == "__main__":
    analyze_patient_aggregation()
    analyze_stay_level()
    suggest_fix()
