#!/usr/bin/env python3
"""
Quick Patient Aggregation Script (FIXED VERSION)
Ne fait QUE l'agrégation sans re-run l'extraction
"""

import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

FACTS_STAY_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\facts_agent1_stay.jsonl"
OUTPUT_PATIENT_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\facts_agent1_patient.jsonl"


# ============================================================================
# DEDUPLICATION WITH EVIDENCE PRESERVATION (FIXED)
# ============================================================================

def deduplicate_with_evidence_merge(entities: List[Dict], key_field: str) -> List[Dict]:
    """
    Deduplicate entities while preserving ALL evidences from multiple stays
    
    Args:
        entities: List of entities (diseases, labs, or drugs)
        key_field: Field to use as deduplication key ('entity', 'test', or 'name')
    
    Returns:
        Deduplicated list with merged evidences
    """
    seen = {}
    deduplicated = []
    
    for entity in entities:
        # Get the deduplication key
        key_value = entity.get(key_field, '')
        
        # Create a unique key that includes key attributes
        # For labs: include test + polarity to keep RF+ and RF− separate
        if key_field == 'test':
            polarity = entity.get('polarity') or ''  # Handle None
            polarity = polarity.lower()
            # Unique key: test_polarity (e.g., "RF_positive", "RF_negative")
            if polarity in ['positive', 'positif', 'negative', 'négatif', 'negatif']:
                unique_key = f"{key_value}_{polarity}"
            else:
                unique_key = f"{key_value}_unknown"
        else:
            # For diseases and drugs, use just the key field
            unique_key = key_value
        
        # FIXED: Don't skip entities with missing keys - keep them with unique ID
        if not unique_key or unique_key in ['_unknown', '']:
            # Generate unique ID using Python's id() to preserve all entities
            unique_key = f"{key_field}_unknown_{id(entity)}"
        
        if unique_key not in seen:
            # First occurrence - add to list
            entity_copy = entity.copy()
            
            # Convert evidence to list format for consistency
            if 'evidence' in entity_copy:
                if isinstance(entity_copy['evidence'], dict):
                    entity_copy['evidence'] = [entity_copy['evidence']]
                elif not isinstance(entity_copy['evidence'], list):
                    entity_copy['evidence'] = []
            else:
                entity_copy['evidence'] = []
            
            deduplicated.append(entity_copy)
            seen[unique_key] = entity_copy
        else:
            # Duplicate found - merge evidence
            existing = seen[unique_key]
            new_evidence = entity.get('evidence')
            
            if new_evidence:
                # Ensure evidence is a list in existing entity
                if 'evidence' not in existing:
                    existing['evidence'] = []
                elif not isinstance(existing['evidence'], list):
                    existing['evidence'] = [existing['evidence']]
                
                # Add new evidence (avoid true duplicates)
                if isinstance(new_evidence, dict):
                    # Only add if not already present
                    if new_evidence not in existing['evidence']:
                        existing['evidence'].append(new_evidence)
                elif isinstance(new_evidence, list):
                    for ev in new_evidence:
                        if ev not in existing['evidence']:
                            existing['evidence'].append(ev)
    
    return deduplicated


# ============================================================================
# PATIENT AGGREGATION
# ============================================================================

def aggregate_to_patient_level(stay_facts_path: str, output_path: str):
    """Aggregate stays by patient with evidence preservation"""
    
    print("=" * 80)
    print("QUICK PATIENT AGGREGATION (FIXED - no entity skipping)")
    print("=" * 80)
    
    # Load all stays
    print(f"\nLoading stay-level facts from {stay_facts_path}")
    with open(stay_facts_path, 'r', encoding='utf-8') as f:
        stays = [json.loads(line) for line in f]
    
    print(f"✓ Loaded {len(stays)} stay-level facts\n")
    
    # Group by patient
    patient_groups = defaultdict(list)
    for stay in stays:
        patient_id = stay['patient_id']
        patient_groups[patient_id].append(stay)
    
    print(f"Grouping: {len(patient_groups)} unique patients\n")
    
    # Aggregate each patient
    patient_facts = []
    
    for patient_id, patient_stays in patient_groups.items():
        print(f"  Aggregating patient {patient_id} ({len(patient_stays)} stays)...")
        
        # Merge all entities
        all_disease = []
        all_labs = []
        all_drugs = []
        
        # RAG metadata aggregation
        rag_scores_max = []
        rag_scores_sum = []
        
        for stay in patient_stays:
            all_disease.extend(stay.get('disease_mentions', []))
            all_labs.extend(stay.get('labs', []))
            all_drugs.extend(stay.get('drugs', []))
            
            rag_meta = stay.get('_rag_meta', {})
            rag_scores_max.append(rag_meta.get('rag_score_max', 0.0))
            rag_scores_sum.append(rag_meta.get('rag_score_sum_top3', 0.0))
        
        # DEBUG: Show before deduplication
        print(f"    [BEFORE] Diseases: {len(all_disease)}, Labs: {len(all_labs)}, Drugs: {len(all_drugs)}")
        
        # Deduplicate entities while preserving ALL evidences (FIXED)
        unique_disease = deduplicate_with_evidence_merge(all_disease, key_field='entity')
        unique_labs = deduplicate_with_evidence_merge(all_labs, key_field='test')
        unique_drugs = deduplicate_with_evidence_merge(all_drugs, key_field='name')
        
        # DEBUG: Show after deduplication
        print(f"    [AFTER]  Diseases: {len(unique_disease)}, Labs: {len(unique_labs)}, Drugs: {len(unique_drugs)}")
        
        # DEBUG: Show lab details
        if len(unique_labs) > 0:
            lab_details = []
            for lab in unique_labs[:3]:  # Show first 3
                test = lab.get('test', 'unknown')
                polarity = lab.get('polarity', 'unknown')
                lab_details.append(f"{test}({polarity})")
            print(f"    [LABS]   Examples: {', '.join(lab_details)}")
        
        # Calculate extraction score for comparison
        extraction_score = 0.0
        
        # Disease mentions score
        confirmed = [d for d in unique_disease if d.get('status') == 'confirmed']
        negated = [d for d in unique_disease if d.get('status') == 'negated']
        if confirmed:
            extraction_score += 2.0
        elif unique_disease and not negated:
            extraction_score += 1.0
        if negated:
            extraction_score -= 2.0
        
        # Serology score
        rf_pos = any(l.get('test', '').lower() in ['rf', 'facteur rhumatoïde'] and 
                     l.get('polarity', '').lower() in ['positive', 'positif'] 
                     for l in unique_labs)
        acpa_pos = any('ccp' in l.get('test', '').lower() or 'acpa' in l.get('test', '').lower()
                       and l.get('polarity', '').lower() in ['positive', 'positif']
                       for l in unique_labs)
        if rf_pos and acpa_pos:
            extraction_score += 3.0
        elif rf_pos or acpa_pos:
            extraction_score += 2.0
        
        # Drug score
        has_mtx = any('methotrexate' in d.get('name', '').lower() or 'mtx' in d.get('name', '').lower() 
                      for d in unique_drugs)
        has_biologic = any(d.get('category') == 'bDMARD' for d in unique_drugs)
        has_jak = any(d.get('category') == 'tsDMARD' for d in unique_drugs)
        if has_biologic or has_jak:
            extraction_score += 2.5
        elif has_mtx:
            extraction_score += 1.5
        
        print(f"    [SCORE]  Extraction score: {extraction_score:.2f}")
        print()
        
        patient_facts.append({
            'patient_id': patient_id,
            'n_stays': len(patient_stays),
            'disease_mentions': unique_disease,
            'labs': unique_labs,
            'drugs': unique_drugs,
            '_rag_meta_patient': {
                'rag_score_max': max(rag_scores_max) if rag_scores_max else 0.0,
                'rag_score_mean': sum(rag_scores_max) / len(rag_scores_max) if rag_scores_max else 0.0,
                'rag_score_sum_top3_max': max(rag_scores_sum) if rag_scores_sum else 0.0,
            }
        })
    
    # Save to JSONL
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for patient in patient_facts:
            f.write(json.dumps(patient, ensure_ascii=False) + '\n')
    
    print("\n" + "=" * 80)
    print("✓ AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"✓ Aggregated {len(patient_facts)} patients")
    print(f"✓ Output: {output_path}")
    print(f"✓ Evidence preservation: RF+/RF− kept separate, ALL entities preserved")
    print(f"✓ FIX: Unknown entities kept with unique IDs (not skipped)")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    aggregate_to_patient_level(FACTS_STAY_PATH, OUTPUT_PATIENT_PATH)