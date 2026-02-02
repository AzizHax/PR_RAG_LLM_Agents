#!/usr/bin/env python3
"""
Agent 2: LLM-based Clinical Reasoning with RAG Guidelines (STRICT VERSION)

Architecture:
- Input: facts.json (from Agent 1)
- RAG: Retrieve relevant ACR-EULAR guidelines
- LLM: Apply guidelines with STRICT reasoning to reduce FP
- Output: criteria_assessment.json with full traceability
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("pip install rank-bm25")

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")


# ============================================================================
# CONFIGURATION - HARDCODED PATHS
# ============================================================================

FACTS_PATIENT_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\facts_agent1_patient.jsonl"
OUTPUT_DIR = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\criteria_assessments_llm"

# LLM Configuration
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b-instruct"  # ou "llama3:8b", "mistral:7b"
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT = 120


# ============================================================================
# ACR-EULAR GUIDELINES (source texte pour RAG)
# ============================================================================

ACR_EULAR_2010_GUIDELINES = """
# The 2010 ACR/EULAR Classification Criteria for Rheumatoid Arthritis

## Reference
Aletaha D, Neogi T, Silman AJ, et al. 2010 Rheumatoid arthritis classification criteria.
Arthritis Rheum. 2010;62(9):2569-2581.

## Classification Criteria (Score-based algorithm)

A score of ≥6/10 is needed for classification as definite RA.

### Category A: Joint Involvement (0-5 points)
- 1 large joint: 0 points
- 2-10 large joints: 1 point
- 1-3 small joints (with or without large joints): 2 points
- 4-10 small joints (with or without large joints): 3 points
- >10 joints (at least 1 small joint): 5 points

Large joints: shoulders, elbows, hips, knees, ankles
Small joints: MCPs, PIPs, MTPs (2nd-5th), thumb IPs, wrists

### Category B: Serology (0-3 points)
- Negative RF AND negative ACPA: 0 points
- Low-positive RF OR low-positive ACPA (≤3× ULN): 2 points
- High-positive RF OR high-positive ACPA (>3× ULN): 3 points

RF = Rheumatoid Factor
ACPA = Anti-citrullinated protein antibody (anti-CCP)
ULN = Upper limit of normal (typically RF: 20 IU/mL, anti-CCP: 10 U/mL)

### Category C: Acute-Phase Reactants (0-1 point)
- Normal CRP AND normal ESR: 0 points
- Abnormal CRP OR abnormal ESR: 1 point

### Category D: Duration of Symptoms (0-1 point)
- <6 weeks: 0 points
- ≥6 weeks: 1 point

## Clinical Context
Treatment patterns provide additional evidence:
- Methotrexate (MTX) = first-line csDMARD
- Biologic DMARDs (anti-TNF, anti-IL6, etc.) = established RA
- JAK inhibitors = treatment-refractory RA

In clinical practice:
- Score ≥6/10 = definite RA
- Score 4-5/10 + DMARD treatment = probable established RA
- Score ≥5/10 without treatment = borderline, likely RA
"""


# ============================================================================
# RAG RETRIEVER
# ============================================================================

class ACREULARRetriever:
    """Retrieve relevant guideline sections using BM25"""
    
    def __init__(self):
        self.guidelines_text = ACR_EULAR_2010_GUIDELINES
        self.chunks = self._create_chunks()
        self.bm25 = self._build_bm25_index()
    
    def _create_chunks(self) -> List[Dict[str, str]]:
        """Split guidelines into retrievable chunks"""
        chunks = []
        sections = self.guidelines_text.split('\n##')
        
        for section in sections:
            if not section.strip():
                continue
            
            lines = section.split('\n')
            title = lines[0].strip('# ').strip()
            content = '\n'.join(lines[1:]).strip()
            
            if content:
                chunks.append({
                    'title': title,
                    'content': content,
                    'text': f"{title} {content}"
                })
        
        return chunks
    
    def _build_bm25_index(self) -> BM25Okapi:
        """Build BM25 index on guideline chunks"""
        corpus = [chunk['text'].lower() for chunk in self.chunks]
        tokenized_corpus = [doc.split() for doc in corpus]
        return BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve top-k most relevant guideline chunks"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.chunks[idx].copy()
                chunk['relevance_score'] = float(scores[idx])
                results.append(chunk)
        
        return results


# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """Client for LLM inference via Ollama"""
    
    def __init__(self, url: str = OLLAMA_URL, model: str = LLM_MODEL, 
                 temperature: float = LLM_TEMPERATURE, timeout: int = LLM_TIMEOUT):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
    
    def generate(self, prompt: str, format_json: bool = True) -> Optional[str]:
        """Generate LLM response"""
        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "format": "json" if format_json else None,
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                print(f"[LLM ERROR] HTTP {response.status_code}")
                return None
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            print(f"[LLM ERROR] {type(e).__name__}: {e}")
            return None


# ============================================================================
# AGENT 2: LLM-BASED CLINICAL REASONING (STRICT)
# ============================================================================

class Agent2LLMReasoning:
    """
    Agent 2 with LLM-based clinical reasoning (STRICT VERSION)
    
    Input: facts.json (patient-level from Agent 1)
    Process: RAG guidelines + STRICT LLM reasoning
    Output: criteria_assessment.json
    """
    
    def __init__(self):
        self.retriever = ACREULARRetriever()
        self.llm = LLMClient()
    
    def _build_prompt(self, patient_facts: Dict[str, Any], guidelines: List[Dict]) -> str:
        """Build STRICT LLM prompt with facts + retrieved guidelines"""
        
        patient_id = patient_facts.get('patient_id')
        disease_mentions = patient_facts.get('disease_mentions', [])
        labs = patient_facts.get('labs', [])
        drugs = patient_facts.get('drugs', [])
        
        # Format facts
        facts_summary = self._summarize_facts(disease_mentions, labs, drugs)
        
        # Format guidelines
        guidelines_text = "\n\n".join([
            f"### {g['title']}\n{g['content']}"
            for g in guidelines
        ])
        
        prompt = f"""You are an expert rheumatologist applying ACR-EULAR 2010 criteria for Rheumatoid Arthritis classification.

# PATIENT FACTS (Patient {patient_id})

{facts_summary}

# RELEVANT ACR-EULAR GUIDELINES

{guidelines_text}

# YOUR TASK

Apply the ACR-EULAR 2010 criteria to classify this patient. Score each category (A-D) based on the guidelines and patient facts.

SCORING GUIDELINES:
1. **Joint Involvement (Category A)**: 
   - 5 points: EXPLICIT >10 joints OR clear polyarticular involvement
   - 3 points: Confirmed RA with moderate involvement
   - 2 points: Confirmed RA without details OR multiple joint mentions
   - 1 point: Single RA mention without confirmation
   - Balance: Don't over-score assumptions, but don't ignore clinical patterns

2. **Serology (Category B)**: 
   - 3 points: High-positive RF (>60 IU/mL) OR anti-CCP (>30 U/mL)
   - 2 points: Low-positive RF or anti-CCP
   - 0 points: Negative or absent
   - Check actual values when available, label when not

3. **Acute Phase (Category C)**: 
   - 1 point: Elevated CRP or ESR documented

4. **Duration (Category D)**: 
   - 1 point: Chronic/long-standing mentioned OR patient on DMARDs (suggests established disease)

5. **Treatment Context**: 
   - DMARDs (especially biologics/JAKi) are STRONG evidence of established RA
   - Consider treatment when score is borderline (5-6 points)

BALANCED CLASSIFICATION LOGIC:
- Score ≥7/10 → RA+ (strong evidence)
- Score 6/10 → RA+ if DMARD treatment OR very clear clinical picture
- Score 5/10 → RA+ if biologic/JAKi treatment (very strong evidence)
- Score <5/10 → RA− unless exceptional evidence

IMPORTANT: Established RA under treatment may have lower scores due to disease control.
Balance strict criteria interpretation with clinical reality.

# OUTPUT FORMAT (strict JSON)

{{
  "component_scores": {{
    "joint_involvement": {{
      "score": <0-5>,
      "justification": "..."
    }},
    "serology": {{
      "score": <0-3>,
      "justification": "..."
    }},
    "acute_phase": {{
      "score": <0-1>,
      "justification": "..."
    }},
    "duration": {{
      "score": <0-1>,
      "justification": "..."
    }}
  }},
  "acr_eular_total": <sum of scores>,
  "treatment_context": {{
    "has_dmard": <true/false>,
    "treatment_strength": "none" | "moderate" | "strong",
    "justification": "..."
  }},
  "final_label": "RA+" | "RA−",
  "classification_rationale": "...",
  "confidence": "low" | "medium" | "high"
}}
"""
        
        return prompt
    
    def _summarize_facts(self, disease_mentions: List, labs: List, drugs: List) -> str:
        """Summarize patient facts for LLM"""
        lines = []
        
        lines.append("## Disease Mentions")
        if disease_mentions:
            for dm in disease_mentions:
                entity = dm.get('entity', 'unknown')
                status = dm.get('status', 'unknown')
                lines.append(f"- {entity} (status: {status})")
        else:
            lines.append("- None documented")
        
        lines.append("\n## Laboratory Results")
        if labs:
            for lab in labs:
                test = lab.get('test', 'unknown')
                polarity = lab.get('polarity', 'unknown')
                value = lab.get('value', '')
                if value:
                    lines.append(f"- {test}: {polarity} (value: {value})")
                else:
                    lines.append(f"- {test}: {polarity}")
        else:
            lines.append("- None documented")
        
        lines.append("\n## Medications")
        if drugs:
            for drug in drugs:
                name = drug.get('name', 'unknown')
                category = drug.get('category', 'unknown')
                lines.append(f"- {name} (category: {category})")
        else:
            lines.append("- None documented")
        
        return '\n'.join(lines)
    
    def process_patient(self, patient_facts: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Process single patient with STRICT LLM reasoning"""
        
        patient_id = patient_facts.get('patient_id')
        n_stays = patient_facts.get('n_stays', 0)
        
        print(f"[Agent 2 LLM] Processing patient {patient_id} ({n_stays} stays)...")
        
        # 1. Build query for RAG
        disease_mentions = patient_facts.get('disease_mentions', [])
        labs = patient_facts.get('labs', [])
        drugs = patient_facts.get('drugs', [])
        
        query_parts = []
        if disease_mentions:
            query_parts.append('joint involvement classification')
        if labs:
            query_parts.append('serology RF anti-CCP scoring')
        if drugs:
            query_parts.append('treatment DMARD context')
        
        query = ' '.join(query_parts) if query_parts else 'ACR-EULAR criteria'
        
        # 2. Retrieve guidelines
        guidelines = self.retriever.retrieve(query, top_k=3)
        
        # 3. Build BALANCED prompt
        prompt = self._build_prompt(patient_facts, guidelines)
        
        # 4. LLM inference
        print(f"[Agent 2 LLM]   Calling LLM (BALANCED mode)...")
        llm_response = self.llm.generate(prompt, format_json=True)
        
        if not llm_response:
            print(f"[Agent 2 LLM]   ERROR: LLM returned empty response")
            return self._fallback_assessment(patient_id)
        
        # 5. Parse LLM output
        try:
            assessment = json.loads(llm_response)
        except json.JSONDecodeError as e:
            print(f"[Agent 2 LLM]   ERROR: Failed to parse LLM JSON: {e}")
            return self._fallback_assessment(patient_id)
        
        # 6. Validate and augment assessment
        assessment = self._validate_and_augment(assessment, patient_id, guidelines, patient_facts)
        
        # 7. Save
        output_path = Path(output_dir) / f"criteria_assessment_{patient_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(assessment, f, indent=2, ensure_ascii=False)
        
        print(f"[Agent 2 LLM]   ✓ Saved (label: {assessment['final_label']}, score: {assessment.get('acr_eular_total', 0)})")
        
        return assessment
    
    def _validate_and_augment(self, assessment: Dict, patient_id: str, 
                             guidelines: List[Dict], patient_facts: Dict) -> Dict:
        """Validate LLM output and add metadata"""
        
        # Ensure required fields
        assessment.setdefault('patient_id', patient_id)
        assessment.setdefault('agent', 'Agent2_LLM_Reasoning_BALANCED')
        assessment.setdefault('acr_eular_total', 0)
        assessment.setdefault('final_label', 'RA−')
        assessment.setdefault('confidence', 'low')
        
        # Add guideline references
        assessment['retrieved_guidelines'] = guidelines
        assessment['guideline_citation'] = 'Aletaha D, et al. Arthritis Rheum. 2010;62(9):2569-2581'
        
        # Extract evidence IDs
        assessment['evidence_trace'] = self._extract_evidence_ids(patient_facts)
        
        # Compute confidence if not provided
        if 'confidence' not in assessment or not assessment['confidence']:
            score = assessment.get('acr_eular_total', 0)
            has_treatment = assessment.get('treatment_context', {}).get('has_dmard', False)
            
            if score >= 7 and has_treatment:
                assessment['confidence'] = 'high'
            elif score >= 6:
                assessment['confidence'] = 'medium'
            else:
                assessment['confidence'] = 'low'
        
        return assessment
    
    def _extract_evidence_ids(self, patient_facts: Dict) -> List[Dict]:
        """Extract evidence IDs for traceability"""
        evidence = []
        
        for mention in patient_facts.get('disease_mentions', []):
            ev = mention.get('evidence', {})
            if ev:
                evidence.append({
                    'type': 'disease_mention',
                    'stay_id': ev.get('stay_id'),
                    'line_no': ev.get('line_no'),
                    'snippet': ev.get('snippet', '')[:100]
                })
        
        for lab in patient_facts.get('labs', []):
            ev = lab.get('evidence', {})
            if ev:
                evidence.append({
                    'type': 'lab',
                    'test': lab.get('test'),
                    'stay_id': ev.get('stay_id'),
                    'line_no': ev.get('line_no'),
                    'snippet': ev.get('snippet', '')[:100]
                })
        
        for drug in patient_facts.get('drugs', []):
            ev = drug.get('evidence', {})
            if ev:
                evidence.append({
                    'type': 'drug',
                    'name': drug.get('name'),
                    'stay_id': ev.get('stay_id'),
                    'line_no': ev.get('line_no'),
                    'snippet': ev.get('snippet', '')[:100]
                })
        
        return evidence
    
    def _fallback_assessment(self, patient_id: str) -> Dict:
        """Fallback assessment if LLM fails"""
        return {
            'patient_id': patient_id,
            'agent': 'Agent2_LLM_Reasoning_FALLBACK',
            'acr_eular_total': 0,
            'final_label': 'RA−',
            'confidence': 'low',
            'classification_rationale': 'LLM inference failed, defaulting to RA−',
            'component_scores': {},
            'treatment_context': {},
            'retrieved_guidelines': [],
            'evidence_trace': []
        }
    
    def process_jsonl(self, jsonl_path: str, output_dir: str):
        """Process JSONL file with patient-level facts"""
        
        jsonl_path = Path(jsonl_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[Agent 2 LLM] Reading patient facts from: {jsonl_path}")
        
        assessments = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    patient_facts = json.loads(line)
                    assessment = self.process_patient(patient_facts, output_dir)
                    assessments.append(assessment)
                except json.JSONDecodeError as e:
                    print(f"[Agent 2 LLM] ERROR: Invalid JSON at line {line_no}: {e}")
                    continue
                except Exception as e:
                    print(f"[Agent 2 LLM] ERROR: Processing failed for line {line_no}: {e}")
                    continue
        
        print(f"\n[Agent 2 LLM] ✓ Batch processing complete: {len(assessments)} patients")
        print(f"[Agent 2 LLM] ✓ Output directory: {output_dir}")
        
        return assessments


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    
    print("=" * 80)
    print("AGENT 2: LLM-based Clinical Reasoning (BALANCED VERSION)")
    print("=" * 80)
    print(f"\nLLM Model: {LLM_MODEL}")
    print(f"Strategy: BALANCED (optimize F1-score)")
    print(f"Input:  {FACTS_PATIENT_PATH}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Check input file
    if not Path(FACTS_PATIENT_PATH).exists():
        print(f"[Agent 2 LLM] ERROR: Input file not found!")
        print(f"              Expected: {FACTS_PATIENT_PATH}")
        return
    
    # Initialize Agent 2
    agent2 = Agent2LLMReasoning()
    
    # Process all patients
    agent2.process_jsonl(FACTS_PATIENT_PATH, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("✓ Agent 2 LLM BALANCED processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()