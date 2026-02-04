#!/usr/bin/env python3
"""
Agent 2 (FINAL v3): LLM + RAG-Guidelines + Deterministic Guardrails

NOUVEAU dans v3:
- RAG-Guidelines complet avec retrieval BM25 sur critères ACR/EULAR structurés
- Guidelines structurées avec seuils cliniques précis
- Retrieval contextuel selon facts extraits (serology, inflammation, etc.)
- Explicability améliorée (citations des guidelines utilisées)

Goal:
- Use LLM to apply ACR/EULAR 2010 logic with RAG-augmented guidelines
- Add deterministic guardrails (anti-hallucination / anti-FP)
- Keep patient_id as integer-like strings (e.g., "1", "2", ...), no zfill

Input:
- facts_agent1_patient.jsonl (patient-level facts from Agent 1)

Output:
- criteria_assessments_llm/criteria_assessment_<patient_id>.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("pip install rank-bm25")

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")


# =============================================================================
# CONFIG - HARDCODED PATHS
# =============================================================================

FACTS_PATIENT_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\facts_agent1_patient.jsonl"
OUTPUT_DIR = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\criteria_assessments_llm"

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b-instruct"
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT = 120


# =============================================================================
# RAG-GUIDELINES: STRUCTURED ACR/EULAR 2010 KNOWLEDGE BASE
# =============================================================================

ACR_EULAR_2010_STRUCTURED = {
    "overview": {
        "title": "ACR/EULAR 2010 Classification Criteria for RA",
        "description": "A score of ≥6/10 is needed for classification as definite RA",
        "reference": "Aletaha D, et al. Arthritis Rheum. 2010;62(9):2569-2581",
        "scoring_range": "0-10 points",
        "classification_threshold": "≥6 points"
    },
    
    "joint_involvement": {
        "category": "A",
        "title": "Joint Involvement",
        "max_score": 5,
        "description": "Number and type of joints involved (large vs small joints)",
        "criteria": [
            {
                "condition": "1 large joint",
                "score": 0,
                "details": "Shoulders, elbows, hips, knees, ankles"
            },
            {
                "condition": "2-10 large joints",
                "score": 1,
                "details": "Multiple large joints involved"
            },
            {
                "condition": "1-3 small joints",
                "score": 2,
                "details": "MCPs, PIPs, 2nd-5th MTPs, thumb IPs, wrists (with or without large joints)"
            },
            {
                "condition": "4-10 small joints",
                "score": 3,
                "details": "Multiple small joints (with or without large joints)"
            },
            {
                "condition": ">10 joints (at least 1 small)",
                "score": 5,
                "details": "Extensive polyarticular involvement"
            }
        ],
        "clinical_notes": [
            "Small joints: MCPs, PIPs, 2nd-5th MTPs, thumb IPs, wrists",
            "Large joints: shoulders, elbows, hips, knees, ankles",
            "DIP joints, 1st CMC, 1st MTP excluded",
            "Clinical examination or imaging (US, MRI) can confirm synovitis"
        ]
    },
    
    "serology": {
        "category": "B",
        "title": "Serology (RF and/or Anti-CCP/ACPA)",
        "max_score": 3,
        "description": "Rheumatoid factor and anti-citrullinated protein antibodies",
        "criteria": [
            {
                "condition": "Negative RF AND negative anti-CCP",
                "score": 0,
                "details": "Both markers below threshold"
            },
            {
                "condition": "Low-positive RF OR low-positive anti-CCP",
                "score": 2,
                "details": "≤3× ULN (upper limit of normal)"
            },
            {
                "condition": "High-positive RF OR high-positive anti-CCP",
                "score": 3,
                "details": ">3× ULN"
            }
        ],
        "thresholds": {
            "RF_positive": "> 20 UI/mL (typical ULN)",
            "RF_low_positive": "20-60 UI/mL (≤3× ULN)",
            "RF_high_positive": "> 60 UI/mL (>3× ULN)",
            "anti_CCP_positive": "> 10 U/mL",
            "anti_CCP_low_positive": "10-30 U/mL (≤3× ULN)",
            "anti_CCP_high_positive": "> 30 U/mL (>3× ULN)"
        },
        "clinical_notes": [
            "ULN varies by laboratory; adjust accordingly",
            "Either RF OR anti-CCP positivity sufficient",
            "High titers = stronger RA specificity",
            "ACPA = anti-CCP = anti-citrullinated protein antibodies"
        ]
    },
    
    "acute_phase_reactants": {
        "category": "C",
        "title": "Acute-Phase Reactants",
        "max_score": 1,
        "description": "Evidence of systemic inflammation",
        "criteria": [
            {
                "condition": "Normal CRP AND normal ESR",
                "score": 0,
                "details": "No acute inflammation markers"
            },
            {
                "condition": "Abnormal CRP OR abnormal ESR",
                "score": 1,
                "details": "At least one marker elevated"
            }
        ],
        "thresholds": {
            "CRP_abnormal": "> 10 mg/L",
            "ESR_abnormal": "> 20 mm/h (> 30 mm/h if elderly)",
            "ESR_alternative_name": "VHS (vitesse de sédimentation)"
        },
        "clinical_notes": [
            "CRP more specific than ESR for inflammation",
            "ESR can be affected by age, anemia, other factors",
            "Normal inflammatory markers don't exclude RA"
        ]
    },
    
    "duration": {
        "category": "D",
        "title": "Duration of Symptoms",
        "max_score": 1,
        "description": "Chronicity of synovitis symptoms",
        "criteria": [
            {
                "condition": "< 6 weeks",
                "score": 0,
                "details": "Acute or early presentation"
            },
            {
                "condition": "≥ 6 weeks",
                "score": 1,
                "details": "Chronic or persistent symptoms"
            }
        ],
        "clinical_notes": [
            "Patient-reported symptom duration",
            "Symptoms: joint pain, stiffness, swelling",
            "Morning stiffness >30 min suggestive of inflammatory arthritis"
        ]
    },
    
    "treatment_context": {
        "title": "Treatment Context (Not in ACR/EULAR scoring but clinically relevant)",
        "description": "Presence of DMARDs or biologics strongly suggests established RA",
        "categories": {
            "csDMARD": {
                "name": "Conventional synthetic DMARDs",
                "examples": ["methotrexate (MTX)", "leflunomide", "sulfasalazine", "hydroxychloroquine"],
                "interpretation": "Moderate evidence of RA diagnosis"
            },
            "bDMARD": {
                "name": "Biologic DMARDs",
                "examples": ["anti-TNF (adalimumab, etanercept, infliximab)", "rituximab", "tocilizumab", "abatacept"],
                "interpretation": "Strong evidence of established RA"
            },
            "tsDMARD": {
                "name": "Targeted synthetic DMARDs (JAK inhibitors)",
                "examples": ["tofacitinib", "baricitinib", "upadacitinib"],
                "interpretation": "Strong evidence of established RA"
            }
        },
        "clinical_notes": [
            "DMARD prescription implies rheumatologist has diagnosed RA",
            "Biologics/JAK inhibitors reserved for moderate-to-severe RA",
            "Presence of DMARD can support RA+ when score borderline (5-6)"
        ]
    },
    
    "interpretation_guide": {
        "title": "Clinical Interpretation Guidelines",
        "scoring_logic": {
            "definite_RA": "Total score ≥6/10",
            "borderline_cases": {
                "score_6": "RA+ if DMARD present OR clear clinical RA diagnosis documented",
                "score_5": "RA+ only if biologic/JAK OR confirmed RA diagnosis, else RA−",
                "score_4_or_less": "RA− unless exceptional documented evidence"
            }
        },
        "special_cases": [
            "Erosions typical of RA on imaging → can classify as RA even if score <6",
            "Seronegative RA exists (RF− and anti-CCP−) but less common",
            "Extra-articular manifestations (nodules, ILD) support RA diagnosis"
        ]
    }
}


# =============================================================================
# RAG RETRIEVER (BM25 over structured guidelines)
# =============================================================================

class RAGGuidelinesRetriever:
    """
    RAG retriever pour guidelines ACR/EULAR structurées
    Utilise BM25 pour retrieval contextuel selon facts extraits
    """
    
    def __init__(self):
        self.guidelines = ACR_EULAR_2010_STRUCTURED
        self.chunks = self._create_chunks()
        self.bm25 = self._build_bm25()
    
    def _create_chunks(self) -> List[Dict[str, Any]]:
        """Aplatir guidelines structurées en chunks pour BM25"""
        chunks = []
        
        # Overview
        overview = self.guidelines["overview"]
        chunks.append({
            "section": "overview",
            "title": overview["title"],
            "content": overview,
            "text": f"{overview['title']} {overview['description']} threshold {overview['classification_threshold']}"
        })
        
        # Joint involvement
        joint = self.guidelines["joint_involvement"]
        joint_text = f"{joint['title']} {joint['description']} "
        for criterion in joint["criteria"]:
            joint_text += f"{criterion['condition']} {criterion['score']} points {criterion['details']} "
        joint_text += " ".join(joint["clinical_notes"])
        chunks.append({
            "section": "joint_involvement",
            "title": joint["title"],
            "content": joint,
            "text": joint_text
        })
        
        # Serology
        serology = self.guidelines["serology"]
        sero_text = f"{serology['title']} {serology['description']} RF rheumatoid factor anti-CCP ACPA "
        for criterion in serology["criteria"]:
            sero_text += f"{criterion['condition']} {criterion['score']} points "
        for k, v in serology["thresholds"].items():
            sero_text += f"{k} {v} "
        sero_text += " ".join(serology["clinical_notes"])
        chunks.append({
            "section": "serology",
            "title": serology["title"],
            "content": serology,
            "text": sero_text
        })
        
        # Acute phase
        acute = self.guidelines["acute_phase_reactants"]
        acute_text = f"{acute['title']} {acute['description']} CRP ESR VHS inflammation "
        for criterion in acute["criteria"]:
            acute_text += f"{criterion['condition']} {criterion['score']} points "
        for k, v in acute["thresholds"].items():
            acute_text += f"{k} {v} "
        acute_text += " ".join(acute["clinical_notes"])
        chunks.append({
            "section": "acute_phase_reactants",
            "title": acute["title"],
            "content": acute,
            "text": acute_text
        })
        
        # Duration
        duration = self.guidelines["duration"]
        dur_text = f"{duration['title']} {duration['description']} "
        for criterion in duration["criteria"]:
            dur_text += f"{criterion['condition']} {criterion['score']} points "
        dur_text += " ".join(duration["clinical_notes"])
        chunks.append({
            "section": "duration",
            "title": duration["title"],
            "content": duration,
            "text": dur_text
        })
        
        # Treatment context
        treatment = self.guidelines["treatment_context"]
        treat_text = f"{treatment['title']} {treatment['description']} DMARD biologic MTX "
        for cat_name, cat_data in treatment["categories"].items():
            treat_text += f"{cat_data['name']} {' '.join(cat_data['examples'])} {cat_data['interpretation']} "
        treat_text += " ".join(treatment["clinical_notes"])
        chunks.append({
            "section": "treatment_context",
            "title": treatment["title"],
            "content": treatment,
            "text": treat_text
        })
        
        # Interpretation
        interp = self.guidelines["interpretation_guide"]
        interp_text = f"{interp['title']} {interp['scoring_logic']['definite_RA']} "
        for k, v in interp["scoring_logic"]["borderline_cases"].items():
            interp_text += f"{k} {v} "
        interp_text += " ".join(interp["special_cases"])
        chunks.append({
            "section": "interpretation_guide",
            "title": interp["title"],
            "content": interp,
            "text": interp_text
        })
        
        return chunks
    
    def _build_bm25(self) -> BM25Okapi:
        """Build BM25 index"""
        corpus = [chunk["text"].lower() for chunk in self.chunks]
        tokenized = [doc.split() for doc in corpus]
        return BM25Okapi(tokenized)
    
    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve top-k guideline sections pertinentes
        
        Args:
            query: Query built from patient facts
            top_k: Number of sections to retrieve
        
        Returns:
            List of relevant guideline chunks with scores
        """
        q_tokens = query.lower().split()
        scores = self.bm25.get_scores(q_tokens)
        
        # Top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            
            chunk = self.chunks[idx].copy()
            chunk["relevance_score"] = float(scores[idx])
            results.append(chunk)
        
        return results
    
    def build_query_from_facts(self, patient_facts: Dict[str, Any]) -> str:
        """
        Construit query pour retrieval selon facts extraits
        """
        query_parts = []
        
        # Labs présents → query serology/inflammation
        labs = patient_facts.get("labs", []) or []
        has_rf = any("rf" in (l.get("test") or "").lower() for l in labs)
        has_ccp = any("ccp" in (l.get("test") or "").lower() or "acpa" in (l.get("test") or "").lower() for l in labs)
        has_crp = any("crp" in (l.get("test") or "").lower() for l in labs)
        has_esr = any("vhs" in (l.get("test") or "").lower() or "esr" in (l.get("test") or "").lower() for l in labs)
        
        if has_rf:
            query_parts.append("serology RF rheumatoid factor thresholds")
        if has_ccp:
            query_parts.append("serology anti-CCP ACPA thresholds")
        if has_crp:
            query_parts.append("acute phase CRP inflammation thresholds")
        if has_esr:
            query_parts.append("acute phase ESR VHS inflammation thresholds")
        
        # Diseases → query joint involvement
        diseases = patient_facts.get("disease_mentions", []) or []
        if any("arthrite" in (d.get("entity") or "").lower() or "polyarthrite" in (d.get("entity") or "").lower() for d in diseases):
            query_parts.append("joint involvement polyarthritis small joints MCPs PIPs")
        
        # Drugs → query treatment context
        drugs = patient_facts.get("drugs", []) or []
        if drugs:
            has_mtx = any("methotrexate" in (d.get("name") or "").lower() or "mtx" in (d.get("name") or "").lower() for d in drugs)
            has_biologic = any((d.get("category") or "") == "bDMARD" for d in drugs)
            has_jak = any((d.get("category") or "") == "tsDMARD" for d in drugs)
            
            if has_mtx:
                query_parts.append("treatment DMARD methotrexate")
            if has_biologic:
                query_parts.append("treatment biologic anti-TNF")
            if has_jak:
                query_parts.append("treatment JAK inhibitor tsDMARD")
        
        # Always include overview and interpretation
        query_parts.append("ACR EULAR classification criteria interpretation scoring")
        
        return " ".join(query_parts)
    
    def format_guidelines_for_prompt(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formate guidelines récupérées pour prompt LLM
        """
        if not retrieved_chunks:
            return "No specific guidelines retrieved."
        
        formatted = []
        
        for chunk in retrieved_chunks:
            section = chunk["section"]
            content = chunk["content"]
            
            if section == "overview":
                text = f"## {content['title']}\n"
                text += f"- {content['description']}\n"
                text += f"- Classification threshold: {content['classification_threshold']}\n"
                text += f"- Reference: {content['reference']}\n"
            
            elif section in ["joint_involvement", "serology", "acute_phase_reactants", "duration"]:
                text = f"## {content['title']} (Category {content.get('category', '?')}, max {content['max_score']} points)\n"
                text += f"**Description:** {content['description']}\n\n"
                
                text += "**Criteria:**\n"
                for criterion in content.get("criteria", []):
                    text += f"- {criterion['condition']}: **{criterion['score']} points**\n"
                    if criterion.get("details"):
                        text += f"  → {criterion['details']}\n"
                
                if "thresholds" in content:
                    text += "\n**Thresholds:**\n"
                    for k, v in content["thresholds"].items():
                        text += f"- {k}: {v}\n"
                
                if "clinical_notes" in content:
                    text += "\n**Clinical Notes:**\n"
                    for note in content["clinical_notes"]:
                        text += f"- {note}\n"
            
            elif section == "treatment_context":
                text = f"## {content['title']}\n"
                text += f"{content['description']}\n\n"
                for cat_name, cat_data in content["categories"].items():
                    text += f"**{cat_data['name']} ({cat_name}):**\n"
                    text += f"- Examples: {', '.join(cat_data['examples'])}\n"
                    text += f"- Interpretation: {cat_data['interpretation']}\n"
            
            elif section == "interpretation_guide":
                text = f"## {content['title']}\n"
                text += f"**Definite RA:** {content['scoring_logic']['definite_RA']}\n\n"
                text += "**Borderline Cases:**\n"
                for k, v in content["scoring_logic"]["borderline_cases"].items():
                    text += f"- {k}: {v}\n"
            
            else:
                text = f"## {chunk.get('title', 'Unknown Section')}\n{json.dumps(content, indent=2, ensure_ascii=False)}"
            
            formatted.append(text)
        
        return "\n\n".join(formatted)


# =============================================================================
# LLM CLIENT (Ollama)
# =============================================================================

class LLMClient:
    def __init__(self, url: str, model: str, temperature: float, timeout: int):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, json_mode: bool = True) -> Optional[str]:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": self.temperature,
            }
            if json_mode:
                payload["format"] = "json"

            r = requests.post(f"{self.url}/api/generate", json=payload, timeout=self.timeout)
            if r.status_code != 200:
                print(f"[LLM ERROR] HTTP {r.status_code}")
                return None
            data = r.json()
            return (data.get("response") or "").strip()
        except Exception as e:
            print(f"[LLM ERROR] {type(e).__name__}: {e}")
            return None


# =============================================================================
# AGENT 2 FINAL v3 (AVEC RAG-GUIDELINES)
# =============================================================================

class Agent2FinalWithRAG:
    """
    Agent 2 v3: LLM + RAG-Guidelines + Deterministic Guardrails
    
    Nouveautés:
    - RAG-Guidelines complet avec retrieval BM25
    - Guidelines structurées avec seuils cliniques
    - Explicability améliorée (citations guidelines)
    """

    # Heuristic ULN assumptions
    RF_ULN = 20.0
    CCP_ULN = 10.0

    def __init__(self):
        self.rag_retriever = RAGGuidelinesRetriever()
        self.llm = LLMClient(OLLAMA_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT)

    # --------------------------
    # Evidence helpers
    # --------------------------

    @staticmethod
    def _iter_evidence(patient_facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        evs: List[Dict[str, Any]] = []

        def pull(obj: Dict[str, Any]):
            ev = obj.get("evidence")
            if isinstance(ev, dict):
                evs.append(ev)
            elif isinstance(ev, list):
                for x in ev:
                    if isinstance(x, dict):
                        evs.append(x)

        for m in (patient_facts.get("disease_mentions") or []):
            if isinstance(m, dict):
                pull(m)
        for l in (patient_facts.get("labs") or []):
            if isinstance(l, dict):
                pull(l)
        for d in (patient_facts.get("drugs") or []):
            if isinstance(d, dict):
                pull(d)

        return evs

    @staticmethod
    def _has_confirmed_ra(patient_facts: Dict[str, Any]) -> bool:
        for m in (patient_facts.get("disease_mentions") or []):
            if not isinstance(m, dict):
                continue
            status = (m.get("status") or "").lower()
            ent = (m.get("entity") or "").lower()
            if status == "confirmed" and (
                "polyarthrite" in ent or "rhumato" in ent or "rheumatoid" in ent or ent in ["ra", "pr"]
            ):
                return True
        return False

    @staticmethod
    def _has_any_dmard(drugs: List[Dict[str, Any]]) -> bool:
        for d in drugs or []:
            if not isinstance(d, dict):
                continue
            cat = (d.get("category") or "")
            name = (d.get("name") or "").lower()
            if cat in ["csDMARD", "bDMARD", "tsDMARD"]:
                return True
            if "methotrexate" in name or "mtx" in name:
                return True
        return False

    @staticmethod
    def _has_biologic_or_jak(drugs: List[Dict[str, Any]]) -> bool:
        for d in drugs or []:
            if not isinstance(d, dict):
                continue
            cat = d.get("category")
            if cat in ["bDMARD", "tsDMARD"]:
                return True
        return False

    # --------------------------
    # Deterministic scoring
    # --------------------------

    def _parse_numeric(self, s: Any) -> Optional[float]:
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return float(s)
        txt = str(s)
        txt = txt.replace(",", ".")
        m = re.search(r"(-?\d+(?:\.\d+)?)", txt)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    def _compute_serology_points(self, patient_facts: Dict[str, Any]) -> int:
        labs = patient_facts.get("labs") or []

        rf_best = None
        ccp_best = None

        rf_pos_flag = False
        ccp_pos_flag = False

        for lab in labs:
            if not isinstance(lab, dict):
                continue
            test = (lab.get("test") or "").lower()
            polarity = (lab.get("polarity") or "").lower()
            value = lab.get("value")

            num = self._parse_numeric(value)

            if test in ["rf", "facteur rhumatoïde", "rheumatoid factor"]:
                if polarity in ["positive", "positif"]:
                    rf_pos_flag = True
                if num is not None:
                    rf_best = num if (rf_best is None) else max(rf_best, num)

            if ("ccp" in test) or ("acpa" in test) or ("anti-citrullin" in test):
                if polarity in ["positive", "positif"]:
                    ccp_pos_flag = True
                if num is not None:
                    ccp_best = num if (ccp_best is None) else max(ccp_best, num)

        rf_pts = 0
        ccp_pts = 0

        if rf_best is not None:
            if rf_best > 3 * self.RF_ULN:
                rf_pts = 3
            elif rf_best > self.RF_ULN:
                rf_pts = 2

        if ccp_best is not None:
            if ccp_best > 3 * self.CCP_ULN:
                ccp_pts = 3
            elif ccp_best > self.CCP_ULN:
                ccp_pts = 2

        if rf_best is None and rf_pos_flag:
            rf_pts = max(rf_pts, 2)
        if ccp_best is None and ccp_pos_flag:
            ccp_pts = max(ccp_pts, 2)

        return max(rf_pts, ccp_pts, 0)

    def _compute_acute_phase_points(self, patient_facts: Dict[str, Any]) -> int:
        labs = patient_facts.get("labs") or []
        for lab in labs:
            if not isinstance(lab, dict):
                continue
            test = (lab.get("test") or "").lower()
            polarity = (lab.get("polarity") or "").lower()
            val = self._parse_numeric(lab.get("value"))

            if polarity in ["positive", "positif"]:
                if any(k in test for k in ["crp", "esr", "vhs"]):
                    return 1

            if "crp" in test and val is not None and val >= 10:
                return 1
            if ("esr" in test or "vhs" in test) and val is not None and val >= 20:
                return 1
        return 0

    def _compute_joint_points(self, patient_facts: Dict[str, Any]) -> int:
        cues_small = [r"\bmcp\b", r"\bpip\b", r"\bpoignet", r"\bphalange", r"\bmtp\b", r"\bmétacarp", r"\bdoigt"]
        cues_synovitis = [r"\bsynovite\b", r"\béchographie\b.*\bsynov", r"\bépaississement\s+synov"]

        for ev in self._iter_evidence(patient_facts):
            snip = (ev.get("snippet") or "").lower()
            if any(re.search(p, snip) for p in cues_synovitis):
                return 2
            if any(re.search(p, snip) for p in cues_small):
                return 2

        return 0

    def _compute_duration_points(self, patient_facts: Dict[str, Any]) -> int:
        patterns = [
            r">\s*6\s*semaines",
            r"\b6\s*semaines\b",
            r"\b(\d+)\s*mois\b",
            r"\b(\d+)\s*ans\b",
            r"\bdepuis\s+\d+\s*(semaines|mois|ans)\b",
        ]

        for ev in self._iter_evidence(patient_facts):
            snip = (ev.get("snippet") or "").lower()
            for p in patterns:
                m = re.search(p, snip)
                if not m:
                    continue
                if "mois" in p and m.group(1).isdigit():
                    if int(m.group(1)) >= 2:
                        return 1
                else:
                    return 1

        n_stays = int(patient_facts.get("n_stays", 0) or 0)
        drugs = patient_facts.get("drugs") or []
        if self._has_any_dmard(drugs) and n_stays >= 2:
            return 1

        return 0

    def _deterministic_assessment(self, patient_facts: Dict[str, Any]) -> Dict[str, Any]:
        a = self._compute_joint_points(patient_facts)
        b = self._compute_serology_points(patient_facts)
        c = self._compute_acute_phase_points(patient_facts)
        d = self._compute_duration_points(patient_facts)
        total = a + b + c + d

        drugs = patient_facts.get("drugs") or []
        has_dmard = self._has_any_dmard(drugs)
        has_strong = self._has_biologic_or_jak(drugs)

        det_label = "RA+" if (total >= 6 or (total >= 5 and has_strong) or (total == 6 and has_dmard)) else "RA−"

        return {
            "component_scores_det": {
                "joint_involvement": a,
                "serology": b,
                "acute_phase": c,
                "duration": d,
            },
            "acr_eular_total_det": total,
            "det_label": det_label,
            "has_dmard_det": bool(has_dmard),
            "has_biologic_or_jak_det": bool(has_strong),
        }

    # --------------------------
    # Prompting avec RAG
    # --------------------------

    def _summarize_facts(self, patient_facts: Dict[str, Any]) -> str:
        disease_mentions = patient_facts.get("disease_mentions") or []
        labs = patient_facts.get("labs") or []
        drugs = patient_facts.get("drugs") or []

        lines = []
        lines.append("## Disease Mentions")
        if disease_mentions:
            for dm in disease_mentions:
                if not isinstance(dm, dict):
                    continue
                lines.append(f"- {dm.get('entity','?')} (status: {dm.get('status','?')})")
        else:
            lines.append("- None")

        lines.append("\n## Labs")
        if labs:
            for lab in labs:
                if not isinstance(lab, dict):
                    continue
                t = lab.get("test", "?")
                pol = lab.get("polarity", "?")
                v = lab.get("value", "")
                if v not in [None, ""]:
                    lines.append(f"- {t}: {pol} (value: {v})")
                else:
                    lines.append(f"- {t}: {pol}")
        else:
            lines.append("- None")

        lines.append("\n## Drugs")
        if drugs:
            for d in drugs:
                if not isinstance(d, dict):
                    continue
                lines.append(f"- {d.get('name','?')} (category: {d.get('category','?')})")
        else:
            lines.append("- None")

        return "\n".join(lines)

    def _build_prompt_with_rag(self, patient_facts: Dict[str, Any], retrieved_guidelines: str) -> str:
        patient_id = patient_facts.get("patient_id")
        facts_summary = self._summarize_facts(patient_facts)

        prompt = f"""You are an expert rheumatologist applying ACR/EULAR 2010 criteria for Rheumatoid Arthritis.

# PATIENT FACTS (patient_id={patient_id})
{facts_summary}

# RETRIEVED ACR/EULAR GUIDELINES
{retrieved_guidelines}

# TASK
Score categories A-D according to ACR/EULAR 2010 using ONLY the provided facts and guidelines.

**IMPORTANT RULES:**
- Do NOT invent joint counts or lab values
- If data is missing, use minimum plausible score
- Treatment (DMARD/biologic) is strong evidence of established RA
- Use the thresholds from guidelines for serology scoring

# DECISION LOGIC (refer to guidelines)
- total ≥ 7 → RA+
- total == 6 → RA+ if DMARD or confirmed RA mention, else RA−
- total == 5 → RA+ only if biologic/JAK OR confirmed RA, else RA−
- total < 5 → RA−

Return STRICT JSON:

{{
  "component_scores": {{
    "joint_involvement": {{"score": <0-5>, "justification": "..."}},
    "serology": {{"score": <0-3>, "justification": "..."}},
    "acute_phase": {{"score": <0-1>, "justification": "..."}},
    "duration": {{"score": <0-1>, "justification": "..."}}
  }},
  "acr_eular_total": <sum>,
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

    # --------------------------
    # Evidence trace
    # --------------------------

    def _extract_evidence_trace(self, patient_facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        evidence = []

        def ev_list(ev):
            if isinstance(ev, dict):
                return [ev]
            if isinstance(ev, list):
                return [x for x in ev if isinstance(x, dict)]
            return []

        for mention in patient_facts.get("disease_mentions", []) or []:
            if not isinstance(mention, dict):
                continue
            for ev in ev_list(mention.get("evidence")):
                evidence.append({
                    "type": "disease_mention",
                    "stay_id": ev.get("stay_id"),
                    "line_no": ev.get("line_no"),
                    "snippet": (ev.get("snippet") or "")[:160],
                })

        for lab in patient_facts.get("labs", []) or []:
            if not isinstance(lab, dict):
                continue
            for ev in ev_list(lab.get("evidence")):
                evidence.append({
                    "type": "lab",
                    "test": lab.get("test"),
                    "stay_id": ev.get("stay_id"),
                    "line_no": ev.get("line_no"),
                    "snippet": (ev.get("snippet") or "")[:160],
                })

        for drug in patient_facts.get("drugs", []) or []:
            if not isinstance(drug, dict):
                continue
            for ev in ev_list(drug.get("evidence")):
                evidence.append({
                    "type": "drug",
                    "name": drug.get("name"),
                    "stay_id": ev.get("stay_id"),
                    "line_no": ev.get("line_no"),
                    "snippet": (ev.get("snippet") or "")[:160],
                })

        return evidence

    # --------------------------
    # Main patient processing
    # --------------------------

    def process_patient(self, patient_facts: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        patient_id = str(patient_facts.get("patient_id", "")).strip()
        n_stays = int(patient_facts.get("n_stays", 0) or 0)
        print(f"[Agent 2 v3+RAG] Processing patient {patient_id} ({n_stays} stays)...")

        # 1. RAG: Build query and retrieve guidelines
        query = self.rag_retriever.build_query_from_facts(patient_facts)
        retrieved_chunks = self.rag_retriever.retrieve(query, top_k=4)
        retrieved_formatted = self.rag_retriever.format_guidelines_for_prompt(retrieved_chunks)
        
        print(f"[Agent 2 v3+RAG]   Retrieved {len(retrieved_chunks)} guideline sections")
        
        # 2. Build prompt with RAG context
        prompt = self._build_prompt_with_rag(patient_facts, retrieved_formatted)
        
        # 3. LLM generation
        llm_text = self.llm.generate(prompt, json_mode=True)

        if not llm_text:
            assessment = self._fallback(patient_id, reason="LLM empty response")
        else:
            try:
                assessment = json.loads(llm_text)
            except json.JSONDecodeError as e:
                assessment = self._fallback(patient_id, reason=f"LLM JSON parse error: {e}")

        # Normalize
        assessment.setdefault("patient_id", patient_id)
        assessment.setdefault("agent", "Agent2_v3_RAG_Guidelines")
        assessment.setdefault("component_scores", {})
        assessment.setdefault("acr_eular_total", 0)
        assessment.setdefault("final_label", "RA−")
        assessment.setdefault("confidence", "low")
        assessment.setdefault("classification_rationale", "")

        # Add RAG metadata
        assessment["rag_guidelines"] = {
            "query": query,
            "sections_retrieved": [chunk["section"] for chunk in retrieved_chunks],
            "relevance_scores": [chunk["relevance_score"] for chunk in retrieved_chunks]
        }
        assessment["guideline_citation"] = "Aletaha D, et al. Arthritis Rheum. 2010;62(9):2569-2581"
        assessment["evidence_trace"] = self._extract_evidence_trace(patient_facts)

        # Deterministic check
        det = self._deterministic_assessment(patient_facts)
        assessment.update(det)

        # Treatment context
        drugs = patient_facts.get("drugs") or []
        has_dmard = self._has_any_dmard(drugs)
        has_strong = self._has_biologic_or_jak(drugs)
        if "treatment_context" not in assessment or not isinstance(assessment["treatment_context"], dict):
            assessment["treatment_context"] = {}
        assessment["treatment_context"].setdefault("has_dmard", bool(has_dmard))
        assessment["treatment_context"].setdefault("treatment_strength", "strong" if has_strong else ("moderate" if has_dmard else "none"))

        # Override policy
        llm_label = assessment.get("final_label", "RA−")
        det_label = det["det_label"]
        det_total = det["acr_eular_total_det"]
        det_serology = det["component_scores_det"]["serology"]
        has_confirmed = self._has_confirmed_ra(patient_facts)

        assessment["llm_vs_det_disagree"] = (llm_label != det_label)

        if (
            llm_label == "RA+"
            and det_label == "RA−"
            and det_total <= 3
            and det_serology == 0
            and (not det["has_dmard_det"])
            and (not det["has_biologic_or_jak_det"])
            and (not has_confirmed)
        ):
            assessment["final_label"] = "RA−"
            assessment["confidence"] = "low"
            assessment["classification_rationale"] = (
                (assessment.get("classification_rationale", "") + " | ").strip()
                + f"OVERRIDE: Very unlikely RA (det_total={det_total}, no serology, no DMARD)."
            )
            assessment["override_applied"] = True
        else:
            assessment["override_applied"] = False

        # Confidence fallback
        conf = (assessment.get("confidence") or "").lower()
        if conf not in ["low", "medium", "high"]:
            score = assessment.get("acr_eular_total", 0) or 0
            if score >= 7 and has_dmard:
                assessment["confidence"] = "high"
            elif score >= 6:
                assessment["confidence"] = "medium"
            else:
                assessment["confidence"] = "low"

        # Save
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"criteria_assessment_{patient_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(assessment, f, indent=2, ensure_ascii=False)

        print(f"[Agent 2 v3+RAG]   LLM={llm_label} det={det_label} det_total={det_total} override={assessment['override_applied']}")
        print(f"[Agent 2 v3+RAG]   ✓ Saved label={assessment['final_label']} score={assessment.get('acr_eular_total', 0)}")

        return assessment

    def _fallback(self, patient_id: str, reason: str) -> Dict[str, Any]:
        return {
            "patient_id": str(patient_id),
            "agent": "Agent2_v3_RAG_Guidelines_FALLBACK",
            "component_scores": {},
            "acr_eular_total": 0,
            "final_label": "RA−",
            "confidence": "low",
            "classification_rationale": f"Fallback: {reason}",
            "treatment_context": {},
            "rag_guidelines": {"query": "", "sections_retrieved": [], "relevance_scores": []},
            "evidence_trace": [],
            "component_scores_det": {"joint_involvement": 0, "serology": 0, "acute_phase": 0, "duration": 0},
            "acr_eular_total_det": 0,
            "det_label": "RA−",
            "has_dmard_det": False,
            "has_biologic_or_jak_det": False,
            "llm_vs_det_disagree": False,
            "override_applied": False,
        }

    def process_jsonl(self, jsonl_path: str, output_dir: str) -> List[Dict[str, Any]]:
        jsonl_path = Path(jsonl_path)
        out = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    patient_facts = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[Agent 2 v3+RAG] ERROR line {i}: {e}")
                    continue

                try:
                    out.append(self.process_patient(patient_facts, output_dir))
                except Exception as e:
                    pid = str(patient_facts.get("patient_id", ""))
                    print(f"[Agent 2 v3+RAG] ERROR patient {pid}: {type(e).__name__}: {e}")
        print(f"\n[Agent 2 v3+RAG] ✓ Done: {len(out)} patients -> {output_dir}")
        return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("AGENT 2 v3: LLM + RAG-Guidelines + Deterministic Guardrails")
    print("=" * 80)
    print(f"LLM model: {LLM_MODEL}")
    print(f"Input:  {FACTS_PATIENT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"RAG: BM25 retrieval sur guidelines ACR/EULAR structurées\n")

    if not Path(FACTS_PATIENT_PATH).exists():
        print(f"ERROR: facts file not found: {FACTS_PATIENT_PATH}")
        return

    agent = Agent2FinalWithRAG()
    agent.process_jsonl(FACTS_PATIENT_PATH, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("✓ Agent 2 v3 complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()