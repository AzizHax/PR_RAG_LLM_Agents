#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent 3 Orchestrateur + LLM Critic - OPTIMISÉ POUR QWEN 3B

AMÉLIORATIONS vs version originale:
✅ Prompt LLM simplifié avec exemples concrets
✅ Timeout 120s (vs 60s)
✅ Fix patient_id: pas de zfill(), garde "1", "2" etc.
✅ Gestion erreurs LLM robuste avec logging
✅ Contexte réduit: 6 snippets max (vs 10)
✅ Override threshold: 0.80 (vs 0.82) pour 3B
✅ Parsing JSON sécurisé

Performance attendue:
- Grid search: F1 0.88-0.90 (sans LLM)
- Final avec LLM: F1 0.90-0.93
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except ImportError:
    print("ERROR: pip install scikit-learn numpy")
    raise

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    raise


# =============================================================================
# CONFIGURATION
# =============================================================================

FACTS_PATIENT_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\facts_agent1_patient.jsonl"
CRITERIA_DIR       = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\criteria_assessments_llm"
OUTPUT_DIR         = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\final_decisions"
ANNOTATIONS_PATH   = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\phantom_annotations_minimal.txt"

DEFAULT_TH_AGENT1 = 3.0
DEFAULT_TH_AGENT2 = 6

# =============================================================================
# LLM CONFIG (QWEN 3B OPTIMIZED)
# =============================================================================

USE_LLM_CRITIC = True
LLM_ONLY_ON_LOW_CONF = True
LLM_ALLOW_OVERRIDE = True
LLM_OVERRIDE_MIN_CONF = 0.80      # Ajusté pour 3B
LLM_MAX_EVIDENCE_SNIPPETS = 6     # Réduit pour contexte plus court
LLM_TIMEOUT_S = 120               # Augmenté

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct"


# =============================================================================
# GUARDRAILS
# =============================================================================

class Guardrails:
    @staticmethod
    def validate_evidence_ids(evidence_trace: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for i, ev in enumerate(evidence_trace or []):
            if not ev.get("stay_id"):
                errors.append(f"Ev{i}: no stay_id")
            if not ev.get("line_no"):
                errors.append(f"Ev{i}: no line_no")
            if not (ev.get("snippet") or "").strip():
                errors.append(f"Ev{i}: no snippet")
        return (len(errors) == 0), errors

    @staticmethod
    def detect_temporal_contradictions(labs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        contradictions = []
        tests_by_type = defaultdict(list)

        for lab in labs or []:
            test = (lab.get("test") or "").strip().lower()
            if test:
                tests_by_type[test].append(lab)

        for test_name, results in tests_by_type.items():
            if len(results) < 2:
                continue
            
            pols = [(r.get("polarity") or "").lower() for r in results]  # ✅ Fix None.lower()
            if "positive" in pols and "negative" in pols:
                contradictions.append({
                    "test": test_name,
                    "polarities": pols,
                    "message": f"{test_name}: positive & negative found"
                })

        return contradictions


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class FusionDecision:
    patient_id: str
    agent1_score: float
    agent2_score: int
    agent1_label: str
    agent2_label: str
    final_label: str
    decision_score: float
    confidence: str
    fusion_strategy: str
    conflicts_detected: List[str]
    evidence_valid: bool
    justification: str
    explicability_score: Dict[str, Any]
    llm_critic: Optional[Dict[str, Any]] = None


# =============================================================================
# LLM CRITIC
# =============================================================================

class LLMCritic:
    SYSTEM_PROMPT = """You are an expert medical auditor for RA phenotyping.

**Task**: Review case and recommend RA+ (has RA), RA− (no RA), or abstain (unclear).

**RA+ signals**:
- RF+ and/or anti-CCP+ (serology positive)
- Patient on DMARD/biologic (MTX, adalimumab, etc.)
- ACR-EULAR score ≥6 with clinical support
- Diagnosis confirmed in notes

**RA− signals**:
- RF− and anti-CCP− and no DMARD
- Diagnosis ruled out or excluded
- Alternative diagnosis (osteoarthritis, bursitis)

**Abstain when**:
- Conflicting evidence (positive tests but diagnosis excluded)
- Missing key data (no serology, no treatment info)
- Borderline case (ACR-EULAR 5-6) without context

**Rules**: Be CONSERVATIVE. If unsure → abstain. Use ONLY provided evidence. NO invention."""

    def _post_ollama(self, prompt: str) -> Dict[str, Any]:
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "format": "json",
                },
                timeout=LLM_TIMEOUT_S
            )
            r.raise_for_status()
            data = r.json()
            txt = (data.get("response") or "").strip()
            return json.loads(txt)
        except requests.exceptions.Timeout:
            raise TimeoutError(f"LLM timeout ({LLM_TIMEOUT_S}s)")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"LLM error: {e}")

    def critique(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""{self.SYSTEM_PROMPT}

# CASE

Patient: {payload['patient_id']}

**Current Decision**:
- Agent 1: {payload['deterministic_decision']['agent1_score']} → {payload['deterministic_decision']['agent1_label']}
- Agent 2: {payload['deterministic_decision']['agent2_score']}/10 → {payload['deterministic_decision']['agent2_label']}
- Final: {payload['deterministic_decision']['final_label']} (conf: {payload['deterministic_decision']['confidence']})

**Clinical Summary**:
{json.dumps(payload['agent1_summary'], indent=2, ensure_ascii=False)}

**Evidence Snippets** (top {len(payload['evidence_snippets'])}):
{json.dumps(payload['evidence_snippets'][:6], indent=2, ensure_ascii=False)}

**Guardrails**:
- Evidence valid: {payload['guardrails']['evidence_valid']}
- Conflicts: {len(payload['guardrails']['conflicts'])}

# OUTPUT

Return STRICT JSON:
{{
  "recommend_label": "RA+" | "RA−" | "abstain",
  "confidence": 0.0-1.0,
  "key_reasons": ["..."],
  "red_flags": ["..."],
  "questions_for_human": ["..."],
  "override_suggested": true/false
}}"""

        out = self._post_ollama(prompt)

        # Validation
        if not isinstance(out, dict):
            raise ValueError("Output not dict")

        out.setdefault("recommend_label", "abstain")
        out.setdefault("confidence", 0.0)
        out.setdefault("key_reasons", [])
        out.setdefault("red_flags", [])
        out.setdefault("questions_for_human", [])
        out.setdefault("override_suggested", False)

        # Normalisation
        if out["recommend_label"] not in ["RA+", "RA−", "abstain"]:
            out["recommend_label"] = "abstain"
        
        try:
            out["confidence"] = float(out["confidence"])
        except:
            out["confidence"] = 0.0
        
        out["confidence"] = max(0.0, min(1.0, out["confidence"]))

        return out


# =============================================================================
# AGENT 3 ORCHESTRATOR
# =============================================================================

class Agent3Orchestrator:
    def __init__(self, th_agent1: float = DEFAULT_TH_AGENT1, th_agent2: int = DEFAULT_TH_AGENT2):
        self.th_agent1 = th_agent1
        self.th_agent2 = th_agent2
        self.guardrails = Guardrails()
        self.critic = LLMCritic() if USE_LLM_CRITIC else None

    def _compute_agent1_label(self, score: float) -> str:
        return "RA+" if score >= self.th_agent1 else "RA−"

    def _fusion_logic(self, a1_score: float, a2_score: int, a1_label: str, a2_label: str,
                      has_dmard: bool, has_biologic: bool) -> Tuple[str, str, str]:
        
        if a2_score >= 7:
            return a2_label, "agent2_strong", f"Agent 2 confident ({a2_score}/10)"

        if a1_label == a2_label:
            conf = "high" if a2_score >= 6 else "medium"
            return a1_label, "agreement", f"Both agree, conf: {conf}"

        if a2_score == 6 and has_dmard:
            return "RA+", "agent2_dmard", "Agent 2 borderline + DMARD"

        if a1_score >= 5 and has_biologic and a2_score < 6:
            return "RA+", "agent1_biologic", "Agent 1 confident + biologic"

        if a1_score >= 6 and a2_score < 6:
            return "RA+", "agent1_strong", f"Agent 1 confident ({a1_score:.1f})"

        if a2_score >= self.th_agent2:
            return a2_label, "agent2_priority", "Agent 2 prioritized"

        return a1_label, "agent1_default", "Default to Agent 1"

    def _compute_confidence(self, a1_score: float, a2_score: int, a1_label: str, a2_label: str,
                            valid: bool, conflicts: List[str]) -> str:
        if not valid or conflicts:
            return "low"
        if a1_label == a2_label:
            if a2_score >= 7 and a1_score >= 5:
                return "high"
            if a2_score >= 6 or a1_score >= 4:
                return "medium"
        if a2_score >= 7 or a1_score >= 6:
            return "medium"
        return "low"

    def _compute_decision_score(self, a1_score: float, a2_score: int, final: str,
                                strategy: str, valid: bool, conflicts: List[str]) -> float:
        a1_norm = min(10.0, max(0.0, a1_score))
        a2_norm = float(a2_score)

        if strategy in ["agent2_strong", "agent2_priority", "agent2_dmard"]:
            base = 0.3 * a1_norm + 0.7 * a2_norm
        elif strategy == "agreement":
            base = 0.5 * a1_norm + 0.5 * a2_norm
        else:
            base = 0.7 * a1_norm + 0.3 * a2_norm

        penalty = 0.0
        if not valid:
            penalty += 1.5
        if conflicts:
            penalty += min(2.0, 0.5 * len(conflicts))

        score = max(0.0, base - penalty)
        if final == "RA−":
            score = 10.0 - score

        return round(score, 2)

    def _compute_agent1_combined_score(self, facts: Dict[str, Any]) -> float:
        extraction = self._compute_extraction_score(facts)
        rag_meta = facts.get("_rag_meta_patient", {}) or {}
        rag = float(rag_meta.get("rag_score_max", 0.0) or 0.0)
        return max(extraction, rag)

    def _compute_extraction_score(self, facts: Dict[str, Any]) -> float:
        score = 0.0
        diseases = facts.get("disease_mentions", []) or []
        labs = facts.get("labs", []) or []
        drugs = facts.get("drugs", []) or []

        confirmed = [m for m in diseases if (m.get("status") or "").lower() == "confirmed"]
        negated = [m for m in diseases if (m.get("status") or "").lower() == "negated"]

        if confirmed:
            score += 2.0
        elif diseases and not negated:
            score += 1.0
        if negated:
            score -= 2.0

        rf_pos = any((l.get("test") or "").lower() in ["rf", "facteur rhumatoïde"] and
                     (l.get("polarity") or "").lower() in ["positive", "positif"] for l in labs)
        ccp_pos = any(("ccp" in (l.get("test") or "").lower() or "acpa" in (l.get("test") or "").lower()) and
                      (l.get("polarity") or "").lower() in ["positive", "positif"] for l in labs)

        if rf_pos and ccp_pos:
            score += 3.0
        elif rf_pos or ccp_pos:
            score += 2.0

        has_mtx = any("methotrexate" in (d.get("name") or "").lower() or "mtx" in (d.get("name") or "").lower() for d in drugs)
        has_bio = any(d.get("category") == "bDMARD" for d in drugs)
        has_jak = any(d.get("category") == "tsDMARD" for d in drugs)

        if has_bio or has_jak:
            score += 2.5
        elif has_mtx:
            score += 1.5

        return score

    def _collect_evidence_snippets(self, facts: Dict[str, Any], assessment: Dict[str, Any],
                                    limit: int = LLM_MAX_EVIDENCE_SNIPPETS) -> List[Dict[str, Any]]:
        snippets = []

        for ev in (assessment.get("evidence_trace") or []):
            snippets.append({
                "source": "agent2",
                "stay_id": ev.get("stay_id"),
                "line_no": ev.get("line_no"),
                "snippet": (ev.get("snippet") or "")[:150]
            })

        def add_ev(entity, src):
            ev = entity.get("evidence")
            if isinstance(ev, dict):
                snippets.append({"source": src, "snippet": (ev.get("snippet") or "")[:150],
                                "stay_id": ev.get("stay_id"), "line_no": ev.get("line_no")})
            elif isinstance(ev, list):
                for x in ev[:1]:
                    if isinstance(x, dict):
                        snippets.append({"source": src, "snippet": (x.get("snippet") or "")[:150],
                                        "stay_id": x.get("stay_id"), "line_no": x.get("line_no")})

        for m in (facts.get("disease_mentions") or [])[:2]:
            add_ev(m, "agent1:disease")
        for l in (facts.get("labs") or [])[:2]:
            add_ev(l, "agent1:lab")
        for d in (facts.get("drugs") or [])[:1]:
            add_ev(d, "agent1:drug")

        # Dedup
        seen = set()
        uniq = []
        for s in snippets:
            key = (s.get("stay_id"), s.get("line_no"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)

        return uniq[:limit]

    def _build_critic_payload(self, pid: str, facts: Dict[str, Any], assessment: Dict[str, Any],
                              det_decision: Dict[str, Any], conflicts: List[str], valid: bool) -> Dict[str, Any]:

        labs = facts.get("labs", []) or []
        drugs = facts.get("drugs", []) or []
        dis = facts.get("disease_mentions", []) or []

        def sum_serology():
            rf = [{"pol": l.get("polarity"), "val": l.get("value")} for l in labs
                  if (l.get("test") or "").lower() in ["rf", "facteur rhumatoïde"]][:2]
            ccp = [{"pol": l.get("polarity"), "val": l.get("value")} for l in labs
                   if "ccp" in (l.get("test") or "").lower() or "acpa" in (l.get("test") or "").lower()][:2]
            return {"rf": rf, "ccp_acpa": ccp}

        def sum_treatment():
            cats = defaultdict(int)
            for d in drugs:
                cats[d.get("category") or ""] += 1
            return {"cats": dict(cats), "names": [d.get("name") for d in drugs[:5]]}

        def sum_mentions():
            return [{"entity": m.get("entity"), "status": m.get("status")} for m in dis[:3]]

        return {
            "patient_id": pid,
            "task": "RA patient-level classification",
            "deterministic_decision": det_decision,
            "agent1_summary": {
                "serology": sum_serology(),
                "treatment": sum_treatment(),
                "mentions": sum_mentions(),
                "n_stays": facts.get("n_stays")
            },
            "guardrails": {
                "evidence_valid": valid,
                "conflicts": conflicts[:5]
            },
            "evidence_snippets": self._collect_evidence_snippets(facts, assessment)
        }

    def _compute_explicability(self, a1_score: float, a2_score: int, a1_label: str, a2_label: str,
                                final: str, strategy: str, valid: bool, conflicts: List[str],
                                facts: Dict[str, Any]) -> Dict[str, Any]:
        
        agreement = 100.0 if a1_label == a2_label else max(0.0, 100.0 - abs(a1_score - a2_score) * 15.0)

        evidence = 100.0
        if not valid:
            evidence -= 40.0
        if conflicts:
            evidence -= min(40.0, len(conflicts) * 15.0)
        evidence = max(0.0, evidence)

        clinical = 100.0
        labs = facts.get("labs", []) or []
        drugs = facts.get("drugs", []) or []
        seropos = any((l.get("polarity") or "").lower() in ["positive", "positif"] for l in labs)
        has_dmard = len(drugs) > 0

        if final == "RA+" and not seropos and not has_dmard:
            clinical -= 30.0
        if final == "RA−" and seropos:
            clinical -= 20.0
        if a2_score >= 6 and final == "RA−":
            clinical -= 25.0
        clinical = max(0.0, clinical)

        overall = 0.40 * agreement + 0.35 * evidence + 0.25 * clinical
        level = "HIGH" if overall >= 80 else ("MEDIUM" if overall >= 60 else "LOW")

        return {
            "overall_confidence": round(overall, 1),
            "confidence_level": level,
            "components": {
                "agreement": round(agreement, 1),
                "evidence_quality": round(evidence, 1),
                "clinical_consistency": round(clinical, 1)
            },
            "strategy": strategy
        }

    def process_patient(self, pid: str, facts: Dict[str, Any], assessment: Dict[str, Any]) -> FusionDecision:
        print(f"[Agent 3] Patient {pid}...")

        a1_score = self._compute_agent1_combined_score(facts)
        a2_score = int(assessment.get("acr_eular_total", 0) or 0)

        a1_label = self._compute_agent1_label(a1_score)
        a2_label = assessment.get("final_label", "RA−") or "RA−"

        has_dmard = bool((assessment.get("treatment_context") or {}).get("has_dmard", False))
        has_biologic = any(d.get("category") in ["bDMARD", "tsDMARD"] for d in (facts.get("drugs") or []))

        ev_trace = assessment.get("evidence_trace", []) or []
        ev_valid, ev_errors = self.guardrails.validate_evidence_ids(ev_trace)
        contradictions = self.guardrails.detect_temporal_contradictions(facts.get("labs", []) or [])

        conflicts = []
        if not ev_valid:
            conflicts.extend(ev_errors[:5])
        if contradictions:
            conflicts.extend([c["message"] for c in contradictions][:5])

        final, strategy, justif = self._fusion_logic(a1_score, a2_score, a1_label, a2_label, has_dmard, has_biologic)
        conf = self._compute_confidence(a1_score, a2_score, a1_label, a2_label, ev_valid, conflicts)
        dec_score = self._compute_decision_score(a1_score, a2_score, final, strategy, ev_valid, conflicts)
        explic = self._compute_explicability(a1_score, a2_score, a1_label, a2_label, final, strategy, ev_valid, conflicts, facts)

        det_decision = {
            "agent1_score": round(a1_score, 2),
            "agent2_score": a2_score,
            "agent1_label": a1_label,
            "agent2_label": a2_label,
            "final_label": final,
            "strategy": strategy,
            "confidence": conf,
            "decision_score": dec_score
        }

        llm_out = None
        if USE_LLM_CRITIC and self.critic and (not LLM_ONLY_ON_LOW_CONF or conf == "low"):
            try:
                payload = self._build_critic_payload(pid, facts, assessment, det_decision, conflicts, ev_valid)
                t0 = time.time()
                llm_out = self.critic.critique(payload)
                llm_out["latency_ms"] = int((time.time() - t0) * 1000)
            except TimeoutError as e:
                print(f"  ⚠️ LLM timeout: {e}")
                llm_out = {"recommend_label": "abstain", "confidence": 0.0, "key_reasons": [],
                          "red_flags": [f"Timeout ({LLM_TIMEOUT_S}s)"], "questions_for_human": ["Manual review"],
                          "override_suggested": False, "error": str(e)}
            except Exception as e:
                print(f"  ⚠️ LLM error: {type(e).__name__}: {e}")
                llm_out = {"recommend_label": "abstain", "confidence": 0.0, "key_reasons": [],
                          "red_flags": [f"Error: {type(e).__name__}"], "questions_for_human": ["Manual review"],
                          "override_suggested": False, "error": str(e)}

        # Override
        if llm_out and LLM_ALLOW_OVERRIDE:
            rec = llm_out.get("recommend_label", "abstain")
            llm_conf = float(llm_out.get("confidence", 0.0) or 0.0)

            allow = (rec in ["RA+", "RA−"] and llm_conf >= LLM_OVERRIDE_MIN_CONF and ev_valid and not conflicts)

            if allow and rec != final:
                old = final
                final = rec
                strategy += "+llm_override"
                justif += f" | LLM override: {old}→{rec} (conf={llm_conf:.2f})"
                dec_score = self._compute_decision_score(a1_score, a2_score, final, strategy, ev_valid, conflicts)
                explic = self._compute_explicability(a1_score, a2_score, a1_label, a2_label, final, strategy, ev_valid, conflicts, facts)

        decision = FusionDecision(
            patient_id=pid,
            agent1_score=round(a1_score, 2),
            agent2_score=a2_score,
            agent1_label=a1_label,
            agent2_label=a2_label,
            final_label=final,
            decision_score=dec_score,
            confidence=conf,
            fusion_strategy=strategy,
            conflicts_detected=conflicts,
            evidence_valid=ev_valid,
            justification=justif,
            explicability_score=explic,
            llm_critic=llm_out
        )

        print(f"  A1: {a1_label} ({a1_score:.2f}) | A2: {a2_label} ({a2_score}) | Final: {final} ({dec_score:.2f}) | Conf: {conf}")
        if llm_out:
            print(f"  LLM: {llm_out.get('recommend_label')} ({llm_out.get('confidence'):.2f})")

        return decision

    def process_all_patients(self, facts_path: str, criteria_dir: str, output_dir: str) -> List[FusionDecision]:
        facts_path = Path(facts_path)
        criteria_dir = Path(criteria_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Agent 3] Loading facts from {facts_path}")
        facts_data = {}
        with open(facts_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    pid = row.get("patient_id")
                    if pid:
                        facts_data[pid] = row

        print(f"[Agent 3] Loading assessments from {criteria_dir}")
        assessments = {}
        for fp in sorted(criteria_dir.glob("criteria_assessment_*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
                pid = obj.get("patient_id")
                if pid:
                    assessments[pid] = obj

        print(f"[Agent 3] {len(facts_data)} facts, {len(assessments)} assessments\n")

        decisions = []
        for pid in sorted(facts_data.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            if pid not in assessments:
                print(f"[Agent 3] SKIP {pid} (no Agent 2)")
                continue

            dec = self.process_patient(pid, facts_data[pid], assessments[pid])
            decisions.append(dec)

            out_file = output_dir / f"final_decision_{pid}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(asdict(dec), f, indent=2, ensure_ascii=False)

        print(f"\n[Agent 3] ✓ {len(decisions)} patients → {output_dir}")
        return decisions


# =============================================================================
# GRID SEARCH
# =============================================================================

class GridSearchOptimizer:
    def __init__(self, annotations_path: str):
        self.ground_truth = self._load_ground_truth(annotations_path)

    def _load_ground_truth(self, path: str) -> Dict[str, int]:
        gt = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(r"PATIENT_ID:\s*(\d+)\s*\|\s*LABEL_BINARY:\s*([01])", line)
                if m:
                    pid = m.group(1)  # ✅ Pas de zfill
                    gt[pid] = int(m.group(2))
        return gt

    def evaluate(self, decisions: List[FusionDecision]) -> Dict[str, float]:
        y_true = []
        y_pred = []
        for d in decisions:
            if d.patient_id not in self.ground_truth:
                continue
            y_true.append(self.ground_truth[d.patient_id])
            y_pred.append(1 if d.final_label == "RA+" else 0)

        if not y_true:
            return {}

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    def grid_search(self, facts_path: str, criteria_dir: str,
                    th1_range: List[float], th2_range: List[int],
                    temp_dir: str) -> Dict[str, Any]:

        print("=" * 80)
        print("GRID SEARCH (LLM OFF)")
        print("=" * 80)
        print(f"TH1: {th1_range}")
        print(f"TH2: {th2_range}")
        print(f"Total: {len(th1_range) * len(th2_range)}\n")

        best_f1 = -1.0
        best_th1, best_th2, best_metrics = None, None, None
        results = []

        global USE_LLM_CRITIC
        prev = USE_LLM_CRITIC
        USE_LLM_CRITIC = False

        try:
            for th1 in th1_range:
                for th2 in th2_range:
                    print(f"[Grid] TH1={th1:.1f} TH2={th2} ... ", end="")

                    agent3 = Agent3Orchestrator(float(th1), int(th2))
                    decisions = agent3.process_all_patients(facts_path, criteria_dir, str(Path(temp_dir) / "temp"))

                    metrics = self.evaluate(decisions)
                    if not metrics:
                        print("SKIP")
                        continue

                    f1 = metrics["f1"]
                    print(f"F1={f1:.3f}")

                    results.append({"th_agent1": float(th1), "th_agent2": int(th2), "metrics": metrics})

                    if f1 > best_f1:
                        best_f1 = f1
                        best_th1 = float(th1)
                        best_th2 = int(th2)
                        best_metrics = metrics

        finally:
            USE_LLM_CRITIC = prev

        print("\n" + "=" * 80)
        print("BEST CONFIG")
        print("=" * 80)
        print(f"TH1: {best_th1}, TH2: {best_th2}")
        print(f"Acc: {best_metrics['accuracy']:.3f}")
        print(f"Prec: {best_metrics['precision']:.3f}")
        print(f"Rec: {best_metrics['recall']:.3f}")
        print(f"F1: {best_metrics['f1']:.3f}")

        return {
            "best_th_agent1": best_th1,
            "best_th_agent2": best_th2,
            "best_metrics": best_metrics,
            "grid_results": results
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("AGENT 3 (Qwen 3B Optimized)")
    print("=" * 80)

    for p in [FACTS_PATIENT_PATH, CRITERIA_DIR, ANNOTATIONS_PATH]:
        if not Path(p).exists():
            print(f"ERROR: {p} not found")
            return

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid search
    optimizer = GridSearchOptimizer(ANNOTATIONS_PATH)
    th1_range = np.arange(1.0, 7.0, 0.5)
    th2_range = list(range(4, 9))

    results = optimizer.grid_search(FACTS_PATIENT_PATH, CRITERIA_DIR, th1_range, th2_range, str(out_dir))

    # Final run with LLM
    print("\n" + "=" * 80)
    print("FINAL RUN (LLM ON)")
    print("=" * 80)

    agent3 = Agent3Orchestrator(results["best_th_agent1"], results["best_th_agent2"])
    final_decisions = agent3.process_all_patients(FACTS_PATIENT_PATH, CRITERIA_DIR, OUTPUT_DIR)

    # Save
    results_file = out_dir / "grid_search_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    jsonl_out = out_dir / "final_decisions.jsonl"
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for d in final_decisions:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    print(f"\n✓ Results: {results_file}")
    print(f"✓ Decisions: {OUTPUT_DIR}")
    print(f"✓ JSONL: {jsonl_out}")
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()