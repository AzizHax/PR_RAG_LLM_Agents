import os
import re
import json
import time
import logging
from math import exp
from typing import List, Dict, Any, Optional, Tuple
import requests

# =========================
# LOGGING + TIMERS
# =========================
def setup_logger(log_path: str = "agent1_run.log") -> logging.Logger:
    logger = logging.getLogger("agent1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

def ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# =========================
# OLLAMA LLM INTEGRATION
# =========================
class OllamaLLM:
    """Interface pour interagir avec Ollama et le modèle Qwen 2.5:3b-instruct"""
    
    def __init__(self, model_name: str = "qwen2.5:3b-instruct", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
        
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """Génère une réponse à partir d'un prompt"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Erreur Ollama: {e}")
            return ""
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1, 
             max_tokens: int = 1000) -> str:
        """Utilise l'API chat pour des conversations multi-tours"""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            response = requests.post(self.chat_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Erreur Ollama chat: {e}")
            return ""
    
    def extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extrait du JSON d'une réponse qui peut contenir du texte additionnel"""
        try:
            # Cherche un bloc JSON dans la réponse
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)
        except json.JSONDecodeError:
            return None

# =========================
# SPLIT CORPUS
# =========================
PATIENT_RE = re.compile(r"^PATIENT_ID:\s*(\d+)\s*$", re.MULTILINE)
STAY_RE = re.compile(r"^===\s*STAY_ID:\s*([A-Za-z0-9_-]+)\s*\|", re.MULTILINE)

def split_patients(corpus: str) -> List[Dict[str, str]]:
    matches = list(PATIENT_RE.finditer(corpus))
    out = []
    for i, m in enumerate(matches):
        pid = m.group(1)
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(corpus)
        chunk = corpus[start:end].strip()
        out.append({"patient_id": pid, "text": chunk})
    return out

def extract_stays(text: str) -> List[Dict[str, str]]:
    stay_matches = list(STAY_RE.finditer(text))
    if not stay_matches:
        return [{"stay_id": None, "text": text}]
    stays = []
    for i, m in enumerate(stay_matches):
        sid = m.group(1)
        start = m.start()
        end = stay_matches[i+1].start() if i+1 < len(stay_matches) else len(text)
        stays.append({"stay_id": sid, "text": text[start:end]})
    return stays

# =========================
# CUES + POLARITY
# =========================
NEGATION_CUES = ["écartée", "non retenue", "pas d'argument", "pas d'argument", "absence de", "exclue", "exclu", "pas de"]
UNCERTAIN_CUES = ["suspicion", "probable", "à discuter", "a discuter", "possible", "bilan en cours", "à confirmer", "a confirmer"]
CONFIRM_CUES = ["diagnostic retenu", "pr connue", "polyarthrite rhumatoïde (pr)", "rhumatoïde (pr)", "rheumatoid arthritis"]

def cue_status(context: str) -> str:
    ctx = (context or "").lower()
    if any(c in ctx for c in NEGATION_CUES):
        return "negated"
    if any(c in ctx for c in CONFIRM_CUES):
        return "confirmed"
    if any(c in ctx for c in UNCERTAIN_CUES):
        return "suspected"
    return "mentioned"

def polarity_from_line(line: str) -> Optional[str]:
    l = (line or "").lower()
    if "posit" in l or " +" in l or l.strip().endswith("+"):
        return "positive"
    if "negat" in l or " -" in l or l.strip().endswith("-"):
        return "negative"
    return None

# =========================
# SAFE ANCHORS + RAG HYBRID
# =========================
RA_LONG = re.compile(r"\b(polyarthrite rhumato[iï]de|rheumatoid arthritis)\b", re.I)
PR_ABBR = re.compile(r"\bPR\b", re.I)
RF_ABBR = re.compile(r"\bRF\b", re.I)

PR_CONTEXT = re.compile(r"(polyarthrite|rhumato|arthritis|anti[-\s]?ccp|acpa|facteur rhumato|rf|methotrex|mtx|dmard|bioth[eé]rapie|synovit|[ée]rosions?)", re.I)
PR_TRAP_CTX = re.compile(r"\b(pression|art[ée]rielle|pa\b|proth[eè]se|prostate|prurit)\b", re.I)

RF_LONG = re.compile(r"\b(facteur rhumato[iï]de|rheumatoid factor)\b", re.I)
CCP_PAT = re.compile(r"\b(anti[-\s]?ccp|acpa|anti[-\s]?citrullin)\b", re.I)
MTX_PAT = re.compile(r"\b(methotrexate|m[ée]thotrexate|MTX)\b", re.I)
BIO_PAT = re.compile(r"\b(adalimumab|etanercept|infliximab|tocilizumab|abatacept|rituximab|golimumab|certolizumab|sarilumab|tofacitinib|baricitinib|upadacitinib)\b", re.I)

EXCLUDE_PAT = re.compile(r"\b(pas de|absence de|[ée]cart[ée]e?|exclu[e]?|non retenu[e]?)\b", re.I)
CTX_BONUS = re.compile(r"\b(rhumatolog|acr|eular|synovit|[ée]rosions?|dmard|bioth[eé]rap)\b", re.I)

def _line_has_pr_anchor(line: str) -> bool:
    if RA_LONG.search(line):
        return True
    if PR_ABBR.search(line) and PR_CONTEXT.search(line) and not PR_TRAP_CTX.search(line):
        return True
    return False

def _line_has_rf_anchor(line: str) -> bool:
    if RF_LONG.search(line):
        return True
    if RF_ABBR.search(line) and re.search(r"(positif|n[ée]gatif|dosage|taux|IU|UI|valeur)", line, re.I):
        return True
    return False

def _line_has_ccp_anchor(line: str) -> bool:
    return CCP_PAT.search(line) is not None

def _line_has_drug_anchor(line: str) -> bool:
    return (MTX_PAT.search(line) is not None) or (BIO_PAT.search(line) is not None)

def score_window(lines: List[str]) -> float:
    t = " ".join([ln.strip() for ln in lines if ln.strip()])
    s = 0.0
    if RA_LONG.search(t) or (_line_has_pr_anchor(t) and PR_ABBR.search(t)): s += 3.0
    if CCP_PAT.search(t): s += 3.0
    if MTX_PAT.search(t): s += 2.0
    if BIO_PAT.search(t): s += 2.0
    if RF_LONG.search(t) or _line_has_rf_anchor(t): s += 1.5
    if CTX_BONUS.search(t): s += 0.8
    if EXCLUDE_PAT.search(t) and (RA_LONG.search(t) or PR_ABBR.search(t) or CCP_PAT.search(t) or RF_LONG.search(t)): s -= 3.0
    if PR_ABBR.search(t) and PR_TRAP_CTX.search(t) and not PR_CONTEXT.search(t): s -= 4.0
    return s

def merge_overlapping(windows: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if not windows: return []
    windows = sorted(windows)
    merged = [windows[0]]
    for s,e in windows[1:]:
        ps,pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe,e))
        else:
            merged.append((s,e))
    return merged

def hybrid_rag_select_windows(stay_text: str, window: int = 2, top_k: int = 12, head_if_none: int = 20) -> List[Tuple[int, List[str], float]]:
    lines = [ln.rstrip("\n") for ln in stay_text.splitlines()]
    if not lines:
        return [(0, [], 0.0)]

    hit_idxs = []
    for i, ln in enumerate(lines):
        if _line_has_pr_anchor(ln) or _line_has_rf_anchor(ln) or _line_has_ccp_anchor(ln) or _line_has_drug_anchor(ln):
            hit_idxs.append(i)

    if not hit_idxs:
        head = lines[:min(len(lines), head_if_none)]
        return [(0, head, score_window(head))]

    raw = []
    for idx in hit_idxs:
        s = max(0, idx - window)
        e = min(len(lines), idx + window + 1)
        raw.append((s,e))
    raw = merge_overlapping(raw)

    cand = []
    for s,e in raw:
        wlines = lines[s:e]
        cand.append((s, wlines, score_window(wlines)))

    cand.sort(key=lambda x: x[2], reverse=True)
    picked = [x for x in cand if x[2] > 0][:top_k]
    if not picked:
        picked = cand[:top_k]
    picked.sort(key=lambda x: x[0])
    return picked

# =========================
# AGENT1 AVEC LLM (OLLAMA)
# =========================
def agent1_llm_one_stay(patient_id: str, stay_id: str, stay_text: str, 
                        llm: OllamaLLM, rag_window=2, rag_top_k=12) -> Dict[str, Any]:
    """Version avec LLM pour enrichir l'extraction"""
    
    # D'abord, extraction regex de base
    selected = hybrid_rag_select_windows(stay_text, window=rag_window, top_k=rag_top_k)
    disease_mentions, labs, drugs = [], [], []

    # Extraction regex
    for (start_idx, wlines, _sc) in selected:
        for j, line in enumerate(wlines):
            line_clean = (line or "").strip()
            if not line_clean:
                continue
            line_no = start_idx + j + 1

            ctx_lines = wlines[max(0, j-2): min(len(wlines), j+3)]
            ctx = " ".join([x.strip() for x in ctx_lines if x.strip()])
            status = cue_status(ctx)

            if _line_has_pr_anchor(line_clean):
                disease_mentions.append({
                    "mention": "polyarthrite rhumatoïde" if RA_LONG.search(line_clean) else "PR",
                    "status": status,
                    "evidence": [{"stay_id": stay_id, "line_no": line_no, "snippet": line_clean}],
                })

            if _line_has_rf_anchor(line_clean):
                labs.append({
                    "name": "RF",
                    "value": polarity_from_line(line_clean) or "unknown",
                    "evidence": [{"stay_id": stay_id, "line_no": line_no, "snippet": line_clean}],
                })

            if _line_has_ccp_anchor(line_clean):
                labs.append({
                    "name": "anti-CCP",
                    "value": polarity_from_line(line_clean) or "unknown",
                    "evidence": [{"stay_id": stay_id, "line_no": line_no, "snippet": line_clean}],
                })

            if MTX_PAT.search(line_clean):
                dose_m = re.search(r"(\d+\s?mg\s?/?\s?(sem|semaine|week))", line_clean, flags=re.I)
                drugs.append({
                    "name": "methotrexate",
                    "status": "started" if "début" in line_clean.lower() or "debut" in line_clean.lower() else "mentioned",
                    "dose": dose_m.group(1) if dose_m else None,
                    "evidence": [{"stay_id": stay_id, "line_no": line_no, "snippet": line_clean}],
                })

            if BIO_PAT.search(line_clean):
                d = BIO_PAT.search(line_clean).group(1)
                drugs.append({
                    "name": d.lower(),
                    "status": "mentioned",
                    "dose": None,
                    "evidence": [{"stay_id": stay_id, "line_no": line_no, "snippet": line_clean}],
                })

    # Enrichissement avec LLM si des mentions ont été trouvées
    llm_enrichment = None
    if disease_mentions or labs or drugs:
        # Préparer le contexte pour le LLM
        context_text = "\n".join([
            f"Ligne {s+1}-{s+len(w)}: {' '.join(w)}"
            for s, w, sc in selected[:3]  # Top 3 fenêtres seulement
        ])
        
        system_prompt = """Tu es un assistant médical spécialisé dans l'extraction d'informations sur la polyarthrite rhumatoïde.
Analyse le texte et identifie:
1. Le statut du diagnostic (confirmé, suspecté, écarté)
2. Les valeurs de laboratoire manquées (RF, anti-CCP)
3. Les traitements et leurs dosages
Réponds en JSON uniquement."""

        prompt = f"""Contexte médical:
{context_text[:1500]}

Extrais les informations sur:
- Diagnostic de polyarthrite rhumatoïde (confirmé/suspecté/écarté)
- Facteur rhumatoïde (RF) et anti-CCP (positif/négatif/valeur)
- Traitements (méthotrexate, biothérapies)

Réponds en JSON:
{{
  "diagnostic_status": "confirmed|suspected|excluded|unknown",
  "confidence": 0-1,
  "additional_findings": "..."
}}"""

        try:
            llm_response = llm.generate(prompt, system_prompt=system_prompt, temperature=0.1, max_tokens=500)
            llm_enrichment = llm.extract_json_from_response(llm_response)
        except Exception as e:
            logging.warning(f"Erreur LLM enrichissement: {e}")

    return {
        "patient_id": patient_id,
        "stay_id": stay_id,
        "disease_mentions": disease_mentions[:10],
        "labs": labs[:10],
        "drugs": drugs[:10],
        "missing": [],
        "llm_enrichment": llm_enrichment,
        "_rag_meta": {
            "selected_windows": [
                {"start_line": s+1, "end_line": s+len(w), "score": round(sc,2)}
                for (s,w,sc) in selected
            ]
        }
    }

# Version regex pure (originale)
def agent1_regex_one_stay(patient_id: str, stay_id: str, stay_text: str, rag_window=2, rag_top_k=12) -> Dict[str, Any]:
    selected = hybrid_rag_select_windows(stay_text, window=rag_window, top_k=rag_top_k)
    disease_mentions, labs, drugs = [], [], []

    for (start_idx, wlines, _sc) in selected:
        for j, line in enumerate(wlines):
            line_clean = (line or "").strip()
            if not line_clean:
                continue
            line_no = start_idx + j + 1

            ctx_lines = wlines[max(0, j-2): min(len(wlines), j+3)]
            ctx = " ".join([x.strip() for x in ctx_lines if x.strip()])
            status = cue_status(ctx)

            if _line_has_pr_anchor(line_clean):
                disease_mentions.append({
                    "mention": "polyarthrite rhumatoïde" if RA_LONG.search(line_clean) else "PR",
                    "status": status,
                    "evidence": [{"stay_id": stay_id, "line_no": line_no, "snippet": line_clean}],
                })

            if _line_has_rf_anchor(line_clean):
                labs.append({
                    "name": "RF",
                    "value": polarity_from_line(line_clean) or "unknown",
                    "evidence": [{"stay_id": stay_id, "line_no": line_no, "snippet": line_clean}],
                })

            if _line_has_ccp_anchor(line_clean):
                labs.append({
                    "name": "anti-CCP",
                    "value": polarity_from_line(line_clean) or "unknown",
                    "evidence": [{"stay_id": stay_id, "line_no": line_no, "snippet": line_clean}],
                })

            if MTX_PAT.search(line_clean):
                dose_m = re.search(r"(\d+\s?mg\s?/?\s?(sem|semaine|week))", line_clean, flags=re.I)
                drugs.append({
                    "name": "methotrexate",
                    "status": "started" if "début" in line_clean.lower() or "debut" in line_clean.lower() else "mentioned",
                    "dose": dose_m.group(1) if dose_m else None,
                    "evidence": [{"stay_id": stay_id, "line_no": line_no, "snippet": line_clean}],
                })

            if BIO_PAT.search(line_clean):
                d = BIO_PAT.search(line_clean).group(1)
                drugs.append({
                    "name": d.lower(),
                    "status": "mentioned",
                    "dose": None,
                    "evidence": [{"stay_id": stay_id, "line_no": line_no, "snippet": line_clean}],
                })

    return {
        "patient_id": patient_id,
        "stay_id": stay_id,
        "disease_mentions": disease_mentions[:10],
        "labs": labs[:10],
        "drugs": drugs[:10],
        "missing": [],
        "_rag_meta": {
            "selected_windows": [
                {"start_line": s+1, "end_line": s+len(w), "score": round(sc,2)}
                for (s,w,sc) in selected
            ]
        }
    }

# =========================
# RUNNER
# =========================
def run(corpus_path: str, out_patient_jsonl: str, out_stay_jsonl: str, use_llm: bool = True):
    logger = setup_logger("agent1_run.log")
    T = Timer()

    ensure_parent_dir(out_patient_jsonl)
    ensure_parent_dir(out_stay_jsonl)

    logger.info(f"START | corpus={corpus_path} | use_llm={use_llm}")
    if not os.path.exists(corpus_path):
        logger.error("Corpus not found")
        raise FileNotFoundError(corpus_path)

    # Initialiser le LLM si demandé
    llm = None
    if use_llm:
        try:
            llm = OllamaLLM(model_name="qwen2.5:3b-instruct")
            logger.info("LLM Ollama initialisé: qwen2.5:3b-instruct")
        except Exception as e:
            logger.warning(f"Impossible d'initialiser Ollama: {e}. Utilisation regex seul.")
            use_llm = False

    corpus = open(corpus_path, "r", encoding="utf-8").read()
    patients = split_patients(corpus)
    logger.info(f"Corpus chars={len(corpus)} | patients={len(patients)}")

    with open(out_patient_jsonl, "w", encoding="utf-8") as fp, open(out_stay_jsonl, "w", encoding="utf-8") as fs:
        for i, p in enumerate(patients, start=1):
            pid = p["patient_id"]
            pT = Timer()
            stays = extract_stays(p["text"])
            logger.info(f"[P{i}/{len(patients)}] pid={pid} stays={len(stays)}")

            all_dm, all_labs, all_drugs = [], [], []
            invalid = 0

            for j, st in enumerate(stays, start=1):
                sid = st["stay_id"] or "UNKNOWN"
                stT = Timer()
                
                # Choisir la fonction appropriée
                if use_llm and llm:
                    out = agent1_llm_one_stay(pid, sid, st["text"], llm)
                else:
                    out = agent1_regex_one_stay(pid, sid, st["text"])
                    
                fs.write(json.dumps(out, ensure_ascii=False) + "\n")

                dm_n = len(out.get("disease_mentions", []))
                lab_n = len(out.get("labs", []))
                dr_n = len(out.get("drugs", []))
                logger.info(f"  [S{j}/{len(stays)}] sid={sid} dm={dm_n} labs={lab_n} drugs={dr_n} t={stT.elapsed():.2f}s")

                all_dm.extend(out.get("disease_mentions", []))
                all_labs.extend(out.get("labs", []))
                all_drugs.extend(out.get("drugs", []))

            patient_obj = {
                "patient_id": pid,
                "facts": {
                    "disease_mentions": all_dm,
                    "labs": all_labs,
                    "drugs": all_drugs
                },
                "quality_flags": {
                    "invalid_stays": invalid,
                    "notes": f"Agent1 {'LLM+regex' if use_llm else 'regex'} + hybrid RAG"
                }
            }
            fp.write(json.dumps(patient_obj, ensure_ascii=False) + "\n")
            logger.info(f"[P{i}] done pid={pid} t={pT.elapsed():.2f}s")

    logger.info(f"END | total_t={T.elapsed():.2f}s")
    logger.info(f"Outputs: {out_patient_jsonl} | {out_stay_jsonl}")
    logger.info(f"Patient bytes={os.path.getsize(out_patient_jsonl)} | Stay bytes={os.path.getsize(out_stay_jsonl)}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Pour utiliser avec LLM:
    run(
        corpus_path="phantom_ehr_corpus.txt",
        out_patient_jsonl="facts_agent1_patient.jsonl",
        out_stay_jsonl="facts_agent1_stay.jsonl",
        use_llm=True  # Mettre False pour désactiver le LLM
    )