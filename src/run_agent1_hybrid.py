import json
import csv
import re
from typing import Any, Dict, List, Optional
from collections import defaultdict

# ============================================================
# Utils
# ============================================================
def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def safe_lower(x: Any) -> str:
    return (x or "").strip().lower()

def norm_patient_id(pid: Any, width: int = 4) -> str:
    """
    IMPORTANT: ensures patient_id joins across files.
    Examples: 1, "1", "0001" -> "0001"
    """
    s = str(pid).strip()
    if not s:
        return ""
    if re.fullmatch(r"\d+", s):
        return s.zfill(width)
    return s

def normalize_lab_name(name: str) -> str:
    n = safe_lower(name)
    n = n.replace("_", "-").replace(" ", "")
    if "anti" in n and "ccp" in n:
        return "anti-ccp"
    if "acpa" in n:
        return "anti-ccp"
    if n in ["rf", "facteurrhumatoïde", "facteurrhumatoide", "rheumatoidfactor"]:
        return "rf"
    return n

def normalize_drug_name(name: str) -> str:
    n = safe_lower(name)
    if "methotrex" in n or n == "mtx":
        return "methotrexate"
    return n

def is_ra_mention(mention: str) -> bool:
    m = safe_lower(mention)
    return ("polyarthrite rhumato" in m) or ("rheumatoid arthritis" in m) or (m in ["pr", "ra"])

def collect_evidence(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    ev = item.get("evidence", [])
    if not isinstance(ev, list):
        return []
    out = []
    for e in ev:
        if not isinstance(e, dict):
            continue
        stay_id = e.get("stay_id")
        line_no = e.get("line_no")
        snippet = e.get("snippet")
        if stay_id is not None and line_no is not None and snippet:
            out.append({"stay_id": stay_id, "line_no": line_no, "snippet": snippet})
    return out

def best_status(statuses: List[str]) -> str:
    prio = {"confirmed": 4, "suspected": 3, "mentioned": 2, "negated": 1}
    best = None
    best_p = -1
    for s in statuses:
        p = prio.get(s, 0)
        if p > best_p:
            best_p = p
            best = s
    return best or "unknown"

def best_lab(vals: List[str]) -> str:
    vals = [safe_lower(v) for v in vals]
    if "positive" in vals:
        return "positive"
    if "negative" in vals:
        return "negative"
    if "unknown" in vals:
        return "unknown"
    return "missing"

def dedup_evidence(evidence_used: List[Dict[str, Any]], cap: int = 25) -> List[Dict[str, Any]]:
    uniq = []
    seen = set()
    for e in evidence_used:
        k = (e.get("stay_id"), e.get("line_no"), e.get("snippet"))
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq[:cap]

def get_facts(patient_obj: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(patient_obj.get("facts"), dict):
        return patient_obj["facts"]
    return {
        "disease_mentions": patient_obj.get("disease_mentions", []),
        "labs": patient_obj.get("labs", []),
        "drugs": patient_obj.get("drugs", []),
    }

# ============================================================
# RAG helpers (robust)
# ============================================================
def _rag_scores_from_meta(rag: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Returns:
      rag_score_max, rag_score_sum_top3
    Uses direct keys if present, else computes from selected_windows scores.
    """
    if not isinstance(rag, dict):
        return {"rag_score_max": None, "rag_score_sum_top3": None}

    key_candidates_max = ["rag_score_max", "score_max", "max_score", "rag_max"]
    key_candidates_sum3 = ["rag_score_sum_top3", "score_sum_top3", "sum_top3", "rag_sum_top3"]

    smax = None
    for k in key_candidates_max:
        v = rag.get(k)
        if isinstance(v, (int, float)):
            smax = float(v)
            break

    ssum3 = None
    for k in key_candidates_sum3:
        v = rag.get(k)
        if isinstance(v, (int, float)):
            ssum3 = float(v)
            break

    sel = rag.get("selected_windows")
    if isinstance(sel, list) and sel:
        scores = []
        for w in sel:
            if isinstance(w, dict):
                sc = w.get("score")
                if isinstance(sc, (int, float)):
                    scores.append(float(sc))
        if scores:
            if smax is None:
                smax = max(scores)
            if ssum3 is None:
                ssum3 = sum(sorted(scores, reverse=True)[:3])

    return {"rag_score_max": smax, "rag_score_sum_top3": ssum3}

def load_rag_patient_index(stay_jsonl_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregates per patient from stay-level jsonl:
      - rag_score_max_patient (max over stays)
      - rag_sum_top3_patient (max over stays)
      - rag_best_stay (stay that achieved max)
    Also prints diagnostics.
    """
    if not stay_jsonl_path:
        return {}

    idx: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "rag_score_max_patient": None,
        "rag_sum_top3_patient": None,
        "rag_best_stay": None,
        "stays_seen": 0,
    })

    n_lines = 0
    n_with_rag_meta = 0
    n_with_any_score = 0
    n_with_selected_windows = 0

    with open(stay_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            obj = json.loads(line)

            pid = norm_patient_id(obj.get("patient_id"))
            if not pid:
                continue

            idx[pid]["stays_seen"] += 1

            rag = obj.get("_rag_meta")
            if not isinstance(rag, dict):
                continue
            n_with_rag_meta += 1

            if isinstance(rag.get("selected_windows"), list):
                n_with_selected_windows += 1

            scores = _rag_scores_from_meta(rag)
            smax = scores["rag_score_max"]
            ssum3 = scores["rag_score_sum_top3"]

            if smax is None and ssum3 is None:
                continue
            n_with_any_score += 1

            sid = obj.get("stay_id") or obj.get("stay") or obj.get("doc_id") or "UNKNOWN"

            if isinstance(smax, (int, float)):
                cur = idx[pid]["rag_score_max_patient"]
                if cur is None or float(smax) > float(cur):
                    idx[pid]["rag_score_max_patient"] = float(smax)
                    idx[pid]["rag_best_stay"] = sid

            if isinstance(ssum3, (int, float)):
                cur2 = idx[pid]["rag_sum_top3_patient"]
                if cur2 is None or float(ssum3) > float(cur2):
                    idx[pid]["rag_sum_top3_patient"] = float(ssum3)

    print("==== RAG INDEX DIAGNOSTICS ====")
    print(f"Stay lines read                 : {n_lines}")
    print(f"Lines with _rag_meta            : {n_with_rag_meta}")
    print(f"Lines with selected_windows     : {n_with_selected_windows}")
    print(f"Lines with usable rag score     : {n_with_any_score}")
    print(f"Patients with any rag score     : {sum(1 for _,v in idx.items() if v['rag_score_max_patient'] is not None or v['rag_sum_top3_patient'] is not None)}")
    print("================================")

    return idx

# ============================================================
# Extraction scoring v3 (phantom-friendly)
#   - No penalty for RF-/anti-CCP- (seroneg PR exists in your corpus)
#   - Hard exclude if alternative dx + explicit PR exclusion in evidence snippets
#   - MTX weaker if no PR mention
# ============================================================
ALT_DX_STRONG = ["arthrose", "prothèse", "prothese", "gonarthrose", "coxarthrose"]
EXCLUDE_CUES = [
    "pas de polyarthrite", "pas de pr", "pr écartée", "pr ecartee",
    "écartée", "ecartee", "exclue", "non retenue", "rule out", "ruled out"
]
NO_SYNOVITIS_CUES = ["pas de synovite", "absence de synovite"]

def any_in_text(txt: str, pats: List[str]) -> bool:
    t = safe_lower(txt)
    return any(p in t for p in pats)

def evidence_text_blob(evidence: List[Dict[str, Any]]) -> str:
    return " ".join([safe_lower(e.get("snippet")) for e in evidence if isinstance(e, dict)])

def score_patient(patient_obj: Dict[str, Any]) -> Dict[str, Any]:
    pid = norm_patient_id(patient_obj.get("patient_id"))
    facts = get_facts(patient_obj)

    disease_mentions = as_list(facts.get("disease_mentions", []))
    labs = as_list(facts.get("labs", []))
    drugs = as_list(facts.get("drugs", []))

    evidence_used: List[Dict[str, Any]] = []
    reasons: List[str] = []
    contradictions: List[str] = []
    score = 0.0

    # A) PR status
    ra_statuses: List[str] = []
    for dm in disease_mentions:
        if not isinstance(dm, dict):
            continue
        mention = dm.get("mention") or dm.get("term") or ""
        st = safe_lower(dm.get("status"))
        if is_ra_mention(mention):
            ra_statuses.append(st)
            evidence_used.extend(collect_evidence(dm))

    ra_best = best_status(ra_statuses)

    # B) Labs (NO negative penalty)
    rf_vals: List[str] = []
    ccp_vals: List[str] = []

    for lab in labs:
        if not isinstance(lab, dict):
            continue
        name = normalize_lab_name(lab.get("name") or "")
        val = safe_lower(lab.get("value"))

        if name == "rf":
            rf_vals.append(val)
            evidence_used.extend(collect_evidence(lab))
        elif name == "anti-ccp":
            ccp_vals.append(val)
            evidence_used.extend(collect_evidence(lab))

    rf_best = best_lab(rf_vals)
    ccp_best = best_lab(ccp_vals)

    if ccp_best == "positive":
        score += 3.0; reasons.append("anti-CCP/ACPA positive (+3)")
    if rf_best == "positive":
        score += 2.0; reasons.append("RF positive (+2)")

    # C) Drugs
    biologics = {
        "adalimumab", "etanercept", "infliximab",
        "tocilizumab", "abatacept", "rituximab",
        "golimumab", "certolizumab", "sarilumab",
        "tofacitinib", "baricitinib", "upadacitinib",
    }

    has_mtx = False
    has_biologic = False

    for dr in drugs:
        if not isinstance(dr, dict):
            continue
        dname = normalize_drug_name(dr.get("name") or "")

        if dname == "methotrexate":
            has_mtx = True
            evidence_used.extend(collect_evidence(dr))
        if dname in biologics:
            has_biologic = True
            evidence_used.extend(collect_evidence(dr))

    if has_mtx:
        if ra_best in ["confirmed", "suspected", "mentioned"]:
            score += 2.0; reasons.append("MTX + PR mention (+2)")
        else:
            score += 0.5; reasons.append("MTX without PR mention (+0.5)")

    if has_biologic:
        score += 2.5; reasons.append("bDMARD/JAKi present (+2.5)")

    # D) PR mention/status weight
    if ra_best == "confirmed":
        score += 2.5; reasons.append("RA confirmed (+2.5)")
    elif ra_best == "suspected":
        score += 1.0; reasons.append("RA suspected (+1)")
    elif ra_best == "mentioned":
        score += 0.5; reasons.append("RA mentioned (+0.5)")
    elif ra_best == "negated":
        score -= 5.0; reasons.append("RA negated (-5)")

    # E) Hard exclude from evidence snippets (works only if evidence contains those lines)
    # NOTE: If "arthrose/pas de PR" isn't in evidence, you won't catch it here.
    ev_blob = evidence_text_blob(evidence_used)
    hard_exclude = False
    if any_in_text(ev_blob, ALT_DX_STRONG) and (any_in_text(ev_blob, EXCLUDE_CUES) or any_in_text(ev_blob, NO_SYNOVITIS_CUES)):
        hard_exclude = True
        score -= 6.0
        reasons.append("Hard exclude: alt dx + PR exclusion (-6)")

    # Contradiction
    if ra_best == "negated" and (ccp_best == "positive" or rf_best == "positive" or has_biologic):
        contradictions.append("RA negated but strong supporting evidence exists")
        score -= 0.8
        reasons.append("Penalty contradiction (-0.8)")

    # Internal state
    if hard_exclude:
        internal_state = "excluded"
    elif score >= 5.0:
        internal_state = "confirmed"
    elif score >= 2.0:
        internal_state = "suspected"
    else:
        internal_state = "indeterminate"

    pred = 1 if internal_state == "confirmed" else 0

    # Confidence
    if internal_state == "confirmed":
        confidence = 0.85 + (0.07 if ccp_best == "positive" else 0.0)
    elif internal_state == "excluded":
        confidence = 0.90
    elif internal_state == "suspected":
        confidence = 0.55
    else:
        confidence = 0.50

    evidence_used = dedup_evidence(evidence_used, cap=25)

    return {
        "patient_id": pid,
        "pred": pred,
        "internal_state": internal_state,
        "score": round(score, 3),
        "confidence": round(confidence, 4),
        "signals": {
            "ra_best": ra_best,
            "rf_best": rf_best,
            "ccp_best": ccp_best,
            "has_mtx": has_mtx,
            "has_biologic": has_biologic,
            "hard_exclude": hard_exclude,
        },
        "contradictions": contradictions,
        "reasons": reasons,
        "evidence_used": evidence_used,
    }

# ============================================================
# Runner
# ============================================================
def load_labels(labels_csv: Optional[str]) -> Dict[str, int]:
    """
    Reads labels from CSV with columns: patient_id,label_binary (or label)
    No pandas needed.
    """
    labels: Dict[str, int] = {}
    if not labels_csv:
        return labels

    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = norm_patient_id(row.get("patient_id"))
            if not pid:
                continue
            y = row.get("label_binary")
            if y is None or y == "":
                y = row.get("label")
            if y is None or y == "":
                continue
            labels[pid] = int(y)

    print(f"✅ Labels loaded: {len(labels)} patients")
    return labels

def run(
    in_patient_jsonl: str,
    out_csv: str,
    out_jsonl: str,
    labels_csv: Optional[str] = None,
    in_stay_jsonl_for_rag: Optional[str] = None
):
    labels = load_labels(labels_csv)

    rag_idx = load_rag_patient_index(in_stay_jsonl_for_rag) if in_stay_jsonl_for_rag else {}
    if in_stay_jsonl_for_rag:
        print(f"✅ Loaded RAG index: {len(rag_idx)} patients")

    rows_csv: List[Dict[str, Any]] = []
    n_in = 0
    n_joined = 0

    with open(in_patient_jsonl, "r", encoding="utf-8") as f_in, \
         open(out_jsonl, "w", encoding="utf-8") as f_jsonl:

        for line in f_in:
            if not line.strip():
                continue
            n_in += 1
            obj = json.loads(line)

            out = score_patient(obj)
            pid = out["patient_id"]

            if pid in labels:
                out["label_true"] = labels[pid]

            rag = rag_idx.get(pid)
            if rag:
                n_joined += 1
                out["rag_score_max_patient"] = rag.get("rag_score_max_patient")
                out["rag_sum_top3_patient"] = rag.get("rag_sum_top3_patient")
                out["rag_best_stay"] = rag.get("rag_best_stay")
            else:
                out["rag_score_max_patient"] = None
                out["rag_sum_top3_patient"] = None
                out["rag_best_stay"] = None

            if out["rag_score_max_patient"] is not None:
                delta = float(out["score"]) - float(out["rag_score_max_patient"])
                out["delta_extraction_minus_rag"] = round(delta, 3)
                out["rag_bucket"] = "pos" if delta > 0 else ("zero" if delta == 0 else "neg")
            else:
                out["delta_extraction_minus_rag"] = None
                out["rag_bucket"] = "no_rag"

            f_jsonl.write(json.dumps(out, ensure_ascii=False) + "\n")

            rows_csv.append({
                "patient_id": out["patient_id"],
                "label_true": out.get("label_true", ""),
                "pred": out["pred"],
                "internal_state": out["internal_state"],
                "confidence": out["confidence"],
                "score_extraction": out["score"],
                "ra_best": out["signals"]["ra_best"],
                "rf_best": out["signals"]["rf_best"],
                "ccp_best": out["signals"]["ccp_best"],
                "has_mtx": int(out["signals"]["has_mtx"]),
                "has_biologic": int(out["signals"]["has_biologic"]),
                "hard_exclude": int(out["signals"]["hard_exclude"]),
                "n_evidence": len(out["evidence_used"]),
                "rag_score_max_patient": out["rag_score_max_patient"],
                "rag_sum_top3_patient": out["rag_sum_top3_patient"],
                "rag_best_stay": out["rag_best_stay"],
                "delta_extraction_minus_rag": out["delta_extraction_minus_rag"],
                "rag_bucket": out["rag_bucket"],
            })

    if rows_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
            w = csv.DictWriter(f_out, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)

    print("==== JOIN DIAGNOSTICS ====")
    print(f"Patients read (patient jsonl)   : {n_in}")
    print(f"Patients with rag attached      : {n_joined}")
    print("==========================")
    print(f"✅ Done\n- {out_csv}\n- {out_jsonl}")

if __name__ == "__main__":
    run(
        in_patient_jsonl="facts_agent1_patient.jsonl",
        out_csv="predictions_patient_level.csv",
        out_jsonl="predictions_patient_level.jsonl",
        labels_csv="labels.csv",                 # mets None si tu veux
        in_stay_jsonl_for_rag="facts_agent1_stay.jsonl"
    )
