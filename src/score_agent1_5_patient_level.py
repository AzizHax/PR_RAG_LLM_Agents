import json
import csv
import re
import math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# PATHS (EDIT THESE)
# ============================================================
LABELS_CSV = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\labels.csv"
FACTS_JSONL = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\facts_agent1_stay_umls.jsonl"

OUT_CSV = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\src\delta_patient_level.csv"
OUT_JSONL = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\src\delta_patient_level.jsonl"
OUT_PLOT = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\src\patient_score_zones.png"
OUT_GRID = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\src\grid_search_thresholds.csv"

# ============================================================
# GRID SEARCH CONFIG
# ============================================================
GRID_MIN = -1.0
GRID_MAX = 10.0
GRID_STEP = 0.25

# ============================================================
# Helpers
# ============================================================
def safe_lower(x) -> str:
    return (str(x) if x is not None else "").strip().lower()

def norm_patient_id(pid) -> str:
    s = str(pid).strip() if pid is not None else ""
    if not s:
        return ""
    if re.fullmatch(r"\d+", s):
        return s.zfill(4)
    return s

def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def pick_first(d: dict, keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None

def is_ra_text(s: str) -> bool:
    m = safe_lower(s)
    return (
        ("polyarthrite rhumato" in m)
        or ("rheumatoid arthritis" in m)
        or ("arthrite rhumato" in m)
        or (m.strip() in ["pr", "ra"])
    )

def grid_values(min_v, max_v, step):
    vals = []
    x = float(min_v)
    while x <= float(max_v) + 1e-9:
        vals.append(round(x, 6))
        x += float(step)
    return vals

# ============================================================
# Evidence: snippet + stay_id + line_no
# ============================================================
def make_evidence(stay_id, line_no, snippet, signal):
    snippet = (snippet or "").strip()
    if not stay_id or line_no is None or not snippet:
        return None
    return {"stay_id": str(stay_id), "line_no": int(line_no), "snippet": snippet, "signal": signal}

def extract_best_snippet(x: dict) -> str:
    """
    Returns an exact snippet present in JSON.
    Priority:
      1) x["evidence"][0]["snippet"]
      2) x["snippet"]/text/surface/mention/term/name/label/value/test
      3) json.dumps(x) fallback (still from JSON)
    """
    if not isinstance(x, dict):
        return ""
    ev = x.get("evidence")
    if isinstance(ev, list) and ev:
        e0 = ev[0]
        if isinstance(e0, dict):
            s = e0.get("snippet")
            if isinstance(s, str) and s.strip():
                return s

    v = pick_first(x, ["snippet", "text", "surface", "mention", "term", "name", "label", "value", "test"])
    if v is None:
        return json.dumps(x, ensure_ascii=False)
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)

def dedup_evidence(evidence_refs, max_keep=50):
    seen = set()
    out = []
    for e in evidence_refs:
        if not isinstance(e, dict):
            continue
        k = (e.get("stay_id"), e.get("line_no"), e.get("snippet"), e.get("signal"))
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
        if len(out) >= max_keep:
            break
    return out

# ============================================================
# RAG score extraction (from _rag_meta)
# ============================================================
def rag_scores_from_meta(rag_meta: dict):
    if not isinstance(rag_meta, dict):
        return (None, None)

    max_keys = ["rag_score_max", "score_max", "max_score", "rag_max"]
    sum3_keys = ["rag_score_sum_top3", "score_sum_top3", "sum_top3", "rag_sum_top3"]

    smax = None
    for k in max_keys:
        v = rag_meta.get(k)
        if isinstance(v, (int, float)):
            smax = float(v)
            break

    ssum3 = None
    for k in sum3_keys:
        v = rag_meta.get(k)
        if isinstance(v, (int, float)):
            ssum3 = float(v)
            break

    sel = rag_meta.get("selected_windows")
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
    return (smax, ssum3)

def rag_evidence_snippet(rag_meta: dict) -> str:
    if not isinstance(rag_meta, dict):
        return ""
    chunks = rag_meta.get("chunks")
    if isinstance(chunks, list) and chunks:
        c0 = chunks[0]
        if isinstance(c0, dict):
            s = pick_first(c0, ["text", "snippet", "content"])
            if isinstance(s, str) and s.strip():
                return s
    return json.dumps(rag_meta, ensure_ascii=False)

# ============================================================
# Extraction v3 scoring (patient-level)
#   score = best_status + best_lab + MTX + BIO - contradiction_penalty
# ============================================================
BIOLOGICS = {
    "adalimumab", "etanercept", "infliximab",
    "tocilizumab", "abatacept", "rituximab",
    "golimumab", "certolizumab", "sarilumab",
    "tofacitinib", "baricitinib", "upadacitinib",
}

def normalize_lab_name(name: str) -> str:
    n = safe_lower(name).replace("_", "-").replace(" ", "")
    if "anti" in n and "ccp" in n:
        return "anti-ccp"
    if "acpa" in n:
        return "anti-ccp"
    if n in ["rf", "facteurrhumatoïde", "facteurrhumatoide", "rheumatoidfactor", "facteurrhumato"]:
        return "rf"
    return n

def normalize_drug_name(name: str) -> str:
    n = safe_lower(name)
    if "methotrex" in n or n == "mtx" or "méthotrex" in n:
        return "methotrexate"
    return n

def mention_is_negative(dm: dict) -> bool:
    if not isinstance(dm, dict):
        return False
    for k in ["negated", "is_negated", "ruled_out", "is_ruled_out", "absent"]:
        if k in dm:
            v = dm.get(k)
            if isinstance(v, bool) and v:
                return True
            if safe_lower(v) in ("true", "1", "yes", "y"):
                return True
    st = safe_lower(dm.get("status"))
    if st in ("negated", "ruled_out", "absent", "no", "negative"):
        return True
    pol = safe_lower(dm.get("polarity"))
    if pol in ("negative", "neg", "-1"):
        return True
    return False

def mention_is_positive(dm: dict) -> bool:
    if not isinstance(dm, dict):
        return False
    if mention_is_negative(dm):
        return False
    st = safe_lower(dm.get("status"))
    if st in ("confirmed", "present", "positive", "probable", "suspected", "possible", "history_of", "mentioned"):
        return True
    pol = safe_lower(dm.get("polarity"))
    if pol in ("positive", "pos", "+1"):
        return True
    return False  # conservative

def lab_is_positive(lab: dict) -> bool:
    if not isinstance(lab, dict):
        return False

    for k in ["is_positive", "positive"]:
        if k in lab:
            v = lab.get(k)
            if isinstance(v, bool):
                return v
            if safe_lower(v) in ("true", "1", "yes", "pos", "positive", "reactive", "detected"):
                return True
            if safe_lower(v) in ("false", "0", "no", "neg", "negative", "non reactive", "not detected"):
                return False

    interp = safe_lower(lab.get("interpretation"))
    if interp in ("positive", "pos", "reactive", "detected"):
        return True
    if interp in ("negative", "neg", "non reactive", "not detected"):
        return False

    text = " ".join([safe_lower(lab.get("text")), safe_lower(lab.get("snippet")), safe_lower(lab.get("value"))])
    if "positif" in text or "positive" in text or "reactive" in text or "detected" in text:
        return True
    if "négatif" in text or "negatif" in text or "negative" in text or "non reactive" in text or "not detected" in text:
        return False

    return False  # unknown -> conservative

def score_patient_extraction_v3(pstate: dict):
    if pstate["ev_ra_pos"]:
        best_status = "confirmed"
    elif pstate["ev_ra_neg"]:
        best_status = "negated"
    else:
        best_status = "unknown"

    if pstate["ev_lab_pos"]:
        best_lab = "positive"
    elif pstate["ev_lab_neg"]:
        best_lab = "negative"
    else:
        best_lab = "unknown"

    has_mtx = bool(pstate["ev_mtx"])
    has_bio = bool(pstate["ev_bio"])
    contradiction = bool(pstate["ev_ra_pos"]) and bool(pstate["ev_ra_neg"])

    score = 0.0
    reasons = []

    if best_status == "confirmed":
        score += 3.0
        reasons.append("best_status=confirmed (+3)")
    else:
        reasons.append(f"best_status={best_status} (+0)")

    if best_lab == "positive":
        score += 2.0
        reasons.append("best_lab=positive (+2)")
    else:
        reasons.append(f"best_lab={best_lab} (+0)")

    if has_mtx:
        score += 1.0
        reasons.append("MTX present (+1)")
    if has_bio:
        score += 1.0
        reasons.append("biologic/JAK present (+1)")

    if contradiction:
        score -= 1.5
        reasons.append("contradiction penalty (-1.5)")

    breakdown = {
        "best_status": best_status,
        "best_lab": best_lab,
        "has_mtx": has_mtx,
        "has_biologic": has_bio,
        "contradiction": contradiction,
        "reasons": reasons[:12],
    }
    return float(score), breakdown

# ============================================================
# Metrics
# ============================================================
def confusion(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    return tp, fp, fn, tn

def metrics_from_conf(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return acc, prec, rec, f1

# ============================================================
# MAIN
# ============================================================
def run():
    # Load labels
    labels = {}
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = norm_patient_id(row.get("patient_id"))
            if not pid:
                continue
            y = row.get("label_binary", None)
            if y is None or str(y).strip() not in ("0", "1"):
                continue
            labels[pid] = int(y)
    print(f"✅ Labels loaded: {len(labels)} patients")

    patient_store = defaultdict(lambda: {
        "stays_seen": 0,
        "ev_ra_pos": [],
        "ev_ra_neg": [],
        "ev_lab_pos": [],
        "ev_lab_neg": [],
        "ev_mtx": [],
        "ev_bio": [],
        "ev_rag": [],
        "rag_score_max_patient": None,
        "rag_sum_top3_patient": None,
        "rag_best_stay": None,
        "extract_score_patient": None,
        "extract_breakdown": {},
        "patient_score": None,  # max(extract, rag)
    })

    n_lines = 0
    n_with_rag_meta = 0
    n_with_any_rag_score = 0

    with open(FACTS_JSONL, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "error" in obj or obj.get("extraction_status") == "error":
                continue

            pid = norm_patient_id(obj.get("patient_id"))
            if not pid:
                continue

            sid = obj.get("stay_id") or obj.get("stay") or obj.get("doc_id") or "UNKNOWN"
            st = patient_store[pid]
            st["stays_seen"] += 1

            # disease_mentions
            disease_mentions = as_list(obj.get("disease_mentions", obj.get("diseases", [])))
            for dm in disease_mentions:
                if not isinstance(dm, dict):
                    continue
                mention = pick_first(dm, ["mention", "term", "name", "label", "text", "snippet"])
                if mention is None or not is_ra_text(str(mention)):
                    continue

                snip = extract_best_snippet(dm)
                if mention_is_positive(dm):
                    ev = make_evidence(sid, line_no, snip, "RA_POS")
                    if ev:
                        st["ev_ra_pos"].append(ev)
                elif mention_is_negative(dm):
                    ev = make_evidence(sid, line_no, snip, "RA_NEG")
                    if ev:
                        st["ev_ra_neg"].append(ev)

            # labs
            labs = as_list(obj.get("labs", obj.get("lab_mentions", [])))
            for lab in labs:
                if not isinstance(lab, dict):
                    continue
                name_raw = pick_first(lab, ["name", "test", "label"])
                name = normalize_lab_name(str(name_raw or ""))
                if name not in ("rf", "anti-ccp"):
                    continue

                snip = extract_best_snippet(lab)
                sig = "LAB_POS" if lab_is_positive(lab) else "LAB_NEG"
                ev = make_evidence(sid, line_no, snip, sig)
                if not ev:
                    continue
                if sig == "LAB_POS":
                    st["ev_lab_pos"].append(ev)
                else:
                    st["ev_lab_neg"].append(ev)

            # drugs
            drugs = as_list(obj.get("drugs", obj.get("medications", obj.get("drug_mentions", []))))
            for dr in drugs:
                if not isinstance(dr, dict):
                    continue
                dname_raw = pick_first(dr, ["name", "drug", "label"])
                dname = normalize_drug_name(str(dname_raw or ""))
                snip = extract_best_snippet(dr)

                if dname == "methotrexate":
                    ev = make_evidence(sid, line_no, snip, "MTX")
                    if ev:
                        st["ev_mtx"].append(ev)
                if dname in BIOLOGICS:
                    ev = make_evidence(sid, line_no, snip, "BIO")
                    if ev:
                        st["ev_bio"].append(ev)

            # RAG
            rag = obj.get("_rag_meta")
            if isinstance(rag, dict):
                n_with_rag_meta += 1
                rmax, rsum3 = rag_scores_from_meta(rag)
                if (rmax is not None) or (rsum3 is not None):
                    n_with_any_rag_score += 1

                if isinstance(rmax, (int, float)):
                    cur = st["rag_score_max_patient"]
                    if cur is None or float(rmax) > float(cur):
                        st["rag_score_max_patient"] = float(rmax)
                        st["rag_best_stay"] = sid

                    ev = make_evidence(sid, line_no, rag_evidence_snippet(rag), "RAG")
                    if ev:
                        st["ev_rag"].append(ev)

                if isinstance(rsum3, (int, float)):
                    cur2 = st["rag_sum_top3_patient"]
                    if cur2 is None or float(rsum3) > float(cur2):
                        st["rag_sum_top3_patient"] = float(rsum3)

    # finalize evidence + compute scores
    for pid, st in patient_store.items():
        st["ev_ra_pos"] = dedup_evidence(st["ev_ra_pos"], max_keep=50)
        st["ev_ra_neg"] = dedup_evidence(st["ev_ra_neg"], max_keep=50)
        st["ev_lab_pos"] = dedup_evidence(st["ev_lab_pos"], max_keep=50)
        st["ev_lab_neg"] = dedup_evidence(st["ev_lab_neg"], max_keep=50)
        st["ev_mtx"] = dedup_evidence(st["ev_mtx"], max_keep=50)
        st["ev_bio"] = dedup_evidence(st["ev_bio"], max_keep=50)
        st["ev_rag"] = dedup_evidence(st["ev_rag"], max_keep=20)

        ex_score, ex_bd = score_patient_extraction_v3(st)
        st["extract_score_patient"] = ex_score
        st["extract_breakdown"] = ex_bd

        rag = st["rag_score_max_patient"]
        st["patient_score"] = float(ex_score) if rag is None else max(float(ex_score), float(rag))

    print("==== RAG DIAGNOSTICS ====")
    print(f"Stay lines read                 : {n_lines}")
    print(f"Lines with _rag_meta            : {n_with_rag_meta}")
    print(f"Lines with usable rag score     : {n_with_any_rag_score}")
    print(f"Patients aggregated             : {len(patient_store)}")
    print("=========================")

    # eval patients
    eval_pids = [pid for pid in labels.keys() if pid in patient_store]
    if not eval_pids:
        raise RuntimeError("No overlap between labels.csv and FACTS_JSONL patient_ids (after normalization).")

    # grid search
    grid_vals = grid_values(GRID_MIN, GRID_MAX, GRID_STEP)
    best = None
    best_row = None
    grid_rows = []

    for th_low in grid_vals:
        for th_high in grid_vals:
            if th_low > th_high:
                continue

            y_true = []
            y_pred = []
            n_uncertain = 0

            for pid in eval_pids:
                score = patient_store[pid]["patient_score"]
                y_true.append(labels[pid])

                if score < th_low:
                    zone = "NEG"
                elif score < th_high:
                    zone = "UNCERTAIN"
                    n_uncertain += 1
                else:
                    zone = "POS"

                y_pred.append(1 if zone == "POS" else 0)

            tp, fp, fn, tn = confusion(y_true, y_pred)
            acc, prec, rec, f1 = metrics_from_conf(tp, fp, fn, tn)

            row = {
                "mode": "1D_MAX(extract,rag)",
                "th_low": th_low,
                "th_high": th_high,
                "n": len(y_true),
                "acc": round(acc, 3),
                "prec": round(prec, 3),
                "rec": round(rec, 3),
                "f1": round(f1, 3),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "n_uncertain": n_uncertain,
            }
            grid_rows.append(row)

            key = (f1, acc, -fp, -n_uncertain)
            if best is None or key > best:
                best = key
                best_row = row

    Path(OUT_GRID).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_GRID, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader()
        w.writerows(grid_rows)
    print(f"✅ Wrote: {OUT_GRID}")

    TH_LOW = best_row["th_low"]
    TH_HIGH = best_row["th_high"]

    print("\n==== BEST THRESHOLDS (1D zones) ====")
    print(f"TH_LOW  = {TH_LOW}")
    print(f"TH_HIGH = {TH_HIGH}")
    print(f"Acc={best_row['acc']} Prec={best_row['prec']} Rec={best_row['rec']} F1={best_row['f1']}")
    print(f"TP/FP/FN/TN = {best_row['tp']}/{best_row['fp']}/{best_row['fn']}/{best_row['tn']}")
    print(f"n_uncertain = {best_row['n_uncertain']} (treated as PR-)")
    print("===================================\n")

    # outputs
    rows_out = []
    for pid, st in patient_store.items():
        ex = st["extract_score_patient"]
        rag = st["rag_score_max_patient"]
        score = st["patient_score"]

        if score < TH_LOW:
            zone = "NEG"
        elif score < TH_HIGH:
            zone = "UNCERTAIN"
        else:
            zone = "POS"

        y = labels.get(pid, "")
        pred = (1 if zone == "POS" else 0) if y != "" else ""

        evidence_pack = {
            "RA_POS": st["ev_ra_pos"][:5],
            "RA_NEG": st["ev_ra_neg"][:5],
            "LAB_POS": st["ev_lab_pos"][:5],
            "LAB_NEG": st["ev_lab_neg"][:5],
            "MTX": st["ev_mtx"][:5],
            "BIO": st["ev_bio"][:5],
            "RAG": st["ev_rag"][:3],
        }

        rows_out.append({
            "patient_id": pid,
            "label_true": y,
            "stays_seen": st["stays_seen"],
            "extract_score_patient_v3": round(ex, 3) if ex is not None else "",
            "rag_score_max_patient": rag if rag is not None else "",
            "patient_score_1D": round(score, 3) if score is not None else "",
            "TH_LOW_best": TH_LOW if y != "" else "",
            "TH_HIGH_best": TH_HIGH if y != "" else "",
            "zone": zone,
            "pred@best": pred,
            "extract_breakdown": json.dumps(st["extract_breakdown"], ensure_ascii=False),
            "evidence": json.dumps(evidence_pack, ensure_ascii=False),
        })

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Wrote: {OUT_CSV}")
    print(f"✅ Wrote: {OUT_JSONL}")

    # ============================================================
    # Plot: KDE curves per label (0 vs 1) + thresholds  (INSIDE run)
    # ============================================================
    plot_scores_0 = []
    plot_scores_1 = []
    for r in rows_out:
        if r["patient_score_1D"] == "":
            continue
        if r["label_true"] == "":
            continue
        sc = float(r["patient_score_1D"])
        if int(r["label_true"]) == 0:
            plot_scores_0.append(sc)
        else:
            plot_scores_1.append(sc)

    if not plot_scores_0 and not plot_scores_1:
        print("⚠️ No labeled patient_score to plot.")
        return

    def kde_1d(xs, grid, bw):
        if not xs:
            return [0.0 for _ in grid]
        if bw <= 0:
            bw = 0.1
        out = []
        inv = 1.0 / (bw * math.sqrt(2.0 * math.pi))
        two_bw2 = 2.0 * bw * bw
        n = len(xs)
        for g in grid:
            s = 0.0
            for x in xs:
                d = g - x
                s += math.exp(-(d * d) / two_bw2)
            out.append(inv * s / n)
        return out

    all_scores = plot_scores_0 + plot_scores_1
    mn = min(all_scores)
    mx = max(all_scores)
    pad = 0.5
    xmin = mn - pad
    xmax = mx + pad

    n_grid = 400
    grid = [xmin + i * (xmax - xmin) / (n_grid - 1) for i in range(n_grid)]

    bw = 0.45
    dens0 = kde_1d(plot_scores_0, grid, bw)
    dens1 = kde_1d(plot_scores_1, grid, bw)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, dens0, linewidth=2, label=f"Label 0 (PR-) n={len(plot_scores_0)}")
    ax.plot(grid, dens1, linewidth=2, label=f"Label 1 (PR+) n={len(plot_scores_1)}")

    ax.axvline(TH_LOW, linewidth=1)
    ax.axvline(TH_HIGH, linewidth=1)

    ymax = max(max(dens0), max(dens1)) if (dens0 or dens1) else 1.0
    ax.text((xmin + TH_LOW) / 2, ymax * 0.95, "PR(-)", ha="center", va="top", fontsize=10)
    ax.text((TH_LOW + TH_HIGH) / 2, ymax * 0.95, "Incertain", ha="center", va="top", fontsize=10)
    ax.text((TH_HIGH + xmax) / 2, ymax * 0.95, "PR(+)", ha="center", va="top", fontsize=10)

    ax.set_title("Distribution des scores (KDE) par label — zones PR(-) / Incertain / PR(+)")
    ax.set_xlabel("patient_score_1D = max(extract_score_v3, rag_score_max)")
    ax.set_ylabel("densité (approx.)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=200)
    print(f"✅ Plot saved: {OUT_PLOT}")


if __name__ == "__main__":
    run()
