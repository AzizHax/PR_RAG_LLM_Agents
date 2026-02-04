#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data")
CORPUS_PATH = DATA_DIR / "phantom_ehr_corpus.txt"
ANNOTATIONS_PATH = DATA_DIR / "phantom_annotations_minimal.txt"
OUTPUT_DIR = DATA_DIR / "audit_out_v3"

RE_PATIENT = re.compile(r"^\s*PATIENT_ID:\s*(\d+)\s*$", re.MULTILINE)

RE_RA_CONFIRMED = re.compile(r"\b(polyarthrite\s+rhumato[iï]de|PR)\s+confirm[ée]e\b", re.IGNORECASE)
RE_RA_TREATED = re.compile(r"\bPR\b.*\b(trait[ée]e|traitement)\b", re.IGNORECASE)

RE_RA_NEGATED = re.compile(r"\b(pas\s+de\s+PR|PR\s+[ée]cart[ée]e|pas\s+de\s+polyarthrite\s+rhumato[iï]de)\b", re.IGNORECASE)

RE_BIOLOGIC = re.compile(r"\b(adalimumab|etanercept|infliximab|tocilizumab|abatacept|rituximab)\b", re.IGNORECASE)
RE_JAK = re.compile(r"\b(tofacitinib|baricitinib|upadacitinib)\b", re.IGNORECASE)

RE_RF_POS = re.compile(r"\bRF\b\s*:\s*positif\b|\bRF\s*:\s*positif\s+à\s*[0-9]", re.IGNORECASE)
RE_CCP_POS = re.compile(r"\banti[-\s]?CCP\b\s*:\s*positif\b|\banti[-\s]?CCP\s*:\s*positif\s+à\s*[0-9]", re.IGNORECASE)

ANN_RE = re.compile(r"PATIENT_ID:\s*(\d+)\s*\|\s*LABEL_BINARY:\s*([01])\s*\|\s*COMMENT:\s*(.*)$")

def parse_corpus_by_patient(text: str) -> Dict[str, str]:
    patients = {}
    current = None
    buf = []
    for line in text.splitlines():
        m = re.match(r"^\s*PATIENT_ID:\s*(\d+)\s*$", line)
        if m:
            if current:
                patients[current] = "\n".join(buf).strip()
            current = m.group(1)
            buf = []
        else:
            if current:
                buf.append(line)
    if current:
        patients[current] = "\n".join(buf).strip()
    return patients

def parse_annotations(text: str) -> Dict[str, Tuple[int, str]]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = ANN_RE.match(line)
        if not m:
            continue
        pid = m.group(1)
        lab = int(m.group(2))
        com = m.group(3).strip()
        out[pid] = (lab, com)
    return out

def corpus_ra_ever_evidence(patient_text: str) -> Tuple[int, List[str], str]:
    """
    Returns:
      predicted_label (0/1), evidence_tags, confidence ("high"/"medium"/"low")
    """
    tags = []

    has_confirmed = bool(RE_RA_CONFIRMED.search(patient_text))
    has_biologic = bool(RE_BIOLOGIC.search(patient_text)) or bool(RE_JAK.search(patient_text))
    has_seropos = bool(RE_RF_POS.search(patient_text)) and bool(RE_CCP_POS.search(patient_text))
    has_negated = bool(RE_RA_NEGATED.search(patient_text))

    if has_confirmed:
        tags.append("ra_confirmed")
    if has_biologic:
        tags.append("biologic_or_jak")
    if has_seropos:
        tags.append("rf+_and_ccp+")
    if has_negated:
        tags.append("ra_negated")

    # Decision logic (conservative)
    if has_confirmed or has_biologic:
        return 1, tags, "high"
    if has_seropos and not has_negated:
        return 1, tags, "medium"
    if has_negated and not (has_confirmed or has_biologic or has_seropos):
        return 0, tags, "high"
    # default
    return 0, tags, "low"

def main():
    print("=" * 80)
    print("AUDIT v3: RA-ever consistency (corpus evidence vs annotations)")
    print("=" * 80)

    if not CORPUS_PATH.exists():
        print(f"[ERROR] Missing corpus: {CORPUS_PATH}")
        return
    if not ANNOTATIONS_PATH.exists():
        print(f"[ERROR] Missing annotations: {ANNOTATIONS_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_text = CORPUS_PATH.read_text(encoding="utf-8", errors="replace")
    ann_text = ANNOTATIONS_PATH.read_text(encoding="utf-8", errors="replace")

    corpus = parse_corpus_by_patient(corpus_text)
    ann = parse_annotations(ann_text)

    all_pids = sorted(set(corpus.keys()) | set(ann.keys()))
    rows = []

    auto_fix = []  # only high confidence mismatches

    for pid in all_pids:
        ctext = corpus.get(pid, "")
        alab, acomment = ann.get(pid, (None, ""))
        pred, tags, conf = corpus_ra_ever_evidence(ctext)

        if alab is None:
            status = "missing_annotation"
        elif pid not in corpus:
            status = "missing_in_corpus"
        else:
            status = "ok" if pred == alab else "mismatch"

        row = {
            "patient_id": pid,
            "annotation_label": alab if alab is not None else "",
            "corpus_evidence_label": pred if pid in corpus else "",
            "evidence_tags": ",".join(tags),
            "evidence_confidence": conf,
            "status": status,
            "comment": acomment
        }
        rows.append(row)

        if status == "mismatch" and conf == "high":
            auto_fix.append({
                "patient_id": pid,
                "from_label": alab,
                "to_label": pred,
                "evidence_tags": ",".join(tags),
                "comment": acomment
            })

    out_csv = OUTPUT_DIR / "cases_to_review.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_fix = OUTPUT_DIR / "auto_fix_suggestions.tsv"
    with out_fix.open("w", encoding="utf-8") as f:
        f.write("patient_id\tfrom_label\tto_label\tevidence_tags\tcomment\n")
        for r in auto_fix:
            f.write(f"{r['patient_id']}\t{r['from_label']}\t{r['to_label']}\t{r['evidence_tags']}\t{r['comment']}\n")

    print(f"[OK] Saved review CSV: {out_csv}")
    print(f"[OK] Saved auto-fix TSV (high confidence mismatches): {out_fix}")

    mism = [r for r in rows if r["status"] == "mismatch"]
    print(f"\nMismatch count: {len(mism)} / {len(all_pids)}")
    if mism:
        print("Top mismatches:")
        for r in mism[:10]:
            print(f"- {r['patient_id']}: ann={r['annotation_label']} corpus={r['corpus_evidence_label']} conf={r['evidence_confidence']} tags={r['evidence_tags']}")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
