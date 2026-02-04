#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# HARDCODED PATHS (Windows-safe)
# =============================================================================

DATA_DIR = Path(r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data")
CORPUS_PATH = DATA_DIR / "phantom_ehr_corpus.txt"
ANNOTATIONS_PATH = DATA_DIR / "phantom_annotations_minimal.txt"
OUTPUT_DIR = DATA_DIR / "audit_out"


# =============================================================================
# Regex helpers
# =============================================================================

RE_PATIENT = re.compile(r"^\s*PATIENT_ID:\s*(\d+)\s*$")
RE_STAY_HDR = re.compile(
    r"^\s*===\s*STAY_ID:\s*(\S+)\s*\|\s*DATE:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*\|\s*SERVICE:\s*(.+?)\s*===\s*$"
)

RE_RF_VAL = re.compile(
    r"\bRF\b\s*[:=]?\s*(positif|positive|n[ée]gatif|negative)?\s*(?:à|=|\()?[\s]*([0-9]+(?:[.,][0-9]+)?)\s*(?:UI/mL|IU/mL)?",
    re.IGNORECASE,
)

RE_CCP_VAL = re.compile(
    r"\banti[-\s]?CCP\b\s*[:=]?\s*(positif|positive|n[ée]gatif|negative)?\s*(?:à|=|\()?[\s]*([<>]?\s*[0-9]+(?:[.,][0-9]+)?)\s*(?:U/mL|U\/mL)?",
    re.IGNORECASE,
)

RE_RA_CONFIRMED = re.compile(
    r"\b(polyarthrite\s+rhumato[iï]de|arthrite\s+rhumato[iï]de|PR)\s+(confirm[ée]e|confirm[ée])\b",
    re.IGNORECASE,
)
RE_RA_PROBABLE = re.compile(
    r"\b(polyarthrite\s+rhumato[iï]de|arthrite\s+rhumato[iï]de|PR)\s+(probable|possible|[ée]voqu[ée]e)\b",
    re.IGNORECASE,
)
RE_RA_NEGATED = re.compile(
    r"\b(pas\s+de\s+(polyarthrite\s+rhumato[iï]de|arthrite\s+rhumato[iï]de|PR)|PR\s+[ée]cart[ée]e|diagnostic\s+non\s+retenu|ruled\s+out)\b",
    re.IGNORECASE,
)

RE_MTX = re.compile(r"\b(m[ée]thotrexate|methotrexate|MTX)\b", re.IGNORECASE)
RE_BIOLOGIC = re.compile(r"\b(adalimumab|etanercept|infliximab|tocilizumab|abatacept|rituximab)\b", re.IGNORECASE)
RE_JAK = re.compile(r"\b(tofacitinib|baricitinib|upadacitinib)\b", re.IGNORECASE)

KEYWORDS = {
    "arthrose": re.compile(r"\barthrose\b", re.IGNORECASE),
    "diverticulite": re.compile(r"\bdiverticulite\b", re.IGNORECASE),
    "aspergillose": re.compile(r"\baspergillose\b|\baspergillus\b", re.IGNORECASE),
    "sjogren": re.compile(r"\bSjögren\b|\bsjogren\b", re.IGNORECASE),
    "infection": re.compile(r"\binfection\b|\bpneumonie\b|\bpy[ée]lon[ée]phrite\b", re.IGNORECASE),
}


# =============================================================================
# Data models
# =============================================================================

@dataclass
class StayInfo:
    stay_id: str
    date: Optional[str]
    service: Optional[str]
    text: str


@dataclass
class PatientCorpusSignature:
    patient_id: str
    n_stays: int
    stay_ids: List[str]
    dates: List[str]

    rf_max_value: Optional[float]
    rf_polarity: str

    ccp_max_value: Optional[float]
    ccp_polarity: str

    has_ra_confirmed: bool
    has_ra_probable: bool
    has_ra_negated: bool

    has_mtx: bool
    has_biologic: bool
    has_jak: bool

    keywords_present: List[str]


@dataclass
class AnnotationRow:
    patient_id: str
    label_binary: Optional[int]
    comment: str
    rf_comment_value: Optional[float]
    ccp_comment_value: Optional[float]
    keywords_comment: List[str]


@dataclass
class AuditResultRow:
    patient_id: str
    label_binary: Optional[int]

    corpus_n_stays: int
    corpus_stays: str
    corpus_dates: str

    corpus_rf_max: Optional[float]
    corpus_rf_pol: str
    comment_rf: Optional[float]

    corpus_ccp_max: Optional[float]
    corpus_ccp_pol: str
    comment_ccp: Optional[float]

    corpus_ra_flags: str
    corpus_drugs: str

    corpus_keywords: str
    comment_keywords: str

    compatibility_score: int
    issues: List[str]


# =============================================================================
# Helpers
# =============================================================================

def normalize_patient_id(pid: str) -> str:
    return pid.strip()


def to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = s.strip().replace(",", ".")
    s = re.sub(r"^[<>]\s*", "", s)
    try:
        return float(s)
    except Exception:
        return None


def infer_polarity(word: Optional[str]) -> str:
    if not word:
        return "unknown"
    w = word.lower()
    if "posit" in w:
        return "positive"
    if "nég" in w or "neg" in w:
        return "negative"
    return "unknown"


# =============================================================================
# Parsing
# =============================================================================

def parse_annotations(ann_text: str) -> Dict[str, AnnotationRow]:
    rows: Dict[str, AnnotationRow] = {}
    line_re = re.compile(
        r"PATIENT_ID:\s*(\d+)\s*\|\s*LABEL_BINARY:\s*([01])\s*\|\s*COMMENT:\s*(.*)$"
    )

    for raw in ann_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue

        m = line_re.search(raw)
        if not m:
            continue

        pid = normalize_patient_id(m.group(1))
        label = int(m.group(2))
        comment = m.group(3).strip()

        rf_v = extract_rf_from_comment(comment)
        ccp_v = extract_ccp_from_comment(comment)
        kw_c = [k for k, rx in KEYWORDS.items() if rx.search(comment)]

        rows[pid] = AnnotationRow(
            patient_id=pid,
            label_binary=label,
            comment=comment,
            rf_comment_value=rf_v,
            ccp_comment_value=ccp_v,
            keywords_comment=kw_c,
        )

    return rows


def parse_corpus(corpus_text: str) -> Dict[str, List[StayInfo]]:
    patients: Dict[str, List[StayInfo]] = {}
    cur_pid: Optional[str] = None
    cur_stay_id: Optional[str] = None
    cur_date: Optional[str] = None
    cur_service: Optional[str] = None
    buf_lines: List[str] = []

    def flush_stay():
        nonlocal cur_pid, cur_stay_id, cur_date, cur_service, buf_lines
        if cur_pid and cur_stay_id:
            text = "\n".join(buf_lines).strip()
            patients.setdefault(cur_pid, []).append(
                StayInfo(stay_id=cur_stay_id, date=cur_date, service=cur_service, text=text)
            )
        buf_lines = []

    for line in corpus_text.splitlines():
        m_pid = RE_PATIENT.match(line)
        if m_pid:
            flush_stay()
            cur_pid = normalize_patient_id(m_pid.group(1))
            cur_stay_id, cur_date, cur_service = None, None, None
            continue

        m_stay = RE_STAY_HDR.match(line)
        if m_stay:
            flush_stay()
            cur_stay_id = m_stay.group(1).strip()
            cur_date = m_stay.group(2).strip()
            cur_service = m_stay.group(3).strip()
            continue

        if cur_pid and cur_stay_id:
            buf_lines.append(line)

    flush_stay()
    return patients


# =============================================================================
# Feature extraction
# =============================================================================

def extract_rf_features(text: str) -> Tuple[Optional[float], str]:
    vals: List[Tuple[float, str]] = []
    for m in RE_RF_VAL.finditer(text):
        pol = infer_polarity(m.group(1))
        v = to_float(m.group(2))
        if v is not None:
            vals.append((v, pol))
    if not vals:
        return None, "absent"
    max_v, pol = sorted(vals, key=lambda x: x[0], reverse=True)[0]
    return max_v, pol if pol != "unknown" else "unknown"


def extract_ccp_features(text: str) -> Tuple[Optional[float], str]:
    vals: List[Tuple[float, str]] = []
    for m in RE_CCP_VAL.finditer(text):
        pol = infer_polarity(m.group(1))
        v = to_float(m.group(2))
        if v is not None:
            vals.append((v, pol))
    if not vals:
        return None, "absent"
    max_v, pol = sorted(vals, key=lambda x: x[0], reverse=True)[0]
    return max_v, pol if pol != "unknown" else "unknown"


def extract_ra_flags(text: str) -> Tuple[bool, bool, bool]:
    return bool(RE_RA_CONFIRMED.search(text)), bool(RE_RA_PROBABLE.search(text)), bool(RE_RA_NEGATED.search(text))


def extract_drug_flags(text: str) -> Tuple[bool, bool, bool]:
    return bool(RE_MTX.search(text)), bool(RE_BIOLOGIC.search(text)), bool(RE_JAK.search(text))


def extract_keywords(text: str) -> List[str]:
    return [k for k, rx in KEYWORDS.items() if rx.search(text)]


def extract_rf_from_comment(comment: str) -> Optional[float]:
    m = re.search(r"\bRF\s*\+?\s*(?:[:=]?\s*(?:positif|positive)\s*(?:à|=)\s*)?([0-9]+(?:[.,][0-9]+)?)", comment, re.IGNORECASE)
    return to_float(m.group(1)) if m else None


def extract_ccp_from_comment(comment: str) -> Optional[float]:
    m = re.search(r"\banti[-\s]?CCP\s*\+?\s*(?:[:=]?\s*(?:positif|positive)\s*(?:à|=)\s*)?([0-9]+(?:[.,][0-9]+)?)", comment, re.IGNORECASE)
    return to_float(m.group(1)) if m else None


def build_patient_signature(stays: List[StayInfo], patient_id: str) -> PatientCorpusSignature:
    full_text = "\n".join(
        [f"=== {s.stay_id} {s.date or ''} {s.service or ''} ===\n{s.text}" for s in stays]
    )

    rf_v, rf_pol = extract_rf_features(full_text)
    ccp_v, ccp_pol = extract_ccp_features(full_text)

    ra_conf, ra_prob, ra_neg = extract_ra_flags(full_text)
    has_mtx, has_bio, has_jak = extract_drug_flags(full_text)
    kw = extract_keywords(full_text)

    stay_ids = [s.stay_id for s in stays]
    dates = [s.date for s in stays if s.date]

    return PatientCorpusSignature(
        patient_id=patient_id,
        n_stays=len(stays),
        stay_ids=stay_ids,
        dates=dates,
        rf_max_value=rf_v,
        rf_polarity=rf_pol,
        ccp_max_value=ccp_v,
        ccp_polarity=ccp_pol,
        has_ra_confirmed=ra_conf,
        has_ra_probable=ra_prob,
        has_ra_negated=ra_neg,
        has_mtx=has_mtx,
        has_biologic=has_bio,
        has_jak=has_jak,
        keywords_present=kw,
    )


# =============================================================================
# Compatibility scoring
# =============================================================================

def compatibility_score(sig: PatientCorpusSignature, ann: Optional[AnnotationRow]) -> Tuple[int, List[str]]:
    issues: List[str] = []
    score = 100

    if ann is None:
        return 0, ["missing_annotation"]

    if ann.rf_comment_value is not None:
        if sig.rf_max_value is None:
            score -= 25
            issues.append(f"rf_in_comment_but_absent_in_corpus({ann.rf_comment_value})")
        else:
            if abs(sig.rf_max_value - ann.rf_comment_value) > 5.0:
                score -= 30
                issues.append(f"rf_value_mismatch(corpus={sig.rf_max_value},comment={ann.rf_comment_value})")

    if ann.ccp_comment_value is not None:
        if sig.ccp_max_value is None:
            score -= 25
            issues.append(f"ccp_in_comment_but_absent_in_corpus({ann.ccp_comment_value})")
        else:
            if abs(sig.ccp_max_value - ann.ccp_comment_value) > 5.0:
                score -= 30
                issues.append(f"ccp_value_mismatch(corpus={sig.ccp_max_value},comment={ann.ccp_comment_value})")

    kw_comment = set(ann.keywords_comment)
    kw_corpus = set(sig.keywords_present)
    missing_kw = sorted(list(kw_comment - kw_corpus))
    if missing_kw:
        score -= min(20, 5 * len(missing_kw))
        issues.append(f"comment_keywords_missing_in_corpus({','.join(missing_kw)})")

    # label sanity (soft)
    if ann.label_binary == 1:
        if sig.has_ra_negated and not sig.has_ra_confirmed and not sig.has_ra_probable:
            score -= 20
            issues.append("label_positive_but_corpus_has_ra_negated_only")
    else:
        if sig.has_ra_confirmed and sig.has_biologic:
            score -= 20
            issues.append("label_negative_but_corpus_has_ra_confirmed_and_biologic")

    score = max(0, min(100, score))
    return score, issues


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("AUDIT: Annotations vs Corpus (HARDCODED PATHS)")
    print("=" * 80)

    if not CORPUS_PATH.exists():
        print(f"[ERROR] Corpus file not found:\n  {CORPUS_PATH}")
        return
    if not ANNOTATIONS_PATH.exists():
        print(f"[ERROR] Annotation file not found:\n  {ANNOTATIONS_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_text = CORPUS_PATH.read_text(encoding="utf-8", errors="replace")
    ann_text = ANNOTATIONS_PATH.read_text(encoding="utf-8", errors="replace")

    corpus_patients = parse_corpus(corpus_text)
    ann_rows = parse_annotations(ann_text)

    signatures: Dict[str, PatientCorpusSignature] = {}
    for pid, stays in corpus_patients.items():
        signatures[pid] = build_patient_signature(stays, pid)

    all_pids = sorted(set(signatures.keys()) | set(ann_rows.keys()))

    report_csv = OUTPUT_DIR / "audit_report.csv"
    report_jsonl = OUTPUT_DIR / "audit_report.jsonl"

    rows: List[AuditResultRow] = []

    for pid in all_pids:
        sig = signatures.get(pid)
        ann = ann_rows.get(pid)

        if sig is None:
            r = AuditResultRow(
                patient_id=pid,
                label_binary=ann.label_binary if ann else None,
                corpus_n_stays=0,
                corpus_stays="",
                corpus_dates="",
                corpus_rf_max=None,
                corpus_rf_pol="absent",
                comment_rf=ann.rf_comment_value if ann else None,
                corpus_ccp_max=None,
                corpus_ccp_pol="absent",
                comment_ccp=ann.ccp_comment_value if ann else None,
                corpus_ra_flags="missing_in_corpus",
                corpus_drugs="",
                corpus_keywords="",
                comment_keywords=",".join(ann.keywords_comment) if ann else "",
                compatibility_score=0,
                issues=["missing_in_corpus"],
            )
            rows.append(r)
            continue

        score, issues = compatibility_score(sig, ann)
        ra_flags = f"confirmed={sig.has_ra_confirmed}|probable={sig.has_ra_probable}|negated={sig.has_ra_negated}"
        drugs = f"mtx={sig.has_mtx}|biologic={sig.has_biologic}|jak={sig.has_jak}"
        corpus_kw = ",".join(sig.keywords_present)

        r = AuditResultRow(
            patient_id=pid,
            label_binary=ann.label_binary if ann else None,
            corpus_n_stays=sig.n_stays,
            corpus_stays=",".join(sig.stay_ids),
            corpus_dates=",".join(sig.dates),
            corpus_rf_max=sig.rf_max_value,
            corpus_rf_pol=sig.rf_polarity,
            comment_rf=ann.rf_comment_value if ann else None,
            corpus_ccp_max=sig.ccp_max_value,
            corpus_ccp_pol=sig.ccp_polarity,
            comment_ccp=ann.ccp_comment_value if ann else None,
            corpus_ra_flags=ra_flags,
            corpus_drugs=drugs,
            corpus_keywords=corpus_kw,
            comment_keywords=",".join(ann.keywords_comment) if ann else "",
            compatibility_score=score,
            issues=issues,
        )
        rows.append(r)

    # CSV
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "patient_id", "label_binary",
            "corpus_n_stays", "corpus_stays", "corpus_dates",
            "corpus_rf_max", "corpus_rf_pol", "comment_rf",
            "corpus_ccp_max", "corpus_ccp_pol", "comment_ccp",
            "corpus_ra_flags", "corpus_drugs",
            "corpus_keywords", "comment_keywords",
            "compatibility_score", "issues"
        ])
        for r in rows:
            w.writerow([
                r.patient_id, r.label_binary,
                r.corpus_n_stays, r.corpus_stays, r.corpus_dates,
                r.corpus_rf_max, r.corpus_rf_pol, r.comment_rf,
                r.corpus_ccp_max, r.corpus_ccp_pol, r.comment_ccp,
                r.corpus_ra_flags, r.corpus_drugs,
                r.corpus_keywords, r.comment_keywords,
                r.compatibility_score, ";".join(r.issues)
            ])

    # JSONL
    with report_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Console summary
    mismatches = sorted(rows, key=lambda x: x.compatibility_score)

    print(f"\n[OK] Corpus patients:       {len(signatures)}")
    print(f"[OK] Annotation patients:   {len(ann_rows)}")
    print(f"[OK] Union patients:        {len(rows)}")
    print(f"[OK] Output directory:      {OUTPUT_DIR}")
    print(f"[OK] Saved CSV:             {report_csv}")
    print(f"[OK] Saved JSONL:           {report_jsonl}")

    print("\nTop mismatches (lowest compatibility_score):")
    for r in mismatches[:10]:
        print(f"- {r.patient_id}: score={r.compatibility_score} label={r.label_binary} issues={r.issues}")

    scores = [r.compatibility_score for r in rows]
    if scores:
        avg = sum(scores) / len(scores)
        n_bad = sum(1 for s in scores if s < 60)
        print(f"\nAvg compatibility_score: {avg:.1f}")
        print(f"Patients with score < 60: {n_bad}/{len(scores)}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
