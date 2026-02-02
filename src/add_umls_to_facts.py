import json
from typing import Dict, Any, List
import sys

sys.path.append(r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\Scripts utils")
from umls_mapper import UMLSMapper

ACRONYM_EXPAND = {
    "PR": ["polyarthrite rhumatoïde", "rheumatoid arthritis", "PR polyarthrite rhumatoïde"],
    "RF": ["rheumatoid factor", "facteur rhumatoïde", "RF rheumatoid factor"],
    "ACPA": ["anti-citrullinated protein antibodies", "anticorps anti-citrullinés", "anti-CCP", "ACPA anti-CCP"],
    "ANTI-CCP": ["anti-CCP", "anti-CCP antibody", "anticorps anti-CCP"],
    "MTX": ["methotrexate", "méthotrexate", "MTX methotrexate"],
}

CANON_QUERY = {
    "lab_RF": ["rheumatoid factor", "facteur rhumatoïde"],
    "lab_CCP": ["anti-CCP", "anti-citrullinated protein antibodies", "ACPA"],
    "drug_MTX": ["methotrexate", "MTX"],
}

# ----------------------------
# Robust helpers
# ----------------------------
def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _ensure_dm_item(x: Any) -> Any:
    if isinstance(x, dict):
        if "mention" not in x and "term" in x:
            x["mention"] = x["term"]
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        return {"mention": s, "status": "mentioned", "evidence": []}
    return None

def _ensure_lab_item(x: Any) -> Any:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        return {"name": s, "value": "unknown", "evidence": []}
    return None

def _ensure_drug_item(x: Any) -> Any:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        return {"name": s, "status": "mentioned", "dose": None, "evidence": []}
    return None

def pick_best(mapper, queries, topk=5, sab_prefer=None):
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        hits = mapper.lookup(q, topk=topk, sab_prefer=sab_prefer)
        if hits:
            return {"query_used": q, "hit": hits[0]}
    return None

def get_fact_container(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports both schemas:
      - stay-level: root has disease_mentions/labs/drugs
      - patient-level: obj["facts"] has disease_mentions/labs/drugs
    Returns the dict to mutate (either obj or obj["facts"])
    """
    if isinstance(obj.get("facts"), dict):
        return obj["facts"]
    return obj

# ----------------------------
# Main UMLS injection
# ----------------------------
def add_umls_fields(mapper, obj: Dict[str, Any]) -> Dict[str, Any]:
    warnings = []
    facts = get_fact_container(obj)

    # Normalize containers
    facts["disease_mentions"] = as_list(facts.get("disease_mentions"))
    facts["labs"] = as_list(facts.get("labs"))
    facts["drugs"] = as_list(facts.get("drugs"))

    # Normalize disease mention items
    norm_dm = []
    for dm in facts["disease_mentions"]:
        fixed = _ensure_dm_item(dm)
        if fixed is None:
            warnings.append("disease_mentions_non_parseable_item")
            continue
        norm_dm.append(fixed)
    facts["disease_mentions"] = norm_dm

    # Normalize lab items
    norm_labs = []
    for lab in facts["labs"]:
        fixed = _ensure_lab_item(lab)
        if fixed is None:
            warnings.append("labs_non_parseable_item")
            continue
        norm_labs.append(fixed)
    facts["labs"] = norm_labs

    # Normalize drug items
    norm_drugs = []
    for dr in facts["drugs"]:
        fixed = _ensure_drug_item(dr)
        if fixed is None:
            warnings.append("drugs_non_parseable_item")
            continue
        norm_drugs.append(fixed)
    facts["drugs"] = norm_drugs

    # disease_mentions -> umls
    for dm in facts.get("disease_mentions", []):
        mention = (dm.get("mention") or dm.get("term") or "").strip()
        queries = ACRONYM_EXPAND.get(mention.upper(), [mention])
        best = pick_best(mapper, queries, topk=5, sab_prefer=["SNOMEDCT_US", "MSH", "MSHFRE"])
        dm["umls"] = best

    # labs -> umls
    for lab in facts.get("labs", []):
        name = (lab.get("name") or "").strip().lower()

        if name in ["rf", "facteur rhumatoïde", "rheumatoid factor"] or "facteur rhumatoïde" in name:
            queries = CANON_QUERY["lab_RF"]
        else:
            queries = CANON_QUERY["lab_CCP"]

        best = pick_best(mapper, queries, topk=5, sab_prefer=["SNOMEDCT_US", "MSH", "MSHFRE"])
        lab["umls"] = best

    # drugs -> umls
    for dr in facts.get("drugs", []):
        dname = (dr.get("name") or "").strip().lower()

        if "methotrex" in dname or dname == "mtx":
            queries = CANON_QUERY["drug_MTX"]
        else:
            queries = [dr.get("name") or ""]

        best = pick_best(mapper, queries, topk=5, sab_prefer=["RXNORM", "SNOMEDCT_US", "MSH"])
        dr["umls"] = best

    if warnings:
        obj.setdefault("_schema_warnings", [])
        obj["_schema_warnings"].extend(sorted(set(warnings)))

    return obj

def main():
    UMLS_DB = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\Scripts utils\umls.sqlite"
    IN_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\data\facts_agent1_stay.jsonl"
    OUT_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\data\facts_agent1_stay_umls.jsonl"

    mapper = UMLSMapper(UMLS_DB)

    n_total = 0
    n_skip = 0
    n_warn = 0

    with open(IN_PATH, "r", encoding="utf-8") as f_in, open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            n_total += 1
            obj = json.loads(line)

            # keep error lines as-is
            if "error" in obj or obj.get("extraction_status") == "error":
                n_skip += 1
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            obj2 = add_umls_fields(mapper, obj)
            if obj2.get("_schema_warnings"):
                n_warn += 1

            f_out.write(json.dumps(obj2, ensure_ascii=False) + "\n")

    mapper.close()
    print(f"✅ Done: {OUT_PATH}")
    print(f"lines_total={n_total} | skipped_errors={n_skip} | with_warnings={n_warn}")

if __name__ == "__main__":
    main()
