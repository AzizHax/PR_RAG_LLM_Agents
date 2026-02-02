#!/usr/bin/env python3
"""
Agent 1: KB-Guided Extraction + RAG Grounding for RA Phenotyping
- Load KB spec and retrieve relevant slots
- Prefilter → Chunking → BM25 → Semantic Rerank
- LLM extraction with KB-guided prompt
- Backfill v3 regex fallback
- Output: stay-level facts with RAG metadata
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# External dependencies
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("pip install rank-bm25")

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RAGMetadata:
    """RAG scoring and retrieval metadata"""
    rag_score_max: float = 0.0
    rag_score_sum_top3: float = 0.0
    selected_windows: List[str] = None
    kb_slots_used: List[str] = None
    bm25_top_chunks: List[int] = None
    rerank_scores: List[float] = None
    
    def __post_init__(self):
        if self.selected_windows is None:
            self.selected_windows = []
        if self.kb_slots_used is None:
            self.kb_slots_used = []
        if self.bm25_top_chunks is None:
            self.bm25_top_chunks = []
        if self.rerank_scores is None:
            self.rerank_scores = []


@dataclass
class TimingInfo:
    """Timing information for performance monitoring"""
    prefilter_ms: float = 0.0
    chunking_ms: float = 0.0
    bm25_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0
    backfill_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class StayFacts:
    """Extracted facts for a single stay"""
    stay_id: str
    patient_id: str
    disease_mentions: List[Dict[str, Any]]
    labs: List[Dict[str, Any]]
    drugs: List[Dict[str, Any]]
    _rag_meta: RAGMetadata
    _timing: TimingInfo
    _extraction_mode: str = "llm"  # llm, regex_fallback, or backfill
    

# ============================================================================
# KB MANAGER
# ============================================================================

class KnowledgeBaseManager:
    """Manages KB loading and slot retrieval"""
    
    def __init__(self, kb_path: str):
        self.kb_path = Path(kb_path)
        self.kb = self._load_kb()
        self.slot_index = self._build_slot_index()
        
    def _load_kb(self) -> Dict[str, Any]:
        """Load KB from JSON file"""
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_slot_index(self) -> Dict[str, Dict[str, Any]]:
        """Build index of slots for fast retrieval"""
        index = {}
        for slot in self.kb.get('slots', []):
            slot_name = slot.get('slot')
            if slot_name:
                index[slot_name] = slot
        return index
    
    def retrieve_relevant_slots(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant KB slots using BM25"""
        # Build corpus from slot descriptions and patterns
        corpus = []
        slot_names = []
        
        for slot_name, slot_data in self.slot_index.items():
            # Combine description and patterns for better matching
            text_parts = [slot_data.get('description', '')]
            
            # Add all patterns
            patterns = slot_data.get('patterns', {})
            for pattern_list in patterns.values():
                if isinstance(pattern_list, list):
                    text_parts.extend(pattern_list)
                elif isinstance(pattern_list, dict):
                    for sub_list in pattern_list.values():
                        if isinstance(sub_list, list):
                            text_parts.extend(sub_list)
            
            corpus.append(' '.join(str(p).lower() for p in text_parts))
            slot_names.append(slot_name)
        
        if not corpus:
            return []
        
        # BM25 retrieval
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [self.slot_index[slot_names[i]] for i in top_indices if scores[i] > 0]
    
    def get_slot_prompt_text(self, slot: Dict[str, Any]) -> str:
        """Generate prompt text for a KB slot"""
        slot_name = slot.get('slot', 'unknown')
        description = slot.get('description', '')
        
        # Extract key patterns
        patterns = slot.get('patterns', {})
        positive_patterns = patterns.get('positive', patterns.get('anchor', []))
        negation_patterns = patterns.get('negation', [])
        
        prompt_parts = [f"**{slot_name}**: {description}"]
        
        if positive_patterns:
            prompt_parts.append(f"  - Look for: {', '.join(positive_patterns[:5])}")
        
        if negation_patterns:
            prompt_parts.append(f"  - Negations: {', '.join(negation_patterns[:3])}")
        
        # Add pitfalls if present
        pitfalls = slot.get('pitfalls', [])
        if pitfalls:
            prompt_parts.append(f"  - Avoid: {', '.join(pitfalls[:3])}")
        
        return '\n'.join(prompt_parts)


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """Handles text splitting and preprocessing"""
    
    @staticmethod
    def split_by_patients(corpus: str) -> List[Tuple[str, str]]:
        """Split corpus by PATIENT_ID markers"""
        patient_blocks = []
        current_patient = None
        current_text = []
        
        for line in corpus.split('\n'):
            if line.startswith('PATIENT_ID:'):
                if current_patient and current_text:
                    patient_blocks.append((current_patient, '\n'.join(current_text)))
                current_patient = line.split(':', 1)[1].strip()
                current_text = []
            else:
                current_text.append(line)
        
        if current_patient and current_text:
            patient_blocks.append((current_patient, '\n'.join(current_text)))
        
        return patient_blocks
    
    @staticmethod
    def split_by_stays(patient_text: str) -> List[Tuple[str, str]]:
        """Split patient text by STAY_ID markers"""
        stays = []
        current_stay = None
        current_lines = []
        
        for line in patient_text.split('\n'):
            if line.startswith('===') and 'STAY_ID:' in line:
                if current_stay and current_lines:
                    stays.append((current_stay, '\n'.join(current_lines)))
                match = re.search(r'STAY_ID:\s*(\S+)', line)
                current_stay = match.group(1) if match else "unknown"
                current_lines = []
            else:
                current_lines.append(line)
        
        if current_stay and current_lines:
            stays.append((current_stay, '\n'.join(current_lines)))
        
        return stays


# ============================================================================
# PREFILTERING
# ============================================================================

class SafeAnchorPrefilter:
    """Prefilter lines using safe anchors with context window"""
    
    SAFE_ANCHORS = {
        'ra_long': [
            r'\bpolyarthrite\s+rhumatoïde\b',
            r'\barthrite\s+rhumatoïde\b',
            r'\brheumatoid\s+arthritis\b',
        ],
        'ra_abbrev': [
            r'\bPR\b(?!\s*(artérielle|interval|prostate))',
            r'\bRA\b(?!\s*(artérielle|right\s+atrium))',
        ],
        'rf': [
            r'\bRF\b',
            r'\bfacteur\s+rhumatoïde\b',
            r'\brheumatoid\s+factor\b',
        ],
        'anti_ccp': [
            r'\banti[- ]?CCP\b',
            r'\bACPA\b',
            r'\banti[- ]?citrullin',
        ],
        'dmards': [
            r'\bmethotrexate\b',
            r'\bméthotrexate\b',
            r'\bMTX\b',
            r'\badalimumab\b',
            r'\betanercept\b',
            r'\binfliximab\b',
            r'\btocilizumab\b',
            r'\btofacitinib\b',
            r'\bbaricitinib\b',
        ]
    }
    
    def __init__(self, context_window: int = 2):
        self.context_window = context_window
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile all anchor patterns"""
        patterns = []
        for category_patterns in self.SAFE_ANCHORS.values():
            for pattern in category_patterns:
                patterns.append(re.compile(pattern, re.IGNORECASE))
        return patterns
    
    def filter_lines(self, lines: List[str]) -> List[Tuple[int, str]]:
        """Filter lines matching anchors with context window"""
        matched_indices = set()
        
        # Find all matching lines
        for i, line in enumerate(lines):
            for pattern in self.patterns:
                if pattern.search(line):
                    matched_indices.add(i)
                    break
        
        # Expand with context window
        expanded_indices = set()
        for idx in matched_indices:
            for offset in range(-self.context_window, self.context_window + 1):
                expanded_idx = idx + offset
                if 0 <= expanded_idx < len(lines):
                    expanded_indices.add(expanded_idx)
        
        # Return sorted (line_no, text) tuples
        return [(i, lines[i]) for i in sorted(expanded_indices)]


# ============================================================================
# CHUNKING & RETRIEVAL
# ============================================================================

class ChunkRetriever:
    """Handles chunking and BM25 retrieval"""
    
    def __init__(self, chunk_size: int = 6, stride: int = 3, top_k: int = 10):
        self.chunk_size = chunk_size
        self.stride = stride
        self.top_k = top_k
    
    def create_chunks(self, numbered_lines: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """Create overlapping chunks from numbered lines"""
        chunks = []
        lines = [text for _, text in numbered_lines]
        line_nos = [no for no, _ in numbered_lines]
        
        for i in range(0, len(lines), self.stride):
            chunk_lines = lines[i:i + self.chunk_size]
            chunk_line_nos = line_nos[i:i + self.chunk_size]
            
            if chunk_lines:
                chunks.append({
                    'text': ' '.join(chunk_lines),
                    'lines': chunk_lines,
                    'line_nos': chunk_line_nos,
                    'start_idx': i,
                })
        
        return chunks
    
    def retrieve_bm25(self, chunks: List[Dict[str, Any]], query: str) -> List[int]:
        """Retrieve top-k chunks using BM25"""
        if not chunks:
            return []
        
        corpus = [chunk['text'] for chunk in chunks]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k chunk indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        return top_indices


# ============================================================================
# SEMANTIC RERANKING
# ============================================================================

class SemanticReranker:
    """Semantic reranking using heuristic scoring"""
    
    POSITIVE_INDICATORS = {
        'anti_ccp_pos': [r'\banti[- ]?CCP\s+(positif|positive|\+|élevé)', 3.0],
        'rf_pos': [r'\b(RF|facteur\s+rhumatoïde)\s+(positif|positive|\+|élevé)', 2.5],
        'confirmed': [r'\b(confirmé|diagnostic\s+retenu|établi)', 2.0],
        'mtx': [r'\b(methotrexate|méthotrexate|MTX)\b', 1.5],
        'biologic': [r'\b(adalimumab|etanercept|infliximab|tocilizumab)', 2.0],
        'jak': [r'\b(tofacitinib|baricitinib|upadacitinib)', 2.0],
    }
    
    NEGATIVE_INDICATORS = {
        'excluded': [r'\b(écarté|exclu|non\s+retenu|ruled\s+out)', -3.0],
        'negative': [r'\b(négatif|negative)\b', -1.0],
    }
    
    def __init__(self):
        self.pos_patterns = [(re.compile(p, re.IGNORECASE), w) for p, w in self.POSITIVE_INDICATORS.values()]
        self.neg_patterns = [(re.compile(p, re.IGNORECASE), w) for p, w in self.NEGATIVE_INDICATORS.values()]
    
    def score_chunk(self, chunk_text: str) -> float:
        """Compute heuristic score for a chunk"""
        score = 0.0
        
        for pattern, weight in self.pos_patterns:
            if pattern.search(chunk_text):
                score += weight
        
        for pattern, weight in self.neg_patterns:
            if pattern.search(chunk_text):
                score += weight
        
        return score
    
    def rerank_chunks(self, chunks: List[Dict[str, Any]], bm25_indices: List[int]) -> Tuple[List[int], List[float]]:
        """Rerank BM25 results using semantic scoring"""
        scored_chunks = []
        
        for idx in bm25_indices:
            chunk = chunks[idx]
            score = self.score_chunk(chunk['text'])
            scored_chunks.append((idx, score))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        reranked_indices = [idx for idx, _ in scored_chunks]
        scores = [score for _, score in scored_chunks]
        
        return reranked_indices, scores


# ============================================================================
# LLM EXTRACTION
# ============================================================================

class LLMExtractor:
    """Handles LLM-based extraction with KB guidance"""
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:3b-instruct",
                 max_context_lines: int = 45,
                 temperature: float = 0.1,
                 timeout: int = 120,
                 max_retries: int = 2):
        self.ollama_url = ollama_url
        self.model = model
        self.max_context_lines = max_context_lines
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
    
    def _build_prompt(self, 
                      numbered_lines: List[Tuple[int, str]], 
                      kb_slots: List[Dict[str, Any]],
                      kb_manager: KnowledgeBaseManager) -> str:
        """Build extraction prompt with KB guidance"""
        
        # Format numbered lines
        context_lines = '\n'.join(f"{no:03d}| {text}" for no, text in numbered_lines[:self.max_context_lines])
        
        # Format KB slots
        kb_guidance = ""
        if kb_slots:
            kb_texts = [kb_manager.get_slot_prompt_text(slot) for slot in kb_slots]
            kb_guidance = "\n\n".join(kb_texts)
        
        prompt = f"""You are a medical NLP expert extracting structured information about Rheumatoid Arthritis (RA) from clinical notes.

# KB EXTRACTION GUIDANCE
{kb_guidance if kb_guidance else "No specific KB slots provided."}

# CLINICAL TEXT (numbered lines)
{context_lines}

# EXTRACTION TASK
Extract the following information in strict JSON format:

1. **disease_mentions**: Array of RA mentions with:
   - entity: full disease name
   - status: "confirmed" | "suspected" | "negated" | "mentioned"
   - evidence: {{ stay_id, line_no, snippet }} (exact text from lines above)

2. **labs**: Array of lab results with:
   - test: test name (e.g., "RF", "anti-CCP", "CRP")
   - value: numeric value or "positive"/"negative"
   - unit: if applicable
   - polarity: "positive" | "negative" | "unknown"
   - evidence: {{ stay_id, line_no, snippet }}

3. **drugs**: Array of medications with:
   - name: generic drug name
   - category: "csDMARD" | "bDMARD" | "tsDMARD" | "corticosteroid"
   - evidence: {{ stay_id, line_no, snippet }}

# CRITICAL RULES
- snippet MUST be exact text from the numbered lines above
- line_no MUST match the line number (001, 002, etc.)
- If no information found, return empty arrays []
- DO NOT hallucinate or invent information

# OUTPUT FORMAT (strict JSON)
{{
  "disease_mentions": [...],
  "labs": [...],
  "drugs": [...]
}}"""
        
        return prompt
    
    def extract(self, 
                numbered_lines: List[Tuple[int, str]], 
                stay_id: str,
                kb_slots: List[Dict[str, Any]],
                kb_manager: KnowledgeBaseManager) -> Optional[Dict[str, Any]]:
        """Call LLM to extract structured facts with retry logic"""
        
        prompt = self._build_prompt(numbered_lines, kb_slots, kb_manager)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": self.temperature,
                        "format": "json",
                    },
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    print(f"[LLM Error] HTTP {response.status_code} (attempt {attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                        continue
                    return None
                
                result = response.json()
                output_text = result.get('response', '').strip()
                
                # Parse JSON
                parsed = json.loads(output_text)
                
                # Validate structure
                if not self._validate_output(parsed, stay_id):
                    print(f"[LLM Error] Invalid output structure (attempt {attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    return None
                
                return parsed
                
            except requests.exceptions.Timeout as e:
                print(f"[LLM Error] Timeout after {self.timeout}s (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    # Increase timeout for next attempt
                    self.timeout = min(self.timeout + 60, 300)  # Cap at 5 minutes
                    print(f"  → Retrying with timeout={self.timeout}s...")
                    time.sleep(2)
                    continue
                return None
                
            except (requests.RequestException, json.JSONDecodeError) as e:
                print(f"[LLM Error] {type(e).__name__}: {e} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
                
            except Exception as e:
                print(f"[LLM Error] Unexpected: {type(e).__name__}: {e}")
                return None
        
        return None
    
    def _validate_output(self, parsed: Dict[str, Any], stay_id: str) -> bool:
        """Validate extracted JSON structure"""
        required_keys = {'disease_mentions', 'labs', 'drugs'}
        
        if not all(k in parsed for k in required_keys):
            return False
        
        # Validate each entity has evidence
        for entity_list in parsed.values():
            if not isinstance(entity_list, list):
                return False
            
            for entity in entity_list:
                if 'evidence' not in entity:
                    return False
                
                evidence = entity['evidence']
                if not isinstance(evidence, dict):
                    return False
                
                if 'stay_id' not in evidence or 'line_no' not in evidence or 'snippet' not in evidence:
                    return False
        
        return True


# ============================================================================
# REGEX FALLBACK
# ============================================================================

class RegexFallback:
    """Regex-based extraction fallback"""
    
    DISEASE_PATTERNS = [
        (r'polyarthrite\s+rhumatoïde', 'polyarthrite rhumatoïde'),
        (r'arthrite\s+rhumatoïde', 'arthrite rhumatoïde'),
        (r'rheumatoid\s+arthritis', 'rheumatoid arthritis'),
        (r'\bPR\b(?!\s*artérielle)', 'PR'),
    ]
    
    LAB_PATTERNS = [
        (r'\b(RF|facteur\s+rhumatoïde)\s*(positif|positive|\+)', 'RF', 'positive'),
        (r'\b(RF|facteur\s+rhumatoïde)\s*(négatif|negative|-)', 'RF', 'negative'),
        (r'\banti[- ]?CCP\s*(positif|positive|\+)', 'anti-CCP', 'positive'),
        (r'\banti[- ]?CCP\s*(négatif|negative|-)', 'anti-CCP', 'negative'),
    ]
    
    DRUG_PATTERNS = [
        (r'\b(methotrexate|méthotrexate|MTX)\b', 'methotrexate', 'csDMARD'),
        (r'\badalimumab\b', 'adalimumab', 'bDMARD'),
        (r'\betanercept\b', 'etanercept', 'bDMARD'),
        (r'\binfliximab\b', 'infliximab', 'bDMARD'),
        (r'\btocilizumab\b', 'tocilizumab', 'bDMARD'),
        (r'\btofacitinib\b', 'tofacitinib', 'tsDMARD'),
        (r'\bbaricitinib\b', 'baricitinib', 'tsDMARD'),
    ]
    
    @classmethod
    def extract_from_lines(cls, numbered_lines: List[Tuple[int, str]], stay_id: str) -> Dict[str, Any]:
        """Extract using regex patterns"""
        result = {
            'disease_mentions': [],
            'labs': [],
            'drugs': []
        }
        
        for line_no, text in numbered_lines:
            # Disease mentions
            for pattern, entity in cls.DISEASE_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    result['disease_mentions'].append({
                        'entity': entity,
                        'status': 'mentioned',
                        'evidence': {
                            'stay_id': stay_id,
                            'line_no': line_no,
                            'snippet': text[:100]
                        }
                    })
            
            # Labs
            for pattern, test, polarity in cls.LAB_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    result['labs'].append({
                        'test': test,
                        'polarity': polarity,
                        'evidence': {
                            'stay_id': stay_id,
                            'line_no': line_no,
                            'snippet': text[:100]
                        }
                    })
            
            # Drugs
            for pattern, drug, category in cls.DRUG_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    result['drugs'].append({
                        'name': drug,
                        'category': category,
                        'evidence': {
                            'stay_id': stay_id,
                            'line_no': line_no,
                            'snippet': text[:100]
                        }
                    })
        
        return result


# ============================================================================
# BACKFILL V3
# ============================================================================

class BackfillV3:
    """Backfill extraction results with additional regex patterns"""
    
    @classmethod
    def backfill(cls, extracted: Dict[str, Any], numbered_lines: List[Tuple[int, str]], stay_id: str) -> Dict[str, Any]:
        """Add missing entities via regex on numbered lines"""
        
        # Get regex extraction
        regex_result = RegexFallback.extract_from_lines(numbered_lines, stay_id)
        
        # Merge results (avoid duplicates based on entity/test/drug name)
        result = extracted.copy()
        
        # Merge disease mentions
        existing_diseases = {m['entity'].lower() for m in result.get('disease_mentions', [])}
        for mention in regex_result.get('disease_mentions', []):
            if mention['entity'].lower() not in existing_diseases:
                result.setdefault('disease_mentions', []).append(mention)
        
        # Merge labs
        existing_labs = {lab['test'].lower() for lab in result.get('labs', [])}
        for lab in regex_result.get('labs', []):
            if lab['test'].lower() not in existing_labs:
                result.setdefault('labs', []).append(lab)
        
        # Merge drugs
        existing_drugs = {drug['name'].lower() for drug in result.get('drugs', [])}
        for drug in regex_result.get('drugs', []):
            if drug['name'].lower() not in existing_drugs:
                result.setdefault('drugs', []).append(drug)
        
        return result


# ============================================================================
# AGENT 1 PIPELINE
# ============================================================================

class Agent1Pipeline:
    """Complete Agent 1 pipeline with KB-guided extraction"""
    
    def __init__(self,
                 kb_path: str,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:3b-instruct",
                 prefilter_window: int = 2,
                 chunk_size: int = 6,
                 chunk_stride: int = 3,
                 bm25_top_k: int = 10,
                 kb_slots_top_k: int = 5,
                 max_context_lines: int = 45,
                 llm_timeout: int = 120,
                 llm_max_retries: int = 2):
        
        self.kb_manager = KnowledgeBaseManager(kb_path)
        self.prefilter = SafeAnchorPrefilter(context_window=prefilter_window)
        self.chunker = ChunkRetriever(chunk_size=chunk_size, stride=chunk_stride, top_k=bm25_top_k)
        self.reranker = SemanticReranker()
        self.llm_extractor = LLMExtractor(
            ollama_url=ollama_url,
            model=model,
            max_context_lines=max_context_lines,
            timeout=llm_timeout,
            max_retries=llm_max_retries
        )
        self.kb_slots_top_k = kb_slots_top_k
    
    def process_stay(self, stay_id: str, patient_id: str, stay_text: str) -> StayFacts:
        """Process a single stay through complete pipeline"""
        
        timing = TimingInfo()
        t_start = time.time()
        
        # Stage A: Prefilter
        t0 = time.time()
        lines = stay_text.split('\n')
        numbered_lines = self.prefilter.filter_lines(lines)
        timing.prefilter_ms = (time.time() - t0) * 1000
        
        if not numbered_lines:
            # No relevant lines found
            return StayFacts(
                stay_id=stay_id,
                patient_id=patient_id,
                disease_mentions=[],
                labs=[],
                drugs=[],
                _rag_meta=RAGMetadata(),
                _timing=timing,
                _extraction_mode="empty"
            )
        
        # Stage B: Chunking + BM25
        t0 = time.time()
        chunks = self.chunker.create_chunks(numbered_lines)
        timing.chunking_ms = (time.time() - t0) * 1000
        
        query = "rheumatoid arthritis RA RF anti-CCP methotrexate biologic"
        
        t0 = time.time()
        bm25_indices = self.chunker.retrieve_bm25(chunks, query)
        timing.bm25_ms = (time.time() - t0) * 1000
        
        # Stage C: Semantic rerank
        t0 = time.time()
        reranked_indices, rerank_scores = self.reranker.rerank_chunks(chunks, bm25_indices)
        timing.rerank_ms = (time.time() - t0) * 1000
        
        # Compute RAG metadata
        rag_meta = RAGMetadata(
            rag_score_max=max(rerank_scores) if rerank_scores else 0.0,
            rag_score_sum_top3=sum(rerank_scores[:3]) if len(rerank_scores) >= 3 else sum(rerank_scores),
            selected_windows=[chunks[i]['text'][:100] for i in reranked_indices[:3]],
            bm25_top_chunks=bm25_indices,
            rerank_scores=rerank_scores
        )
        
        # Retrieve KB slots
        kb_slots = self.kb_manager.retrieve_relevant_slots(query, top_k=self.kb_slots_top_k)
        rag_meta.kb_slots_used = [slot['slot'] for slot in kb_slots]
        
        # Select lines for LLM (top reranked chunks)
        selected_line_nos = set()
        for idx in reranked_indices[:5]:  # Top 5 chunks
            selected_line_nos.update(chunks[idx]['line_nos'])
        
        context_lines = [(no, text) for no, text in numbered_lines if no in selected_line_nos]
        context_lines.sort(key=lambda x: x[0])  # Sort by line number
        
        # Stage D: LLM Extraction
        t0 = time.time()
        extracted = self.llm_extractor.extract(context_lines, stay_id, kb_slots, self.kb_manager)
        timing.llm_ms = (time.time() - t0) * 1000
        
        extraction_mode = "llm"
        
        if extracted is None:
            # Fallback to regex
            extracted = RegexFallback.extract_from_lines(numbered_lines, stay_id)
            extraction_mode = "regex_fallback"
        
        # Stage E: Backfill
        t0 = time.time()
        final_extracted = BackfillV3.backfill(extracted, numbered_lines, stay_id)
        timing.backfill_ms = (time.time() - t0) * 1000
        
        timing.total_ms = (time.time() - t_start) * 1000
        
        return StayFacts(
            stay_id=stay_id,
            patient_id=patient_id,
            disease_mentions=final_extracted.get('disease_mentions', []),
            labs=final_extracted.get('labs', []),
            drugs=final_extracted.get('drugs', []),
            _rag_meta=rag_meta,
            _timing=timing,
            _extraction_mode=extraction_mode
        )
    
    def process_corpus(self, corpus_path: str, output_path: str):
        """Process entire corpus and save stay-level results"""
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = f.read()
        
        preprocessor = TextPreprocessor()
        patient_blocks = preprocessor.split_by_patients(corpus)
        
        all_stays = []
        stats = {
            'llm_success': 0,
            'regex_fallback': 0,
            'empty': 0,
            'llm_timeout': 0
        }
        
        total_stays = sum(len(preprocessor.split_by_stays(text)) for _, text in patient_blocks)
        current_stay = 0
        
        print(f"\nTotal patients: {len(patient_blocks)}")
        print(f"Total stays: {total_stays}")
        print("-" * 80)
        
        for patient_id, patient_text in patient_blocks:
            stays = preprocessor.split_by_stays(patient_text)
            
            for stay_id, stay_text in stays:
                current_stay += 1
                print(f"[{current_stay}/{total_stays}] Processing {patient_id}/{stay_id}...", end=' ')
                
                t_start = time.time()
                stay_facts = self.process_stay(stay_id, patient_id, stay_text)
                t_elapsed = time.time() - t_start
                
                # Track stats
                mode = stay_facts._extraction_mode
                if mode == "llm":
                    stats['llm_success'] += 1
                    print(f"✓ LLM ({t_elapsed:.1f}s)")
                elif mode == "regex_fallback":
                    stats['regex_fallback'] += 1
                    print(f"⚠ FALLBACK ({t_elapsed:.1f}s)")
                elif mode == "empty":
                    stats['empty'] += 1
                    print(f"∅ EMPTY ({t_elapsed:.1f}s)")
                
                all_stays.append(asdict(stay_facts))
        
        # Save to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for stay in all_stays:
                f.write(json.dumps(stay, ensure_ascii=False) + '\n')
        
        # Print summary
        print("\n" + "=" * 80)
        print("EXTRACTION SUMMARY")
        print("=" * 80)
        print(f"Total stays processed: {len(all_stays)}")
        print(f"  ✓ LLM success:       {stats['llm_success']} ({100*stats['llm_success']/len(all_stays):.1f}%)")
        print(f"  ⚠ Regex fallback:    {stats['regex_fallback']} ({100*stats['regex_fallback']/len(all_stays):.1f}%)")
        print(f"  ∅ Empty (no match):  {stats['empty']} ({100*stats['empty']/len(all_stays):.1f}%)")
        print("=" * 80)
        print(f"\n✓ Saved {len(all_stays)} stays to {output_path}")


# ============================================================================
# PATIENT-LEVEL AGGREGATION
# ============================================================================

class PatientAggregator:
    """Aggregate stay-level facts to patient-level"""
    
    @staticmethod
    def aggregate(stay_facts_path: str, output_path: str):
        """Aggregate stays by patient"""
        
        # Load all stays
        with open(stay_facts_path, 'r', encoding='utf-8') as f:
            stays = [json.loads(line) for line in f]
        
        # Group by patient
        patient_groups = defaultdict(list)
        for stay in stays:
            patient_id = stay['patient_id']
            patient_groups[patient_id].append(stay)
        
        # Aggregate each patient
        patient_facts = []
        
        for patient_id, patient_stays in patient_groups.items():
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
            
            # Deduplicate entities
            unique_disease = list({json.dumps(d, sort_keys=True): d for d in all_disease}.values())
            unique_labs = list({json.dumps(l, sort_keys=True): l for l in all_labs}.values())
            unique_drugs = list({json.dumps(d, sort_keys=True): d for d in all_drugs}.values())
            
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
        with open(output_path, 'w', encoding='utf-8') as f:
            for patient in patient_facts:
                f.write(json.dumps(patient, ensure_ascii=False) + '\n')
        
        print(f"✓ Aggregated {len(patient_facts)} patients to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    
    # Configuration
    KB_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\kb_pr_phenotype.json"
    CORPUS_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\phantom_ehr_corpus.txt"
    OUTPUT_STAY_PATH = "facts_agent1_stay.jsonl"
    OUTPUT_PATIENT_PATH = "facts_agent1_patient.jsonl"
    
    OLLAMA_URL = "http://localhost:11434"
    MODEL = "qwen2.5:3b-instruct"
    
    # Check if KB exists
    if not Path(KB_PATH).exists():
        print(f"ERROR: KB file not found: {KB_PATH}")
        print("Please provide the KB JSON file.")
        return
    
    # Initialize pipeline
    print("=" * 80)
    print("AGENT 1: KB-Guided Extraction + RAG Grounding")
    print("=" * 80)
    
    agent1 = Agent1Pipeline(
        kb_path=KB_PATH,
        ollama_url=OLLAMA_URL,
        model=MODEL,
        prefilter_window=2,
        chunk_size=6,
        chunk_stride=3,
        bm25_top_k=10,
        kb_slots_top_k=5,
        max_context_lines=45,
        llm_timeout=120,  # 2 minutes timeout
        llm_max_retries=2  # Retry twice on failure
    )
    
    # Process corpus
    print(f"\n[1/2] Processing corpus: {CORPUS_PATH}")
    agent1.process_corpus(CORPUS_PATH, OUTPUT_STAY_PATH)
    
    # Aggregate to patient level
    print(f"\n[2/2] Aggregating to patient-level...")
    PatientAggregator.aggregate(OUTPUT_STAY_PATH, OUTPUT_PATIENT_PATH)
    
    print("\n" + "=" * 80)
    print("✓ Agent 1 pipeline complete!")
    print(f"  - Stay-level facts: {OUTPUT_STAY_PATH}")
    print(f"  - Patient-level facts: {OUTPUT_PATIENT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()