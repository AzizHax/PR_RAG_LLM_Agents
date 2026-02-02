#!/usr/bin/env python3
"""
EDA pour Dossiers Patients Informatisés (DPI) en format .txt

Analyse exploratoire pour préparer l'ingestion de données réelles:
- Structure documentaire (patients, séjours, sections)
- Vocabulaire médical (terminologie, abréviations)
- Patterns temporels (dates, durées, chronologie)
- Complétude des données (champs manquants, taux de remplissage)
- Distribution des entités cliniques (diagnostics, labs, médicaments)
- Qualité du texte (longueur, densité, bruit)
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("⚠️  matplotlib/seaborn non installés - pas de visualisations")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Patterns médicaux RA-spécifiques
RA_TERMINOLOGY = {
    'disease_names': [
        r'\bpolyarthrite\s+rhumatoïde\b',
        r'\barthrite\s+rhumatoïde\b',
        r'\brheumatoid\s+arthritis\b',
        r'\bPR\b(?!\s*(artérielle|interval))',
        r'\bRA\b(?!\s*artérielle)',
    ],
    'serology': [
        r'\bRF\b',
        r'\bfacteur\s+rhumatoïde\b',
        r'\banti[- ]?CCP\b',
        r'\bACPA\b',
        r'\banti[- ]?citrullin',
    ],
    'inflammatory_markers': [
        r'\bCRP\b',
        r'\bprotéine\s+C\s+réactive\b',
        r'\bESR\b',
        r'\bVS\b',
        r'\bvitesse.*sédimentation\b',
    ],
    'dmards': [
        r'\bmethotrexate\b',
        r'\bméthotrexate\b',
        r'\bMTX\b',
        r'\bsulfasalazine\b',
        r'\bleflunomide\b',
        r'\bhydroxychloroquine\b',
    ],
    'biologics': [
        r'\badalimumab\b',
        r'\betanercept\b',
        r'\binfliximab\b',
        r'\btocilizumab\b',
        r'\brituximab\b',
        r'\babatacept\b',
    ],
    'jak_inhibitors': [
        r'\btofacitinib\b',
        r'\bbaricitinib\b',
        r'\bupadacitinib\b',
    ],
    'joints': [
        r'\bMCP\b',
        r'\bPIP\b',
        r'\bMTP\b',
        r'\bpoignet\b',
        r'\bwrist\b',
        r'\bgenou\b',
        r'\bknee\b',
        r'\bépaule\b',
        r'\bshoulder\b',
    ],
}

# Sections typiques d'un DPI
SECTION_MARKERS = {
    'history': [
        r'ANTÉCÉDENTS',
        r'HISTOIRE\s+DE\s+LA\s+MALADIE',
        r'ANAMNÈSE',
        r'HISTORY',
    ],
    'exam': [
        r'EXAMEN\s+CLINIQUE',
        r'PHYSICAL\s+EXAMINATION',
        r'EXAMEN\s+PHYSIQUE',
    ],
    'labs': [
        r'LABORATOIRE',
        r'BIOLOGIE',
        r'RÉSULTATS\s+LAB',
        r'LABORATORY',
    ],
    'imaging': [
        r'IMAGERIE',
        r'RADIOLOGIE',
        r'IMAGING',
        r'ECHOGRAPHIE',
        r'IRM',
        r'SCANNER',
    ],
    'treatment': [
        r'TRAITEMENT',
        r'THÉRAPEUTIQUE',
        r'MEDICATION',
        r'PRESCRIPTIONS',
    ],
    'conclusion': [
        r'CONCLUSION',
        r'DIAGNOSTIC',
        r'PLAN\s+DE\s+SOINS',
        r'ASSESSMENT',
    ],
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DocumentStats:
    """Statistiques par document (patient/séjour)"""
    doc_id: str
    n_lines: int
    n_chars: int
    n_words: int
    n_sentences: int
    avg_line_length: float
    avg_word_length: float
    has_dates: bool
    n_dates: int
    n_numeric_values: int
    sections_detected: List[str]
    medical_terms_count: Dict[str, int]


@dataclass
class CorpusStats:
    """Statistiques globales du corpus"""
    n_documents: int
    n_patients: int
    n_stays_per_patient: Dict[str, int]
    total_lines: int
    total_chars: int
    total_words: int
    avg_doc_length_lines: float
    avg_doc_length_chars: float
    vocabulary_size: int
    medical_term_frequencies: Dict[str, int]
    section_coverage: Dict[str, float]
    date_patterns: List[str]
    completeness_score: float


# ============================================================================
# DOCUMENT ANALYZER
# ============================================================================

class DocumentAnalyzer:
    """Analyse d'un document individuel"""
    
    @staticmethod
    def count_sentences(text: str) -> int:
        """Compte approximatif des phrases"""
        return len(re.findall(r'[.!?]+', text))
    
    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """Extrait les dates (formats variés)"""
        patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY ou DD-MM-YY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b',
        ]
        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        return dates
    
    @staticmethod
    def extract_numeric_values(text: str) -> List[str]:
        """Extrait valeurs numériques (résultats de labs)"""
        return re.findall(r'\b\d+[.,]?\d*\s*(mg|g|UI|U|mmol|µg|%)', text, re.IGNORECASE)
    
    @staticmethod
    def detect_sections(text: str) -> List[str]:
        """Détecte sections présentes"""
        detected = []
        for section_name, patterns in SECTION_MARKERS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected.append(section_name)
                    break
        return detected
    
    @staticmethod
    def count_medical_terms(text: str) -> Dict[str, int]:
        """Compte occurrences de terminologie médicale"""
        counts = {}
        for category, patterns in RA_TERMINOLOGY.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text, re.IGNORECASE))
            counts[category] = count
        return counts
    
    @classmethod
    def analyze_document(cls, doc_id: str, text: str) -> DocumentStats:
        """Analyse complète d'un document"""
        lines = text.split('\n')
        words = text.split()
        
        dates = cls.extract_dates(text)
        numeric_vals = cls.extract_numeric_values(text)
        sections = cls.detect_sections(text)
        medical_terms = cls.count_medical_terms(text)
        
        return DocumentStats(
            doc_id=doc_id,
            n_lines=len(lines),
            n_chars=len(text),
            n_words=len(words),
            n_sentences=cls.count_sentences(text),
            avg_line_length=len(text) / len(lines) if lines else 0,
            avg_word_length=sum(len(w) for w in words) / len(words) if words else 0,
            has_dates=len(dates) > 0,
            n_dates=len(dates),
            n_numeric_values=len(numeric_vals),
            sections_detected=sections,
            medical_terms_count=medical_terms,
        )


# ============================================================================
# CORPUS ANALYZER
# ============================================================================

class CorpusAnalyzer:
    """Analyse globale du corpus"""
    
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.documents: List[DocumentStats] = []
        self.corpus_stats: CorpusStats = None
        
    def parse_corpus(self) -> List[Tuple[str, str, str]]:
        """
        Parse corpus en (patient_id, stay_id, texte)
        Format attendu:
        PATIENT_ID: 0001
        === STAY_ID: S001 ===
        [texte du séjour]
        === STAY_ID: S002 ===
        [texte du séjour]
        PATIENT_ID: 0002
        ...
        """
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            corpus = f.read()
        
        documents = []
        current_patient = None
        current_stay = None
        current_text = []
        
        for line in corpus.split('\n'):
            if line.startswith('PATIENT_ID:'):
                # Sauver le séjour précédent
                if current_patient and current_stay and current_text:
                    documents.append((current_patient, current_stay, '\n'.join(current_text)))
                current_patient = line.split(':', 1)[1].strip()
                current_stay = None
                current_text = []
                
            elif line.startswith('===') and 'STAY_ID:' in line:
                # Sauver le séjour précédent
                if current_patient and current_stay and current_text:
                    documents.append((current_patient, current_stay, '\n'.join(current_text)))
                match = re.search(r'STAY_ID:\s*(\S+)', line)
                current_stay = match.group(1) if match else "unknown"
                current_text = []
                
            else:
                current_text.append(line)
        
        # Dernier document
        if current_patient and current_stay and current_text:
            documents.append((current_patient, current_stay, '\n'.join(current_text)))
        
        return documents
    
    def analyze_corpus(self):
        """Analyse complète du corpus"""
        print("📊 Analyse du corpus DPI...")
        
        # Parse documents
        documents = self.parse_corpus()
        print(f"✓ {len(documents)} documents (séjours) trouvés")
        
        # Analyse document par document
        for patient_id, stay_id, text in documents:
            doc_id = f"{patient_id}_{stay_id}"
            doc_stats = DocumentAnalyzer.analyze_document(doc_id, text)
            self.documents.append(doc_stats)
        
        # Agrégation corpus-level
        self._compute_corpus_stats(documents)
        
        print(f"✓ Analyse complète de {len(self.documents)} documents")
    
    def _compute_corpus_stats(self, documents: List[Tuple[str, str, str]]):
        """Calcule statistiques agrégées"""
        
        # Compter patients uniques
        patients = set(patient_id for patient_id, _, _ in documents)
        n_patients = len(patients)
        
        # Séjours par patient
        stays_per_patient = defaultdict(int)
        for patient_id, _, _ in documents:
            stays_per_patient[patient_id] += 1
        
        # Métriques textuelles
        total_lines = sum(d.n_lines for d in self.documents)
        total_chars = sum(d.n_chars for d in self.documents)
        total_words = sum(d.n_words for d in self.documents)
        
        # Vocabulaire (approximatif)
        all_words = set()
        for _, _, text in documents:
            all_words.update(text.lower().split())
        
        # Fréquences termes médicaux (agrégées)
        medical_freqs = defaultdict(int)
        for doc in self.documents:
            for category, count in doc.medical_terms_count.items():
                medical_freqs[category] += count
        
        # Couverture des sections (% documents avec section)
        section_coverage = defaultdict(int)
        for doc in self.documents:
            for section in doc.sections_detected:
                section_coverage[section] += 1
        section_coverage = {k: v / len(self.documents) for k, v in section_coverage.items()}
        
        # Patterns de dates uniques
        all_dates = []
        for _, _, text in documents:
            all_dates.extend(DocumentAnalyzer.extract_dates(text))
        date_patterns = list(set(all_dates[:50]))  # Échantillon
        
        # Score de complétude (heuristique basé sur sections + dates + termes médicaux)
        completeness = 0.0
        for doc in self.documents:
            doc_score = 0.0
            doc_score += min(len(doc.sections_detected) / 4.0, 1.0) * 0.4  # Au moins 4 sections
            doc_score += (1.0 if doc.has_dates else 0.0) * 0.2
            doc_score += min(sum(doc.medical_terms_count.values()) / 10.0, 1.0) * 0.4
            completeness += doc_score
        completeness /= len(self.documents)
        
        self.corpus_stats = CorpusStats(
            n_documents=len(self.documents),
            n_patients=n_patients,
            n_stays_per_patient=dict(stays_per_patient),
            total_lines=total_lines,
            total_chars=total_chars,
            total_words=total_words,
            avg_doc_length_lines=total_lines / len(self.documents),
            avg_doc_length_chars=total_chars / len(self.documents),
            vocabulary_size=len(all_words),
            medical_term_frequencies=dict(medical_freqs),
            section_coverage=section_coverage,
            date_patterns=date_patterns,
            completeness_score=completeness,
        )
    
    def print_summary(self):
        """Affiche résumé textuel"""
        stats = self.corpus_stats
        
        print("\n" + "=" * 80)
        print("RÉSUMÉ EDA - DOSSIERS PATIENTS INFORMATISÉS (DPI)")
        print("=" * 80)
        
        print(f"\n📁 STRUCTURE DU CORPUS")
        print(f"  Nombre total de documents (séjours): {stats.n_documents}")
        print(f"  Nombre de patients uniques: {stats.n_patients}")
        print(f"  Séjours par patient (moyenne): {stats.n_documents / stats.n_patients:.1f}")
        print(f"  Séjours par patient (distribution):")
        stay_dist = Counter(stats.n_stays_per_patient.values())
        for n_stays, count in sorted(stay_dist.items()):
            print(f"    - {n_stays} séjour(s): {count} patients ({count/stats.n_patients*100:.1f}%)")
        
        print(f"\n📝 CONTENU TEXTUEL")
        print(f"  Lignes totales: {stats.total_lines:,}")
        print(f"  Caractères totaux: {stats.total_chars:,}")
        print(f"  Mots totaux: {stats.total_words:,}")
        print(f"  Longueur moyenne par document: {stats.avg_doc_length_lines:.0f} lignes, {stats.avg_doc_length_chars:.0f} caractères")
        print(f"  Taille du vocabulaire: {stats.vocabulary_size:,} mots uniques")
        
        print(f"\n🏥 TERMINOLOGIE MÉDICALE (RA-SPÉCIFIQUE)")
        for category, count in sorted(stats.medical_term_frequencies.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {category}: {count} occurrences")
        
        print(f"\n📋 COUVERTURE DES SECTIONS CLINIQUES")
        for section, coverage in sorted(stats.section_coverage.items(), key=lambda x: x[1], reverse=True):
            print(f"  {section}: {coverage*100:.1f}% des documents")
        
        print(f"\n📅 PATTERNS TEMPORELS")
        print(f"  Documents avec dates: {sum(1 for d in self.documents if d.has_dates)} / {stats.n_documents} ({sum(1 for d in self.documents if d.has_dates)/stats.n_documents*100:.1f}%)")
        if stats.date_patterns:
            print(f"  Exemples de formats de dates détectés:")
            for date in stats.date_patterns[:10]:
                print(f"    - {date}")
        
        print(f"\n✅ COMPLÉTUDE DES DONNÉES")
        print(f"  Score global de complétude: {stats.completeness_score*100:.1f}%")
        print(f"    (Basé sur: présence de sections, dates, terminologie médicale)")
        
        # Alertes qualité
        print(f"\n⚠️  ALERTES QUALITÉ")
        alerts = []
        if stats.completeness_score < 0.5:
            alerts.append("- Complétude faible (<50%) - données potentiellement incomplètes")
        if stats.section_coverage.get('labs', 0) < 0.3:
            alerts.append("- Section laboratoire rare (<30%) - données biologiques limitées")
        if sum(stats.medical_term_frequencies.values()) / stats.n_documents < 5:
            alerts.append("- Peu de terminologie médicale (< 5 termes/doc) - vérifier pertinence")
        
        if alerts:
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("  Aucune alerte majeure détectée")
        
        print("\n" + "=" * 80)
    
    def save_detailed_report(self, output_path: str):
        """Sauvegarde rapport détaillé en JSON"""
        report = {
            'corpus_stats': asdict(self.corpus_stats),
            'document_stats': [asdict(d) for d in self.documents],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Rapport détaillé sauvegardé: {output_path}")
    
    def plot_distributions(self, output_dir: str = "."):
        """Génère visualisations (si matplotlib disponible)"""
        if not HAS_VIZ:
            print("⚠️  Visualisations non disponibles (installer matplotlib + seaborn)")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        sns.set_style("whitegrid")
        
        # 1. Distribution longueur documents (lignes)
        fig, ax = plt.subplots(figsize=(10, 6))
        lengths = [d.n_lines for d in self.documents]
        ax.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Nombre de lignes")
        ax.set_ylabel("Fréquence")
        ax.set_title("Distribution de la longueur des documents (lignes)")
        plt.tight_layout()
        plt.savefig(output_dir / "doc_length_distribution.png", dpi=150)
        plt.close()
        
        # 2. Distribution longueur documents (caractères)
        fig, ax = plt.subplots(figsize=(10, 6))
        lengths = [d.n_chars for d in self.documents]
        ax.hist(lengths, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel("Nombre de caractères")
        ax.set_ylabel("Fréquence")
        ax.set_title("Distribution de la longueur des documents (caractères)")
        plt.tight_layout()
        plt.savefig(output_dir / "doc_chars_distribution.png", dpi=150)
        plt.close()
        
        # 3. Barplot terminologie médicale
        fig, ax = plt.subplots(figsize=(12, 6))
        categories = list(self.corpus_stats.medical_term_frequencies.keys())
        counts = list(self.corpus_stats.medical_term_frequencies.values())
        ax.barh(categories, counts, color='steelblue')
        ax.set_xlabel("Nombre d'occurrences")
        ax.set_title("Fréquence de la terminologie médicale (RA)")
        plt.tight_layout()
        plt.savefig(output_dir / "medical_terminology.png", dpi=150)
        plt.close()
        
        # 4. Heatmap sections détectées
        fig, ax = plt.subplots(figsize=(10, 8))
        section_matrix = []
        section_names = list(SECTION_MARKERS.keys())
        for doc in self.documents:
            row = [1 if s in doc.sections_detected else 0 for s in section_names]
            section_matrix.append(row)
        
        sns.heatmap(section_matrix, cmap='YlGnBu', cbar_kws={'label': 'Présence'},
                    xticklabels=section_names, yticklabels=False, ax=ax)
        ax.set_xlabel("Section")
        ax.set_ylabel("Documents")
        ax.set_title("Matrice de présence des sections cliniques")
        plt.tight_layout()
        plt.savefig(output_dir / "section_coverage_heatmap.png", dpi=150)
        plt.close()
        
        print(f"\n📊 Visualisations sauvegardées dans: {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Exécution principale"""
    
    # Configuration
    CORPUS_PATH = r"C:\\Users\\Mega-Pc\\Desktop\\Projects\\PhenoRag\\fantôme\\Data\\phantom_ehr_corpus.txt"  # ⚠️ À adapter selon votre fichier
    OUTPUT_REPORT = "eda_report.json"
    OUTPUT_VIZ_DIR = "eda_visualizations"
    
    # Vérifier existence du corpus
    if not Path(CORPUS_PATH).exists():
        print(f"❌ ERREUR: Fichier corpus non trouvé: {CORPUS_PATH}")
        print("   Veuillez spécifier le chemin vers votre fichier DPI .txt")
        return
    
    # Analyse
    analyzer = CorpusAnalyzer(CORPUS_PATH)
    analyzer.analyze_corpus()
    
    # Résumé textuel
    analyzer.print_summary()
    
    # Rapport JSON détaillé
    analyzer.save_detailed_report(OUTPUT_REPORT)
    
    # Visualisations (optionnel)
    if HAS_VIZ:
        analyzer.plot_distributions(OUTPUT_VIZ_DIR)
    
    print("\n✅ EDA terminée avec succès!")
    print(f"   - Résumé: affiché ci-dessus")
    print(f"   - Rapport détaillé: {OUTPUT_REPORT}")
    if HAS_VIZ:
        print(f"   - Visualisations: {OUTPUT_VIZ_DIR}/")


if __name__ == "__main__":
    main()