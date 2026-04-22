#!/usr/bin/env python3
"""
ats_scorer.py — ATS (Applicant Tracking System) Resume Scorer

Takes a resume PDF/DOCX and a job description, then prints a detailed
ATS compatibility score based on keyword matching, skills alignment,
experience, education, and semantic similarity.

Usage:
    # Interactive mode
    python ats_scorer.py

    # CLI mode
    python ats_scorer.py --resume resume.pdf --job "job description text"
    python ats_scorer.py --resume resume.pdf --job-file job.txt
"""

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================

import argparse
import datetime
import json
import os
import re
import sys
import string

# ============================================================
# SECTION 2 — TEXT EXTRACTOR
# ============================================================

def extract_text(file_path: str) -> str:
    """
    Extract text from a PDF or DOCX file.

    - If .pdf  → use pdfplumber, concatenate all pages
    - If .docx → use python-docx, concatenate all paragraphs
    - Strip extra whitespace, keep newlines for section detection
    - If extracted text < 50 chars, print warning about scanned images
    """
    if not os.path.isfile(file_path):
        print(f"Error: File not found — '{file_path}'")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            print("Error: pdfplumber is required. Install with: pip install pdfplumber")
            sys.exit(1)

        text_parts = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            sys.exit(1)

        text = "\n".join(text_parts)

    elif ext == ".docx":
        try:
            import docx
        except ImportError:
            print("Error: python-docx is required. Install with: pip install python-docx")
            sys.exit(1)

        try:
            doc = docx.Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            sys.exit(1)

    else:
        print(f"Error: Unsupported file format '{ext}'. Only .pdf and .docx are supported.")
        sys.exit(1)

    # Strip excessive whitespace but preserve newlines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned = " ".join(line.split())
        cleaned_lines.append(cleaned)
    text = "\n".join(cleaned_lines)

    if len(text.strip()) < 50:
        print("Warning: very little text extracted — file may be a scanned image")

    return text


# ============================================================
# SECTION 3 — SECTION SPLITTER
# ============================================================

SECTION_PATTERNS = {
    "experience": re.compile(r"\b(experience|work\s+history|employment)\b", re.IGNORECASE),
    "education":  re.compile(r"\b(education|academic|qualification|degree)\b", re.IGNORECASE),
    "skills":     re.compile(r"\b(skills|technical|competencies|expertise)\b", re.IGNORECASE),
    "summary":    re.compile(r"\b(summary|objective|profile|about)\b", re.IGNORECASE),
}


def split_sections(text: str) -> dict:
    """
    Returns dict with keys: 'summary', 'experience', 'education', 'skills', 'full'

    Walks lines top to bottom; when a line matches a header pattern, switches
    the current section. 'full' always equals the entire text.
    If no summary section found, uses the first 3 non-empty lines as summary.
    """
    sections = {
        "summary": [],
        "experience": [],
        "education": [],
        "skills": [],
    }

    current_section = None

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current_section:
                sections[current_section].append("")
            continue

        # Check if this line is a section header
        matched_section = None
        for sec_name, pattern in SECTION_PATTERNS.items():
            if pattern.search(stripped):
                # Heuristic: header lines are typically short (< 60 chars)
                if len(stripped) < 60:
                    matched_section = sec_name
                    break

        if matched_section:
            current_section = matched_section
            continue  # Don't include the header line itself

        if current_section:
            sections[current_section].append(stripped)

    # Build result dict with joined text
    result = {}
    for key in sections:
        result[key] = "\n".join(sections[key]).strip()

    # If no summary found, use first 3 non-empty lines
    if not result["summary"]:
        non_empty = [l.strip() for l in text.split("\n") if l.strip()]
        result["summary"] = "\n".join(non_empty[:3])

    result["full"] = text
    return result


# ============================================================
# SECTION 4 — FIVE SCORER FUNCTIONS
# ============================================================

# ------------------------------------------------------------------
# 4a — Keyword scorer  (weight: 0.30)
# ------------------------------------------------------------------

def score_keywords(resume_text: str, job_description: str) -> float:
    """
    Returns float 0.0–1.0

    Combines TF-IDF cosine similarity (50%) with keyword hit rate (50%).
    Uses rapidfuzz for fuzzy matching on individual keywords.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from rapidfuzz import fuzz

    # Step 1 — TF-IDF cosine similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform([job_description, resume_text])
        tfidf_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except ValueError:
        # Edge case: empty vocabulary
        tfidf_score = 0.0

    # Step 2 — Keyword hit rate
    # Tokenize job description into words
    translator = str.maketrans("", "", string.punctuation)
    job_words = job_description.lower().translate(translator).split()
    keywords = list(set(w for w in job_words if len(w) >= 4))

    if not keywords:
        hit_rate = 0.0
    else:
        resume_lower = resume_text.lower()
        matched = 0
        for kw in keywords:
            if kw in resume_lower:
                matched += 1
            elif fuzz.partial_ratio(kw, resume_lower) >= 85:
                matched += 1
        hit_rate = matched / len(keywords)

    return round((tfidf_score * 0.5) + (hit_rate * 0.5), 4)


# ------------------------------------------------------------------
# 4b — Skills scorer  (weight: 0.35)
# ------------------------------------------------------------------

SKILLS_LIST = [
    # programming languages
    "python", "javascript", "typescript", "java", "go", "rust", "c++", "c#",
    "ruby", "php", "swift", "kotlin", "r", "scala", "matlab",
    # web frameworks
    "fastapi", "django", "flask", "react", "vue", "angular", "nextjs",
    "express", "spring", "rails", "laravel", "nodejs",
    # databases
    "postgresql", "mysql", "mongodb", "redis", "sqlite", "cassandra",
    "elasticsearch", "dynamodb", "oracle",
    # cloud / devops
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "jenkins",
    "github actions", "ci/cd", "ansible", "linux",
    # data / ml
    "machine learning", "deep learning", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "spark", "hadoop", "nlp",
    "data engineering", "etl", "tableau", "power bi",
    # soft / general
    "leadership", "agile", "scrum", "communication", "project management",
    # marketing
    "seo", "sem", "google ads", "content marketing", "social media",
    "email marketing", "crm", "salesforce", "hubspot", "google analytics",
    # finance
    "financial modeling", "excel", "accounting", "budgeting", "forecasting",
]

SKILL_ALIASES = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "k8s": "kubernetes", "ml": "machine learning", "dl": "deep learning",
    "sklearn": "scikit-learn", "np": "numpy", "postgres": "postgresql",
    "mongo": "mongodb", "node": "nodejs",
}


def _expand_aliases(text: str) -> str:
    """Replace alias keys with their canonical skill names in text."""
    expanded = text.lower()
    for alias, canonical in SKILL_ALIASES.items():
        # Use word boundary matching to avoid partial replacements
        expanded = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, expanded)
    return expanded


def _extract_skills_from_text(text: str) -> set:
    """Extract all known skills found in text."""
    expanded = _expand_aliases(text)
    found = set()
    for skill in SKILLS_LIST:
        if skill in expanded:
            found.add(skill)
    return found


def score_skills(resume_text: str, job_description: str) -> float:
    """
    Returns float 0.0–1.0

    Extracts required skills from job description, then checks which are
    present in the resume text (with fuzzy matching via rapidfuzz).
    """
    from rapidfuzz import fuzz

    # Step 1 — Extract job required skills
    required_skills = _extract_skills_from_text(job_description)
    if not required_skills:
        return 0.5  # Neutral — can't evaluate without required skills

    # Step 2 — Extract resume skills (exact matching)
    found_skills = _extract_skills_from_text(resume_text)

    # Step 3 — Fuzzy matching for remaining required skills
    resume_lower = _expand_aliases(resume_text)
    # Build 2-gram sliding window of words for phrase matching
    resume_words = resume_lower.split()
    bigrams = []
    for i in range(len(resume_words)):
        bigrams.append(resume_words[i])
        if i + 1 < len(resume_words):
            bigrams.append(resume_words[i] + " " + resume_words[i + 1])

    fuzzy_found = set()
    for skill in required_skills:
        if skill in found_skills:
            fuzzy_found.add(skill)
            continue
        # Try fuzzy match against bigrams
        for bg in bigrams:
            if fuzz.token_sort_ratio(skill, bg) >= 80:
                fuzzy_found.add(skill)
                break

    # Step 4 — Score
    matched = len(fuzzy_found)
    return round(min(1.0, matched / len(required_skills)), 4)


# ------------------------------------------------------------------
# 4c — Experience scorer  (weight: 0.20)
# ------------------------------------------------------------------

def score_experience(resume_text: str, job_description: str) -> tuple:
    """
    Returns (score_float_0_to_1, total_years_float)

    Extracts required years from job description and date ranges from resume,
    then scores the candidate's experience against the requirement.
    """
    from dateutil import parser as date_parser

    # Step 1 — Extract required years from job description
    req_match = re.search(
        r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)?",
        job_description,
        re.IGNORECASE,
    )
    min_years_required = int(req_match.group(1)) if req_match else 0

    # Step 2 — Extract date ranges from resume
    # Pattern: "Month Year – Month Year" or "Year – Year" or "Year – present"
    date_range_patterns = [
        # "Jan 2019 – Mar 2022" / "January 2019 - March 2022"
        re.compile(
            r"([A-Za-z]+\.?\s+\d{4})\s*[-–—to]+\s*([A-Za-z]+\.?\s+\d{4}|present|current|now)",
            re.IGNORECASE,
        ),
        # "2019 – 2022" / "2019 - present"
        re.compile(
            r"(\d{4})\s*[-–—to]+\s*(\d{4}|present|current|now)",
            re.IGNORECASE,
        ),
    ]

    today = datetime.date.today()
    total_months = 0.0

    for pattern in date_range_patterns:
        for match in pattern.finditer(resume_text):
            start_str = match.group(1).strip()
            end_str = match.group(2).strip()

            try:
                start_date = date_parser.parse(start_str, fuzzy=True).date()
            except (ValueError, TypeError):
                continue

            if end_str.lower() in ("present", "current", "now"):
                end_date = today
            else:
                try:
                    end_date = date_parser.parse(end_str, fuzzy=True).date()
                except (ValueError, TypeError):
                    continue

            if end_date < start_date:
                continue

            months = (end_date - start_date).days / 30.44
            total_months += months

    total_years = total_months / 12.0

    # Step 3 — Score
    if min_years_required == 0:
        return (1.0, round(total_years, 2))

    ratio = total_years / min_years_required
    if ratio >= 1.0:
        score = 1.0
    elif ratio >= 0.6:
        score = 0.75
    elif ratio >= 0.3:
        score = 0.4
    else:
        score = max(0.1, ratio)

    return (round(score, 4), round(total_years, 2))


# ------------------------------------------------------------------
# 4d — Education scorer  (weight: 0.10)
# ------------------------------------------------------------------

DEGREE_HIERARCHY = {
    "phd": 6, "ph.d": 6, "doctorate": 6, "doctoral": 6,
    "master": 5, "m.s.": 5, "m.tech": 5, "mba": 5, "m.e.": 5, "m.sc": 5,
    "bachelor": 4, "b.s.": 4, "b.tech": 4, "b.e.": 4, "b.com": 4,
    "b.sc": 4, "undergraduate": 4,
    "associate": 3, "diploma": 2,
    "high school": 1, "12th": 1, "hsc": 1, "secondary": 1,
}


def score_education(resume_text: str, job_description: str) -> tuple:
    """
    Returns (score_float_0_to_1, detected_degree_string)

    Compares the highest degree found in the resume against the requirement
    in the job description.
    """
    def _detect_highest_degree(text: str) -> tuple:
        """Returns (level_int, degree_string) for the highest degree found."""
        text_lower = text.lower()
        best_level = 0
        best_degree = "none"
        for degree_key, level in DEGREE_HIERARCHY.items():
            if degree_key in text_lower:
                if level > best_level:
                    best_level = level
                    best_degree = degree_key
        return best_level, best_degree

    # Step 1 — Detect required level from job description
    req_level, _ = _detect_highest_degree(job_description)
    if req_level == 0:
        req_level = 4  # Default to bachelor if none specified

    # Step 2 — Detect candidate level from resume
    cand_level, detected_degree = _detect_highest_degree(resume_text)

    # Step 3 — Score
    diff = req_level - cand_level
    if diff <= 0:
        return (1.0, detected_degree)
    elif diff == 1:
        return (0.7, detected_degree)
    elif diff == 2:
        return (0.4, detected_degree)
    else:
        return (0.1, detected_degree)


# ------------------------------------------------------------------
# 4e — Semantic scorer  (weight: 0.05)
# ------------------------------------------------------------------

_semantic_model = None  # Module-level cache


def score_semantic(resume_summary: str, job_description: str) -> float:
    """
    Returns float 0.0–1.0

    Uses sentence-transformers 'all-MiniLM-L6-v2' to compute semantic
    similarity between the resume summary and the job description.
    Falls back to 0.5 if sentence-transformers is not installed.
    """
    global _semantic_model

    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print("Warning: sentence-transformers not installed — semantic score defaults to 0.5")
        return 0.5

    try:
        if _semantic_model is None:
            _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = _semantic_model.encode(
            [resume_summary, job_description], convert_to_tensor=True
        )
        sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        return round(max(0.0, min(1.0, sim)), 4)
    except Exception as e:
        print(f"Warning: Semantic scoring failed ({e}) — defaulting to 0.5")
        return 0.5


# ============================================================
# SECTION 5 — WEIGHTED AGGREGATOR
# ============================================================

WEIGHTS = {
    "keyword":    0.30,
    "skills":     0.35,
    "experience": 0.20,
    "education":  0.10,
    "semantic":   0.05,
}


def compute_ats_score(
    keyword_score: float,
    skills_score: float,
    exp_score: float,
    edu_score: float,
    semantic_score: float,
    weights: dict = None,
) -> tuple:
    """
    Computes the final weighted ATS score and detailed breakdown.

    Returns (final_score rounded to 2 decimals, breakdown_dict)
    """
    if weights is None:
        weights = WEIGHTS

    raw_scores = {
        "keyword":    keyword_score,
        "skills":     skills_score,
        "experience": exp_score,
        "education":  edu_score,
        "semantic":   semantic_score,
    }

    breakdown = {}
    final_score = 0.0

    for category, raw in raw_scores.items():
        w = weights[category]
        points = round(raw * w * 100, 2)
        breakdown[category] = {
            "raw": round(raw, 4),
            "weight": w,
            "points": points,
        }
        final_score += raw * w * 100

    return (round(final_score, 2), breakdown)


# ============================================================
# SECTION 6 — MAIN RUNNER
# ============================================================

def _print_report(
    resume_path: str,
    job_desc: str,
    breakdown: dict,
    final_score: float,
    total_years: float,
    detected_degree: str,
):
    """Print the formatted ATS score report."""
    job_preview = job_desc[:60].replace("\n", " ")
    if len(job_desc) > 60:
        job_preview += "..."

    category_labels = {
        "keyword":    "Keyword match",
        "skills":     "Skills match",
        "experience": "Experience",
        "education":  "Education",
        "semantic":   "Semantic sim.",
    }

    print()
    print("=" * 60)
    print(" ATS SCORE REPORT")
    print("=" * 60)
    print(f" Resume  : {os.path.basename(resume_path)}")
    print(f" Job role: {job_preview}")
    print("-" * 60)
    print(f" {'CATEGORY':<16s} {'RAW':>6s}  {'WEIGHT':>6s}  {'POINTS':>7s}")
    print("-" * 60)

    for cat in ["keyword", "skills", "experience", "education", "semantic"]:
        info = breakdown[cat]
        label = category_labels[cat]
        raw_str = f"{info['raw']:.2f}"
        weight_str = f"{int(info['weight'] * 100)}%"
        points_str = f"{info['points']:.2f}"
        print(f" {label:<16s} {raw_str:>6s}  {weight_str:>6s}  {points_str:>7s}")

    print("-" * 60)
    print(f" FINAL ATS SCORE :  {final_score:.2f} / 100")
    print("=" * 60)
    print(" Details:")
    print(f"   Experience detected : {total_years} years")
    print(f"   Education detected  : {detected_degree}")
    print(f"   Semantic model      : all-MiniLM-L6-v2")
    print("=" * 60)
    print()


def main():
    """
    Entry point: parse CLI args or run interactive mode, then execute
    the full ATS scoring pipeline and print the report.
    """
    parser = argparse.ArgumentParser(
        description="ATS Resume Scorer — score a resume against a job description."
    )
    parser.add_argument("--resume", type=str, help="Path to resume PDF or DOCX file")
    parser.add_argument("--job", type=str, help="Job description as a string")
    parser.add_argument("--job-file", type=str, help="Path to .txt file containing job description")
    parser.add_argument(
        "--weights",
        type=str,
        help='Optional JSON string to override weights, e.g. \'{"keyword":0.25,"skills":0.20,...}\'',
    )

    args = parser.parse_args()

    # ---- Determine resume path ----
    resume_path = args.resume
    if not resume_path:
        # Interactive mode
        resume_path = input("Enter path to resume PDF/DOCX: ").strip()
        if not resume_path:
            print("Error: No resume path provided.")
            sys.exit(1)

    # ---- Determine job description ----
    job_desc = None
    if args.job:
        job_desc = args.job
    elif args.job_file:
        if not os.path.isfile(args.job_file):
            print(f"Error: Job description file not found — '{args.job_file}'")
            sys.exit(1)
        with open(args.job_file, "r", encoding="utf-8") as f:
            job_desc = f.read()
    else:
        if not args.resume:
            # Full interactive mode — also ask for job description
            print("Paste job description (press Enter twice when done):")
            lines = []
            blank_count = 0
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip() == "":
                    blank_count += 1
                    if blank_count >= 2:
                        break
                    lines.append(line)
                else:
                    blank_count = 0
                    lines.append(line)
            job_desc = "\n".join(lines).strip()
        else:
            print("Error: No job description provided. Use --job or --job-file.")
            sys.exit(1)

    if not job_desc or len(job_desc.strip()) < 10:
        print("Error: Job description is too short or empty.")
        sys.exit(1)

    # ---- Override weights if provided ----
    weights = WEIGHTS.copy()
    if args.weights:
        try:
            custom = json.loads(args.weights)
            for key in custom:
                if key in weights:
                    weights[key] = float(custom[key])
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse custom weights ({e}), using defaults.")

    # ---- Run pipeline ----
    print("\n⏳ Extracting text from resume...")
    text = extract_text(resume_path)

    print("📄 Splitting resume sections...")
    sections = split_sections(text)

    print("🔑 Scoring keyword match...")
    kw = score_keywords(text, job_desc)

    print("🛠️  Scoring skills match...")
    sk = score_skills(text, job_desc)

    print("📅 Scoring experience...")
    exp_section = sections["experience"] if sections["experience"] else text
    exp_score, total_years = score_experience(exp_section, job_desc)

    print("🎓 Scoring education...")
    edu_score, detected_degree = score_education(text, job_desc)

    print("🧠 Scoring semantic similarity...")
    summary_text = sections["summary"] if sections["summary"] else text[:500]
    sem = score_semantic(summary_text, job_desc)

    print("📊 Computing final score...")
    final, breakdown = compute_ats_score(kw, sk, exp_score, edu_score, sem, weights)

    # ---- Print report ----
    _print_report(resume_path, job_desc, breakdown, final, total_years, detected_degree)

    # ---- Exit code ----
    sys.exit(0 if final >= 60 else 1)


if __name__ == "__main__":
    main()
