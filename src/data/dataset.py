from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

SEED = 42

# ---------------------------------------------------------------------------
# Vocabulary used during featurisation and dataset generation
# ---------------------------------------------------------------------------

TECH_KEYWORDS = [
    "python", "pytorch", "machine", "learning", "sql", "docker",
    "kubernetes", "javascript", "react", "nlp", "deep", "aws",
    "azure", "tensorflow", "pandas", "numpy", "fastapi", "flask",
    "mongodb", "redis", "git", "linux",
]

SKILL_PATTERNS = {
    "python": ("python",),
    "pytorch": ("pytorch",),
    "machine learning": ("machine learning", "ml"),
    "deep learning": ("deep learning",),
    "sql": ("sql",),
    "docker": ("docker",),
    "kubernetes": ("kubernetes", "k8s"),
    "javascript": ("javascript",),
    "react": ("react",),
    "nlp": ("nlp", "natural language processing"),
    "aws": ("aws", "amazon web services"),
    "azure": ("azure",),
    "tensorflow": ("tensorflow",),
    "pandas": ("pandas",),
    "numpy": ("numpy",),
    "fastapi": ("fastapi",),
    "flask": ("flask",),
    "mongodb": ("mongodb",),
    "redis": ("redis",),
    "git": ("git",),
    "linux": ("linux",),
}

COMMUNICATION_SIGNALS = ["team", "communication", "stakeholder", "collaboration", "client"]
PROJECT_SIGNALS = ["project", "delivery", "deployed", "built", "ownership", "shipped"]
CLOUD_SKILLS = {"aws", "azure", "docker", "kubernetes", "linux"}
TOKEN_RE = re.compile(r"[\w+#.]+", re.UNICODE)

SENIORITY_SIGNALS = ["lead", "senior", "principal", "manager", "head", "director"]
DEGREE_SIGNALS = ["bachelor", "master", "phd", "doctorate", "engineer", "licence", "ingénieur"]
EXPERIENCE_SIGNALS = ["years", "experience", "experience", "expérience"]

JOB_TEMPLATES = [
    {
        "job_id": "job_ml_001",
        "job_text": (
            "Machine learning engineer with strong python pytorch deep learning skills. "
            "Must have 3 years experience, sql docker aws and nlp experience. "
            "Project ownership and teamwork are important."
        ),
    },
    {
        "job_id": "job_data_001",
        "job_text": (
            "Data scientist proficient in python pandas numpy sql machine learning. "
            "At least 2 years experience with data analysis and visualization required."
        ),
    },
    {
        "job_id": "job_web_001",
        "job_text": (
            "Web developer with javascript react nodejs experience. "
            "Knowledge of mongodb redis and docker is a plus. Strong collaboration expected."
        ),
    },
    {
        "job_id": "job_devops_001",
        "job_text": (
            "DevOps engineer skilled in docker kubernetes linux aws azure. "
            "Requires 4 years experience. Familiarity with python and git automation required."
        ),
    },
]

# CV templates by fit level
_CV_HIGH = [
    "Senior python engineer with 5 years experience in pytorch deep learning and nlp. "
    "Led multiple projects on aws. Master degree in computer science.",
    "Principal data scientist with 6 years experience. Expert in python pytorch sql pandas. "
    "Managed a team of 4. Phd in machine learning.",
    "Senior ml engineer 4 years experience python pytorch aws docker kubernetes. "
    "Bachelor in software engineering. Extensive project delivery.",
    "Lead engineer with deep learning pytorch nlp python sql experience. "
    "5 years in the field. Managed ci/cd pipelines on azure.",
]

_CV_MEDIUM = [
    "Engineer with 2 years experience in python and some machine learning projects. "
    "Basic sql and git knowledge. Bachelor degree.",
    "Junior data analyst with python pandas experience. "
    "Completed a deep learning course. Familiar with docker.",
    "Software developer with javascript react nodejs skills. "
    "Some python scripting. 3 years experience.",
    "Ingénieur en informatique avec expérience en python et sql. "
    "Projets de machine learning en cours. Licence informatique.",
]

_CV_LOW = [
    "Recent graduate with basic python skills. No work experience yet.",
    "Student looking for internship. Knowledge of html css javascript.",
    "Accountant with excel and word skills. No programming experience.",
    "Marketing professional with social media skills. No technical background.",
    "Étudiant en première année. Cours de programmation basique uniquement.",
]


# ---------------------------------------------------------------------------
# Featurisation
# ---------------------------------------------------------------------------

def featurize(cv_text: str, job_text: str) -> list[float]:
    cv_lower = normalize_text(cv_text)
    job_lower = normalize_text(job_text)
    cv_tokens = set(tokenize(cv_lower))
    job_tokens = set(tokenize(job_lower))
    cv_skills = extract_skills(cv_lower)
    job_skills = extract_skills(job_lower)
    job_cloud = job_skills & CLOUD_SKILLS
    cv_years = extract_years(cv_lower)
    job_years = extract_years(job_lower)

    token_overlap = len(cv_tokens & job_tokens) / max(len(job_tokens), 1)
    skill_coverage = len(cv_skills & job_skills) / max(len(job_skills), 1)
    years_alignment = min(cv_years / max(job_years, 1), 1.0) if job_years else float(cv_years > 0)
    seniority_score = min(sum(1 for s in SENIORITY_SIGNALS if s in cv_lower) / 2.0, 1.0)
    degree_score = 1.0 if any(d in cv_lower for d in DEGREE_SIGNALS) else 0.0
    project_score = float(any(term in cv_lower for term in PROJECT_SIGNALS))
    cloud_alignment = len((cv_skills & job_skills) & CLOUD_SKILLS) / max(len(job_cloud), 1) if job_cloud else 0.0
    communication_score = float(any(term in cv_lower for term in COMMUNICATION_SIGNALS))

    return [
        round(skill_coverage, 4),
        round(token_overlap, 4),
        round(years_alignment, 4),
        round(seniority_score, 4),
        round(degree_score, 4),
        round(project_score, 4),
        round(cloud_alignment, 4),
        round(communication_score, 4),
    ]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def extract_years(text: str) -> int:
    matches = [int(value) for value in re.findall(r"(\d+)\+?\s*(?:years?|ans?)", text.lower())]
    return max(matches, default=0)


def extract_skills(text: str) -> set[str]:
    normalized = normalize_text(text)
    found: set[str] = set()
    for skill, patterns in SKILL_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            found.add(skill)
    return found


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

LABEL2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


def generate_synthetic_dataset(n: int = 300, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    records = []
    candidate_id = 0

    per_bucket = n // 3
    buckets = (
        [("High", _CV_HIGH)] * per_bucket
        + [("Medium", _CV_MEDIUM)] * per_bucket
        + [("Low", _CV_LOW)] * (n - 2 * per_bucket)
    )
    rng.shuffle(buckets)

    for label, pool in buckets:
        job = rng.choice(JOB_TEMPLATES)
        cv = rng.choice(pool)
        # Add minor noise to avoid duplicates
        noise = rng.choice(["", " Team player.", " Good communicator.", " Remote work preferred."])
        records.append(
            {
                "candidate_id": f"cand_{candidate_id:04d}",
                "cv_text": cv + noise,
                "job_id": job["job_id"],
                "job_text": job["job_text"],
                "label": label,
                "label_idx": LABEL2IDX[label],
            }
        )
        candidate_id += 1

    return records


def save_dataset(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class CVScreeningDataset(Dataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        features = featurize(r["cv_text"], r["job_text"])
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(r["label_idx"], dtype=torch.long)
        return x, y
