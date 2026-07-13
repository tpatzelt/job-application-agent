"""Search profiles used by the integration eval.

Each profile is a synthetic but realistic user: a CV summary plus the
preferences dict in the same shape `preferences.json` / the bot intake
produce. Locations are deliberately varied (Berlin, Munich, Amsterdam,
remote) so the eval exercises location handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalProfile:
    name: str
    cv_text: str
    preferences: dict[str, Any] = field(default_factory=dict)

    @property
    def locations(self) -> list[str]:
        return list(self.preferences.get("locations", []))


PROFILES: list[EvalProfile] = [
    EvalProfile(
        name="ml-engineer-berlin",
        cv_text=(
            "Senior Machine Learning Engineer with 8 years of experience "
            "building and shipping NLP products. Deep hands-on work with "
            "LLMs: retrieval-augmented generation (RAG) pipelines, agent "
            "systems with LangChain, fine-tuning and serving open models "
            "from HuggingFace, and production integrations of OpenAI and "
            "Anthropic APIs. Strong PyTorch and TensorFlow background; "
            "built training and evaluation infrastructure with MLflow. "
            "Ships full services: FastAPI backends, PostgreSQL, Docker, "
            "Kubernetes, CI/CD, deployed on Azure and AWS. Comfortable "
            "across the stack with Python, TypeScript and React for "
            "internal tools. Previously led a team of four engineers "
            "building a document-intelligence platform processing millions "
            "of pages per month. MSc in Computer Science. Based in Berlin, "
            "fluent English, working German."
        ),
        preferences={
            "location": "Berlin, Germany",
            "locations": ["Berlin, Germany"],
            "job_titles": [
                "Senior Machine Learning Engineer",
                "Senior AI Engineer",
                "NLP Engineer",
            ],
            "job_description_keywords": [
                "NLP", "LLMs", "RAG", "LangChain", "OpenAI", "Anthropic",
                "HuggingFace", "PyTorch", "TensorFlow", "FastAPI", "Docker",
                "Kubernetes", "Azure", "AWS", "PostgreSQL", "MLflow",
                "Python", "JavaScript", "TypeScript", "React", "REST",
                "CI/CD", "Git", "SQL",
            ],
        },
    ),
    EvalProfile(
        name="project-manager-berlin",
        cv_text=(
            "Senior digital project manager focused on complex digital "
            "transformation initiatives in the public sector. Independently "
            "managed digital projects for public and private clients, "
            "advised federal ministries on digital products and services, "
            "and acted as the liaison across technical, organizational and "
            "business domains. Methodologically versatile: agile, classical "
            "and hybrid project management (Scrum, Kanban, OKRs, Design "
            "Thinking). Domain expertise in user experience research, "
            "digital accessibility and user-centered design; previously "
            "Project Lead and Senior Research Consultant for UX and "
            "accessibility. Trilingual (Spanish native, German and English "
            "C2), international study and work experience. Tools: Jira, "
            "SPSS, SQL, CMS platforms. Committed to public-interest "
            "technology and meaningful digitalization projects."
        ),
        preferences={
            "location": "Berlin, Germany",
            "locations": ["Berlin, Germany"],
            "job_titles": [
                "IT Project Manager",
                "Digital Project Manager",
                "Project Manager Digital Transformation",
                "Projektmanagerin Digitalisierung",
            ],
            "job_description_keywords": [
                "project management", "digital transformation",
                "public sector", "nonprofit", "accessibility",
                "user experience", "agile", "Scrum",
            ],
        },
    ),
    EvalProfile(
        name="frontend-developer-munich",
        cv_text=(
            "Frontend engineer with 6 years of experience building web "
            "applications with React, TypeScript and Next.js. Strong focus "
            "on design systems, accessibility (WCAG), performance budgets "
            "and testing (Jest, Playwright). Built and maintained a "
            "component library used by five product teams; migrated a "
            "large legacy Angular app to React incrementally. Comfortable "
            "with Node.js backends, GraphQL and REST APIs, CI/CD with "
            "GitHub Actions, and Docker. BSc in Media Informatics from LMU "
            "Munich. Native German, fluent English. Looking for a product "
            "company in Munich with a strong engineering culture."
        ),
        preferences={
            "location": "Munich, Germany",
            "locations": ["Munich, Germany"],
            "job_titles": [
                "Frontend Developer",
                "Frontend Engineer",
                "React Developer",
                "Senior Frontend Engineer",
            ],
            "job_description_keywords": [
                "React", "TypeScript", "Next.js", "JavaScript", "CSS",
                "design system", "accessibility", "GraphQL", "REST",
                "Jest", "Playwright", "CI/CD",
            ],
        },
    ),
    EvalProfile(
        name="data-engineer-amsterdam",
        cv_text=(
            "Data engineer with 7 years of experience designing and "
            "operating batch and streaming data platforms. Built lakehouse "
            "architectures on AWS with Spark, Airflow, dbt and Snowflake; "
            "real-time pipelines with Kafka and Flink. Strong SQL and "
            "Python, infrastructure as code with Terraform, observability "
            "with Grafana. Led the migration of a monolithic ETL estate to "
            "an event-driven architecture serving 40+ analysts and three "
            "ML teams. Experience with data governance, GDPR compliance "
            "and cost optimization. MSc in Data Science, TU Delft. Based "
            "in the Netherlands; seeking roles in Amsterdam."
        ),
        preferences={
            "location": "Amsterdam, Netherlands",
            "locations": ["Amsterdam, Netherlands"],
            "job_titles": [
                "Data Engineer",
                "Senior Data Engineer",
                "Analytics Engineer",
            ],
            "job_description_keywords": [
                "Spark", "Airflow", "dbt", "Snowflake", "Kafka", "AWS",
                "Python", "SQL", "Terraform", "data pipeline", "ETL",
                "lakehouse",
            ],
        },
    ),
    EvalProfile(
        name="backend-engineer-remote-germany",
        cv_text=(
            "Backend engineer with 9 years of experience building "
            "distributed systems in Python and Go. Designed and operated "
            "high-throughput APIs (FastAPI, gRPC), event-driven services "
            "on Kafka and RabbitMQ, and PostgreSQL/Redis data layers. "
            "Deep Kubernetes and AWS experience, infrastructure as code "
            "with Terraform, SRE practices (SLOs, on-call, incident "
            "reviews). Worked fully remote for the last five years in "
            "distributed teams across European time zones. Contributor to "
            "open-source Python tooling. Based in Germany and looking for "
            "a remote-first role with a company hiring in Germany."
        ),
        preferences={
            "location": "Remote, Germany",
            "locations": ["Remote", "Germany"],
            "job_titles": [
                "Senior Backend Engineer",
                "Backend Developer Python",
                "Platform Engineer",
            ],
            "job_description_keywords": [
                "Python", "Go", "FastAPI", "gRPC", "Kafka", "PostgreSQL",
                "Redis", "Kubernetes", "AWS", "Terraform", "remote",
                "distributed systems",
            ],
        },
    ),
]

PROFILES_BY_NAME: dict[str, EvalProfile] = {p.name: p for p in PROFILES}
