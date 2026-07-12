MOCK_BRAVE_RESPONSE = {
    "web": {
        "results": [
            {
                "url": "https://example.com/jobs/python-developer",
                "title": "Python Developer",
            },
            {
                "url": "https://example.com/jobs/backend-engineer",
                "title": "Backend Engineer",
            },
        ]
    }
}

MOCK_JOB_TEXT = "We are looking for a Python developer. " + ("Details " * 200)

MOCK_QUERY_RESPONSE = {
    "queries": [
        "python jobs berlin",
        "remote backend developer",
    ]
}

MOCK_EVALUATION_RESPONSE = {
    "score": 85,
    "reason": "Good match",
}

MOCK_PLAN_RESPONSE = {
    "target_roles": ["Python Developer", "Backend Engineer"],
    "key_skills": ["Python", "APIs"],
    "locations": ["Berlin", "Remote"],
    "strategy": "Search job boards for junior Python roles in Berlin or remote.",
}

MOCK_REFLECTION_RESPONSE = {
    "assessment": "Both queries produced new listings.",
    "effective_queries": ["python jobs berlin"],
    "ineffective_queries": [],
    "adjustments": ["Add seniority keywords to narrow results."],
}
