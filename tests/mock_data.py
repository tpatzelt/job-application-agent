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
