# PIBIT AI Assignment - Automated Prompt Optimization Report

**Generated:** 2026-05-30 14:11:19

---

## Executive Summary

This report documents the results of automated prompt optimization for structured extraction from resumes using LLMs with a greedy optimization strategy.

## Optimization Results

| Metric | Value |
|--------|-------|
| Best F1 Score | 0.8571 |
| Total Iterations | 4 |
| Strategy | greedy |
| Max Iterations Configured | 5 |
| Early Stopping Patience | 3 |

### Performance Over Iterations

## Iteration Details

| Iteration | Best Score | Improved | No Improvement Count |
|-----------|-----------|----------|---------------------|
| 0 | 0.8571 | ✓ | 0 |
| 1 | 0.8571 | ✗ | 1 |
| 2 | 0.8571 | ✗ | 2 |
| 3 | 0.8571 | ✗ | 3 |

## Logging Statistics

- Total Prompts Logged: 12
- Total Scores Logged: 12
- Total Iterations: 4

### Prompt Analysis
- Average Prompt Length: 1032 characters
- Min Prompt Length: 499 characters
- Max Prompt Length: 1311 characters

## API and Performance Statistics

### API Calls
- Total API Calls: 0
- Total Tokens: 0
- Total Cost: $0.0000
- Models Used: 

### Latency
- Total Time: 0.00 seconds
- Average Latency: 0.0000 seconds
- Min Latency: 0.0000 seconds
- Max Latency: 0.0000 seconds

## Best Optimized Prompt

```
Extract structured information from the following resume text.
Return a JSON object with the following fields:
- name: Full name of the person
- email: Email address
- phone: Phone number
- skills: List of technical and professional skills
- experience: Work experience entries (job title, company, dates)
- education: Educational background (degree, institution, graduation date)
- certifications: Professional certifications

Resume text:
{resume_text}

Return only valid JSON, no additional text.

Example output format:
{
    "name": "Jane Smith",
    "email": "jane@example.com",
    "phone": "+1-555-1234",
    "skills": ["Python", "Machine Learning"],
    "experience": [{"title": "Data Scientist", "company": "TechCorp", "duration": "2020-2023"}],
    "education": [{"degree": "M.S. Computer Science", "institution": "University", "graduation": "2020"}],
    "certifications": ["AWS Certified"]
}

```

## Recommendations

- **Manual Review:** Review the best optimized prompt and consider fine-tuning it based on domain knowledge.
- **Extended Testing:** Evaluate the best prompt on the test set before production deployment.
- **Prompt Versioning:** Document and version the prompts for reproducibility.

---

*End of Report*
