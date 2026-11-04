import io
import pytest
import pymupdf
from unittest.mock import patch
import api
from fastapi.testclient import TestClient

client = TestClient(api.app)
SAMPLE_RESUME = b"""
Name: Arun Kumar
Skills: Python, SQL, Machine Learning
Experience: 2 years as Python Developer.
Education: B.Tech Computer Science.
Developed Python applications and SQL databases.
"""
JOB_DESCRIPTION = """
Looking for a Python Developer.
Required skills:
Python, SQL, Django, FastAPI.
"""
MOCK_RESULT = {
    "candidate_name": "Arun Kumar",
    "skills": [
        "Python",
        "SQL",
        "Machine Learning"
    ],
    "years_of_experience": 2,
    "education": [
        "B.Tech Computer Science"
    ],
    "strengths": [
        "Python development experience"
    ],
    "gaps": [],
    "skill_evidence": [
        {
            "skill": "Python",
            "evidence": "Developed Python applications."
        }
    ],
    "extraction_method": "Gemini"
}

def mock_gemini_response():
    return (MOCK_RESULT.copy(),100,50)

def upload_resume(
    filename="resume.txt",
    content=SAMPLE_RESUME,
    content_type="text/plain"
):
    return client.post(
        "/analyze",
        files={
            "file": (
                filename,
                content,
                content_type
            )
        },
        data={
            "job_description": JOB_DESCRIPTION
        }
    )

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["project"] == "AI Resume Analyzer"

def test_analyze_success():
    with patch.object(
        api,
        "analyze_with_retry",
        return_value=mock_gemini_response()
    ), patch.object(
        api,
        "count_tokens",
        return_value=150
    ):
        response = upload_resume()
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_name"] == "Arun Kumar"
    assert data["score"] == 50
    assert "python" in data["matched_skills"]
    assert "django" in data["missing_skills"]
    assert data["token_usage"]["input_tokens"] == 100
    assert data["token_usage"]["output_tokens"] == 50
    assert "request_id" in data

def test_empty_file():
    response = upload_resume(filename="empty.txt",content=b"")
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()

def test_wrong_file_type():
    response = upload_resume(
        filename="resume.exe",
        content=b"sample content",
        content_type="application/octet-stream"
    )
    assert response.status_code == 415

def test_invalid_json_error():
    with patch.object(
        api,
        "analyze_with_retry",
        side_effect=ValueError("Invalid JSON")
    ), patch.object(
        api,
        "count_tokens",
        return_value=150
    ):
        response = upload_resume()
    assert response.status_code == 502

def test_gemini_timeout():
    with patch.object(
        api,
        "analyze_with_retry",
        side_effect=TimeoutError("Gemini timeout")
    ), patch.object(
        api,
        "count_tokens",
        return_value=150
    ):
        response = upload_resume()
    assert response.status_code == 503

def test_gemini_server_error():
    with patch.object(
        api,
        "analyze_with_retry",
        side_effect=Exception("Gemini server unavailable")
    ), patch.object(
        api,
        "count_tokens",
        return_value=150
    ):
        response = upload_resume()
    assert response.status_code == 502

def test_missing_job_description():
    response = client.post(
        "/analyze",
        files={
            "file": (
                "resume.txt",
                SAMPLE_RESUME,
                "text/plain"
            )
        }
    )
    assert response.status_code == 422

def test_missing_resume():
    response = client.post(
        "/analyze",
        data={
            "job_description": JOB_DESCRIPTION
        }
    )
    assert response.status_code == 400

def test_api_docs():
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "/analyze" in data["paths"]
    assert "/health" in data["paths"]

def test_empty_job_description():
    response = client.post(
        "/analyze",
        files={
            "file": (
                "resume.txt",
                SAMPLE_RESUME,
                "text/plain"
            )
        },
        data={
            "job_description": ""
        }
    )
    assert response.status_code == 422

def test_wrong_extension():
    response = upload_resume(
        filename="resume.docx",
        content=b"Sample resume",
        content_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        )
    )
    assert response.status_code == 415

def test_pdf_upload():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (50, 50),
        "Name: Arun Kumar\nSkills: Python, SQL"
    )
    pdf_bytes = document.tobytes()
    document.close()
    with patch.object(
        api,
        "analyze_with_retry",
        return_value=mock_gemini_response()
    ), patch.object(
        api,
        "count_tokens",
        return_value=150
    ):
        response = upload_resume(
            filename="resume.pdf",
            content=pdf_bytes,
            content_type="application/pdf"
        )
    assert response.status_code == 200
    assert response.json()["candidate_name"] == "Arun Kumar"

def test_invalid_health_route():
    response = client.get("/health123")
    assert response.status_code == 404

def test_resume_text_input():
    with patch.object(
        api,
        "analyze_with_retry",
        return_value=mock_gemini_response()
    ), patch.object(
        api,
        "count_tokens",
        return_value=150
    ):
        response = client.post(
            "/analyze",
            data={
                "resume_text": SAMPLE_RESUME.decode("utf-8"),
                "job_description": JOB_DESCRIPTION
            }
        )
    assert response.status_code == 200
    assert response.json()["candidate_name"] == "Arun Kumar"

def test_token_limit_exceeded():
    with patch.object(
        api,
        "count_tokens",
        return_value=13000
    ):
        response = upload_resume()
    assert response.status_code == 413

def test_file_size_exceeded():
    large_content = b"A" * (api.MAX_FILE_SIZE + 1)
    response = upload_resume(
        filename="large.txt",
        content=large_content
    )
    assert response.status_code == 413

def test_empty_resume_text():
    response = client.post(
        "/analyze",
        data={
            "resume_text": "",
            "job_description": JOB_DESCRIPTION
        }
    )
    assert response.status_code == 400

def test_invalid_pdf():
    response = upload_resume(
        filename="invalid.pdf",
        content=b"This is not a valid PDF",
        content_type="application/pdf"
    )
    assert response.status_code == 422

def test_gemini_rate_limit():
    with patch.object(
        api,
        "analyze_with_retry",
        side_effect=Exception("ResourceExhausted")
    ), patch.object(
        api,
        "count_tokens",
        return_value=150
    ):
        response = upload_resume()
    assert response.status_code == 503

def test_calculate_score():
    score, matched, missing = api.calculate_score(
        ["Python", "SQL"],
        "Required skills: Python, SQL, Django"
    )
    assert score == 67
    assert "python" in matched
    assert "sql" in matched
    assert "django" in missing

def test_clean_text():
    raw_text = """
    Name: Arun Kumar
    Skills: Python, SQL
    Page 1
    """
    result = api.clean_text(raw_text)
    assert "Arun Kumar" in result
    assert "Python" in result
    assert "Page 1" not in result
def test_estimated_cost():
    with patch.object(
        api,
        "analyze_with_retry",
        return_value=mock_gemini_response()
    ), patch.object(
        api,
        "count_tokens",
        return_value=150
    ):
        response = upload_resume()
    assert response.status_code == 200
    data = response.json()
    assert "estimated_cost_usd" in data
    assert data["estimated_cost_usd"] >= 0
