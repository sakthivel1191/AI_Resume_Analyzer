
# AI Resume Analyzer

## 1. Project Purpose

AI Resume Analyzer is a Python project that uses
Google Gemini to extract information from resumes
and compare candidate skills with job descriptions.

## 2. Technologies Used

- Python
- FastAPI
- Google Gemini API
- PyMuPDF
- Pydantic
- Pytest
- JSON

## 3. Project Architecture

Resume PDF/TXT
      |
      v
FastAPI Upload Endpoint
      |
      v
Text Extraction and Cleaning
      |
      v
Gemini API
      |
      v
Structured JSON Extraction
      |
      v
Skill Matching and Scoring
      |
      v
API Response

## 4. Installation

Install Python packages:

pip install fastapi uvicorn python-multipart
pip install google-genai pydantic pymupdf
pip install python-dotenv pytest httpx pytest-cov

## 5. Environment Setup

Create a .env file:

GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.5-flash

Never upload your actual API key to GitHub.

## 6. Run the Project

Start the API:

uvicorn api:app --reload

Open:

http://127.0.0.1:8000/docs

## 7. API Endpoints

GET /health

Checks whether the API is running.

POST /analyze

Accepts a resume and job description.

Returns:

- Candidate information
- Extracted skills
- Matching score
- Matched skills
- Missing skills
- Reasoning
- Token usage
- Estimated API cost

## 8. Example curl

curl -X POST "http://127.0.0.1:8000/analyze" \
-F "file=@resume_01.pdf" \
-F "job_description=Python SQL Developer"

## 9. Run Tests

pytest -v

Check coverage:

pytest --cov=api --cov-report=term-missing

## 10. Limitations

- Scanned PDFs require OCR support.
- Gemini may occasionally extract incorrect information.
- Matching score uses basic keyword comparison.
- Free API usage has rate limits.
- Token cost is an estimate.
- Results require human verification.

## 11. Fairness Note

This application is an educational resume
analysis project.

It must not automatically reject or select
candidates.

AI results can contain mistakes and bias.

A human reviewer must verify all results.

Personal information should be handled
securely and only with appropriate consent.
