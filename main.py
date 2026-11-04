import os
import re
import json
import time
import pymupdf
from google import genai
from google.genai import types
from schema import ResumeExtraction
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Gemini API key not found in .env file")
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"
RESUME_FOLDER = "resumes"
OUTPUT_FOLDER = "output"
PROMPT_FILE = "prompts/prompt_v1.txt"

def create_sample_resumes():
    os.makedirs(RESUME_FOLDER, exist_ok=True)
    sample_resumes = {
        "resume_01.txt": """
Name: Arun Kumar
Skills: Python, SQL, Machine Learning
Experience: 2 years as Python Developer.
Education: B.Tech Computer Science.
Developed Python applications and SQL databases.
""",
        "resume_02.txt": """
Name: Priya Sharma
Skills: Java, HTML, CSS, JavaScript
Experience: 3 years as Web Developer.
Education: B.Sc Computer Science.
Developed responsive websites using JavaScript.
""",
        "resume_03.txt": """
Name: Rahul Kumar
Skills: Python, Django, MySQL
Experience: 1 year as Backend Developer.
Education: BCA.
Built backend applications using Django.
""",
        "resume_04.txt": """
Name: Sneha Reddy
Skills: Excel, SQL, Power BI
Experience: 4 years as Data Analyst.
Education: MBA Analytics.
Created business dashboards using Power BI.
""",
        "resume_05.txt": """
Name: Karthik Raj
Skills: C, C++, Python
Experience: 2 years as Software Developer.
Education: B.E Computer Science.
Developed software applications using C++.
""",
        "resume_06.txt": """
Name: Divya S
Skills: HTML, CSS, React, JavaScript
Experience: 2 years as Frontend Developer.
Education: B.Tech IT.
Built user interfaces using React.
""",
        "resume_07.txt": """
Name: Vijay Kumar
Skills: Python, SQL, Data Analysis
Experience: 5 years as Data Analyst.
Education: M.Sc Statistics.
Analyzed business data using Python and SQL.
""",
        "resume_08.txt": """
Name: Anitha R
Skills: Java, Spring Boot, MySQL
Experience: 3 years as Java Developer.
Education: B.E Electronics.
Developed applications using Spring Boot.
"""
    }

    for filename, content in sample_resumes.items():
        file_path = os.path.join(
            RESUME_FOLDER,filename
        )
        if not os.path.exists(file_path):
            with open(file_path,"w",encoding="utf-8") as file:
                file.write(content.strip())
    print("8 sample resume files are ready!")

def read_resume(path):
    if not os.path.exists(path):
        return None, "File not found"
    extension = os.path.splitext(path)[1].lower()
    try:
        if extension == ".pdf":
            document = pymupdf.open(path)
            text = ""
            try:
                for page in document:
                    text += page.get_text() + "\n"
            finally:
                document.close()
        elif extension == ".txt":
            with open(path,"r",encoding="utf-8") as file:
                text = file.read()
        else:
            return None, "Unsupported file format"
        if not text.strip():
            return None, "No text found (scanned or empty PDF)"
        text = clean_text(text)
        if not text:
            return None, "No usable text found"
        return text, None
    except Exception as error:
        return None, str(error)

def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(
        r"(?im)^\s*page\s+\d+(\s+of\s+\d+)?\s*$",
        "",text
    )
    text = re.sub(r"(?m)^\s*\d+\s*$","",text)
    text = re.sub(r"\n{2,}","\n",text)
    return text.strip()

def load_prompt():
    with open(PROMPT_FILE,"r",encoding="utf-8") as file:
        return file.read()

def count_tokens(text):
    result = client.models.count_tokens(
        model=MODEL,
        contents=text
    )
    return result.total_tokens

def analyze_resume(text, system_prompt):
    try:
        print("Sending resume to Gemini...")
        response = client.models.generate_content(
            model=MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                response_mime_type="application/json",
                response_schema=ResumeExtraction
            )
        )
        if not response.text:
            raise ValueError("Empty Gemini response")
        result = ResumeExtraction.model_validate_json(
            response.text
        )
        return result.model_dump()
    except Exception as error:
        error_message = str(error)
        print("Gemini Error:", error_message)
        if "503" in error_message:
            print("Gemini server busy. Waiting 5 seconds...")
            time.sleep(5)
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=ResumeExtraction
                    )
                )
                if not response.text:
                    raise ValueError("Empty Gemini response")
                result = ResumeExtraction.model_validate_json(
                    response.text
                )
                return result.model_dump()
            except Exception as retry_error:
                print("Gemini retry failed:", retry_error)
        print("Using Python fallback extraction...")
        return python_fallback(text)

def python_fallback(text):
    result = {
        "candidate_name": None,
        "skills": [],
        "years_of_experience": None,
        "education": [],
        "strengths": [],
        "gaps": [],
        "skill_evidence": [],
        "extraction_method": "Python fallback"
    }
    name_match = re.search(r"(?im)^Name:\s*(.+)$",text)
    if name_match:
        result["candidate_name"] = name_match.group(1).strip()
    skills_match = re.search(r"(?im)^Skills:\s*(.+)$",text)
    if skills_match:
        skills_text = skills_match.group(1)
        skills = [
            skill.strip()
            for skill in skills_text.split(",")
            if skill.strip()
        ]
        result["skills"] = skills
        evidence = skills_match.group(0).strip()
        for skill in skills:
            result["skill_evidence"].append({
                "skill": skill,
                "evidence": evidence
            })
    experience_match = re.search(r"(?i)Experience:\s*(\d+)\s+years?",
        text)
    if experience_match:
        result["years_of_experience"] = int(
            experience_match.group(1)
        )
    education_match = re.search(r"(?im)^Education:\s*(.+)$",text)
    if education_match:
        result["education"].append(
            education_match.group(1).strip()
        )
    return result

def process_resume(filename, system_prompt):
    path = os.path.join(RESUME_FOLDER,filename)
    print("\nResume:", filename)
    text, error = read_resume(path)
    if error:
        print("Error:", error)
        return False
    characters = len(text)
    print("Characters:", characters)
    try:
        tokens = count_tokens(text)
        print("Gemini Tokens:", tokens)
    except Exception as error:
        print("Token count unavailable:", error)
    result = analyze_resume(text,system_prompt)
    output_filename = (os.path.splitext(filename)[0] + ".json")
    output_path = os.path.join(OUTPUT_FOLDER,output_filename)
    with open(output_path,"w",encoding="utf-8") as file:
        json.dump(result,file,indent=4,ensure_ascii=False)
    print("\nExtraction successful!")
    print("Extraction method:", result.get("extraction_method", "Gemini"))
    print("\nJSON Output:")
    print(json.dumps(result, indent=4))
    return True

def main():
    create_sample_resumes()
    os.makedirs(OUTPUT_FOLDER,exist_ok=True)
    system_prompt = load_prompt()
    total_resumes = 0
    successful_resumes = 0
    failed_resumes = 0
    gemini_count = 0
    fallback_count = 0
    resume_files = sorted(
        filename
        for filename in os.listdir(RESUME_FOLDER)
        if filename.lower().endswith((".pdf", ".txt"))
        and os.path.isfile(
            os.path.join(RESUME_FOLDER, filename)
        )
    )

    for index, filename in enumerate(resume_files):
        total_resumes += 1
        try:
            success = process_resume(filename,system_prompt)
            if success:
                successful_resumes += 1
                output_path = os.path.join(
                    OUTPUT_FOLDER,
                    os.path.splitext(filename)[0] + ".json"
                )
                with open(output_path,"r",encoding="utf-8") as file:
                    saved_result = json.load(file)
                if saved_result.get("extraction_method") == "Python fallback":
                    fallback_count += 1
                else:
                    gemini_count += 1
            else:
                failed_resumes += 1
        except Exception as error:
            print("\nResume processing failed:", error)
            failed_resumes += 1
        if index < len(resume_files) - 1:
            print("\nWaiting 15 seconds...")
            time.sleep(15)

    print("\nAI RESUME ANALYZER REPORT")
    print("Total Resumes:", total_resumes)
    print("Successfully Processed:", successful_resumes)
    print("Gemini Extractions:", gemini_count)
    print("Python Fallback Extractions:", fallback_count)
    print("Failed:", failed_resumes)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    except Exception as error:
        print("\nProgram Error:", error)
