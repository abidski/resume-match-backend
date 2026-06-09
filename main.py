import json
import os
import tempfile

import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq, RateLimitError

load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://resume.abid.ink"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze(
    job_description: str = Form(...),
    resume_text: str = Form(""),
    resume_file: UploadFile = File(None),
):
    resume = None
    if resume_file:
        resume = extract_pdf(resume_file)
    else:
        resume = resume_text

    if not job_description:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Missing job_description"
        )
    if resume is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Missing job_description"
        )

    posting = job_description

    result = send_prompt(resume, posting)
    return result


# From Claude.ai
def extract_pdf(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    os.unlink(tmp_path)
    return text


def send_prompt(resume, job_description):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert career coach. You always respond with valid JSON only. Never include markdown, backticks, or any text outside the JSON object.",
                },
                {
                    "role": "user",
                    "content": create_prompt(resume, job_description),
                },
            ],
            temperature=0,
            max_completion_tokens=512,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )

        # From Claude
        raw = completion.choices[0].message.content

        if raw is None:
            raise Exception("Error getting LLM prompt")

        # safety net — strip backticks if the model adds them anyway
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        return json.loads(clean)

    except RateLimitError:
        raise Exception("Error getting llm")


# From Claude.ai
def create_prompt(resume, job_description):
    return f"""You are an expert career coach and ATS (Applicant Tracking System) specialist.

Analyze the resume against the job description below.

CRITICAL INSTRUCTIONS:
- Return ONLY a valid JSON object
- No markdown, no backticks, no explanation before or after
- No text outside the JSON object whatsoever
- Start your response with {{ and end with }}

RESUME:
{resume}

JOB DESCRIPTION:
{job_description}

Return exactly this JSON structure:
{{
  "score": <integer 0-100 overall fit score>,
  "verdict": "<max 6 words, punchy and honest>",
  "summary": "<2-3 sentences honest assessment>",
  "dimensions": [
    {{"name": "Skills Match", "score": <0-100>}},
    {{"name": "Experience Level", "score": <0-100>}},
    {{"name": "Keywords", "score": <0-100>}},
    {{"name": "Education", "score": <0-100>}},
    {{"name": "Culture Fit", "score": <0-100>}}
  ],
  "matching": ["<matched skill or quality>"],
  "partial": ["<transferable or partial match>"],
  "missing": ["<required skill not in resume>"],
  "suggestions": [
    "<specific actionable suggestion 1>",
    "<specific actionable suggestion 2>",
    "<specific actionable suggestion 3>",
    "<specific actionable suggestion 4>"
  ]
}}"""
