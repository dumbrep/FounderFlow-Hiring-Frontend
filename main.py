from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from database import job_descriptions_collection, responses_collection
from models import JobDescription, ResponseOut
from bson import ObjectId, Binary
import tempfile
import os
from ats_scorer import (
    extract_text, split_sections, score_keywords, score_skills,
    score_experience, score_education, score_semantic, compute_ats_score,
)

app = FastAPI(title="Hiring Deployments API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://founder-flow-hiring-frontend-l2ja.vercel.app",
        "https://founderflow-hiring-frontend-main.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/jobs", response_model=list[JobDescription])
async def get_job_descriptions():
    jobs = []
    async for doc in job_descriptions_collection.find():
        jobs.append(JobDescription(
            id=str(doc["_id"]),
            job_role=doc["job_role"],
            description=doc["description"],
        ))
    return jobs


@app.post("/api/apply", response_model=ResponseOut)
async def apply_for_job(
    user_name: str = Form(...),
    email: str = Form(...),
    job_role: str = Form(...),
    resume: UploadFile = File(...),
):
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await resume.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be under 5MB.")

    # Fetch job description from DB for ATS scoring
    job_doc = await job_descriptions_collection.find_one({"job_role": job_role})
    if not job_doc:
        raise HTTPException(status_code=404, detail="Job role not found.")
    job_description_text = job_doc["description"]

    # Write PDF to a temp file, run ATS pipeline, then delete
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        os.write(tmp_fd, file_bytes)
        os.close(tmp_fd)

        resume_text = extract_text(tmp_path)
        sections = split_sections(resume_text)

        kw = score_keywords(resume_text, job_description_text)
        sk = score_skills(resume_text, job_description_text)
        exp_section = sections["experience"] if sections["experience"] else resume_text
        exp_score, _ = score_experience(exp_section, job_description_text)
        edu_score, _ = score_education(resume_text, job_description_text)
        summary_text = sections["summary"] if sections["summary"] else resume_text[:500]
        sem = score_semantic(summary_text, job_description_text)

        final_score, _ = compute_ats_score(kw, sk, exp_score, edu_score, sem)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    doc = {
        "user_name": user_name,
        "email": email,
        "job_role": job_role,
        "resume_filename": resume.filename,
        "resume_data": Binary(file_bytes),
        "ats_score": final_score,
    }
    result = await responses_collection.insert_one(doc)

    return ResponseOut(
        id=str(result.inserted_id),
        user_name=user_name,
        email=email,
        job_role=job_role,
        resume_filename=resume.filename,
        ats_score=final_score,
    )


@app.get("/api/resume/{response_id}")
async def get_resume(response_id: str):
    doc = await responses_collection.find_one({"_id": ObjectId(response_id)})
    if not doc or "resume_data" not in doc:
        raise HTTPException(status_code=404, detail="Resume not found.")
    return Response(
        content=bytes(doc["resume_data"]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{doc["resume_filename"]}"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
