from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise RuntimeError("MONGO_URL environment variable is not set")
DB_NAME = os.getenv("DB_NAME", "hiring_deployments")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

job_descriptions_collection = db["jobDescriptions"]
responses_collection = db["responses"]
