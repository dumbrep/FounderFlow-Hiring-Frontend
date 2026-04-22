from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URL = os.getenv("MONGO_URL", "mongodb+srv://prajwaldumbre_db_user:Prajwal123@cluster0.er5qxvd.mongodb.net/?appName=Cluster0")
DB_NAME = os.getenv("DB_NAME", "hiring_deployments")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

job_descriptions_collection = db["jobDescriptions"]
responses_collection = db["responses"]
