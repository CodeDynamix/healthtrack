from fastapi import FastAPI, UploadFile, File
import tempfile
import os
import subprocess
import json
import re

app = FastAPI()

@app.get("/")
def root():
    return {"status": "AI service running"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, image.filename or "upload.jpg")

        with open(image_path, "wb") as f:
            f.write(await image.read())

        try:
            result = subprocess.check_output(
                ["python", "ai_predict_new.py", image_path],
                stderr=subprocess.STDOUT,
                text=True
            )

            # Extract JSON from output
            matches = re.findall(r"\{.*\}", result, re.S)
            if not matches:
                return {"success": False, "error": "No JSON returned", "raw": result}

            return json.loads(matches[-1])

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": "Prediction failed",
                "details": e.output
            }