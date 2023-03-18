import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.models import BaseModel
from core.opt import Config
cfg = Config.load_yaml("configs/default.yml")

model = BaseModel(cfg)
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"ok": "lol"}

@app.post("/file")
async def create_file(image: UploadFile = File(...)):
    tmp = image.file.read()
    decoded = cv2.imdecode(np.frombuffer(tmp, np.uint8), -1)
    result = model.forward(decoded)
    return {"result": result}

