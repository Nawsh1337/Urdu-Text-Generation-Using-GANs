from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

import model as model
import firebase as fb

import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_dir(ip):
    currDateTime = datetime.now()
    strDate = currDateTime.strftime("%d:%b:%Y")
    strTime = currDateTime.strftime("%H:%M:%S.%f")
    cloudDir = '/images/'+ip+'-'+strDate+'-'+strTime+'/'
    localDir = './images/'+ip+'-'+strDate+'-'+strTime+'/'
    return [cloudDir, localDir]

def clean_dir(local_dir):
    shutil.rmtree(local_dir)





#initializing the model
generator = model.get_model()
#initialize pyrebase
firebase = fb.initFirebase()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/categories/")
async def read_item():
    cats = model.getCategories()
    return {'categories':cats}

@app.get("/predict/")
async def read_item(num_of_examples: int, label: str, ip: str):
    cloudDir, localDir = get_dir(ip)
    cats = model.getCategories()
    intLabel = cats.index(label)
    model.get_images(num_of_examples=num_of_examples, label=intLabel, gen=generator, save_dir=localDir)
    directory = fb.uploadImages(firebase, ip, localDir, cloudDir)
    clean_dir(localDir)
    return  {'msg':'success, files uploaded to bucket','folderName':directory}
