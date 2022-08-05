import pyrebase
import os
from datetime import datetime
firebaseConfig = {
  'apiKey': "AIzaSyDexfRVzloKhyg4Jsxvlbnvg1GtjLRL488",
  'authDomain': "urdugan-fd007.firebaseapp.com",
  'projectId': "urdugan-fd007",
  'databaseURL': "https://databaseName.firebaseio.com",
  'storageBucket': "urdugan-fd007.appspot.com",
  'messagingSenderId': "872703526029",
  'appId': "1:872703526029:web:470644a5e72aa1c83ccec4",
  'measurementId': "G-KBX7DHDMM8"
}

def initFirebase():
    return pyrebase.initialize_app(firebaseConfig)

def getStorage(firebase):
    return firebase.storage()

def uploadImages(firebase, ip, localDir, cloudDir):
  storage = getStorage(firebase)
  images = os.listdir(localDir)
  currDateTime = datetime.now()
  strDate = currDateTime.strftime("%d:%b:%Y")
  strTime = currDateTime.strftime("%H:%M:%S.%f")
  #cloudDirectory = '/images/'+ip+'-'+strDate+'-'+strTime+'/'
  cloudDirectory = cloudDir
  print(images)
  for image in images:
    storage.child(cloudDirectory+image).put(localDir+image)
  return cloudDirectory