import pyrebase

config = {
    "apiKey": "AIzaSyDrmNKmjiIpVXvh2pkV6PYgE6uW5uMGz3U",
    "authDomain": "videouploading-43428.firebaseapp.com",
    "databaseURL": "https://videouploading-43428.firebaseio.com",
    "projectId": "videouploading-43428",
    "storageBucket": "videouploading-43428.appspot.com",
    "messagingSenderId": "25736329298",
    "appId": "1:25736329298:web:4ffec6977654cb6bf9c78b",
    "measurementId": "G-6ZFZB8REKC"
  }


firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

path_to_cloud = "Videos/vidfile.avi"
local_path = "output.avi"
storage.child(path_to_cloud).put(local_path)