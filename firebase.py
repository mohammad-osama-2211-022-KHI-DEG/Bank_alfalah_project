from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# # Initialize Firebase Admin SDK
cred = credentials.Certificate("smilefirebase.json")
firebase_admin.initialize_app(cred)

#Initialize Firestore
db = firestore.client()

# Finally, add the total happy count to Firestore
doc_ref = db.collection('smiles').add({
     'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
     'total_happy_count': happy_count
 })
