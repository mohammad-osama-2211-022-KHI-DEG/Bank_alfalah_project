import firebase_admin
import datetime
from firebase_admin import credentials, firestore

# Initialize Firebase Admin with your credentials (same as before)
cred = credentials.Certificate("/home/hassaanali/Downloads/Clean-messy/key2.json")
firebase_admin.initialize_app(cred)

def save_cleanliness_result(clean_scene):
    # Initialize Firestore
    db = firestore.client()

    # Define the collection where you want to store the results
    collection_name = "Room_Cleanliness"

    # Prepare the data to be stored in the document
    formatted_datetime = datetime.datetime.now().isoformat()

    result_data = {
        formatted_datetime: "Room is clean" if clean_scene == 'Clean' else "Room is Messy",
        # Add more fields as needed
    }

    try:
        new_document_ref, new_document_id = db.collection(collection_name).add(result_data)
        print(f"Data stored in a new document with ID: {new_document_id}")
    except Exception as e:
        print(f"Error storing data in Firestore: {e}")

