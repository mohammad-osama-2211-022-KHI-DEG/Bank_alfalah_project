import firebase_admin
import datetime
from firebase_admin import credentials, firestore

# Initialize Firebase Admin with your credentials (same as before)
cred = credentials.Certificate("/home/hassaanali/Documents/yolov8parkingspace/key2.json")  
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Define the collection where you want to store the parking data
collection_name = "parking_data"

# Initialize variables to store previous values
previous_total_spaces = None
previous_occupied_spaces = None
previous_free_spaces = None
previous_wrongparking = None

def save_parking_data(total_spaces, occupied_spaces, free_spaces, wrongparking):
    global previous_total_spaces, previous_occupied_spaces, previous_free_spaces, previous_wrongparking

    # Check if there's any change in the parking status
    if (
        total_spaces != previous_total_spaces
        or occupied_spaces != previous_occupied_spaces
        or free_spaces != previous_free_spaces
        or wrongparking != previous_wrongparking
    ):
        # Get the current timestamp
        timestamp = datetime.datetime.now().isoformat()

        # Prepare the data to be stored in the document
        result_data = {
            timestamp : {

                "total_spaces": total_spaces,
                "occupied_spaces": occupied_spaces,
                "free_spaces": free_spaces,
                "wrongparking": wrongparking,
            }
            
        }

        # Add the data to Firestore with a new document ID
        new_document_ref = db.collection(collection_name).add(result_data)

        # Get the auto-generated document ID for reference (optional)
        new_document_id = new_document_ref[1].id

        print(f"Data stored in a new document with ID: {new_document_id}")

        # Update the previous values for the next iteration
        previous_total_spaces = total_spaces
        previous_occupied_spaces = occupied_spaces
        previous_free_spaces = free_spaces
        previous_wrongparking = wrongparking
