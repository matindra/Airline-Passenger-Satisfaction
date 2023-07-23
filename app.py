from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomClass
from src.logger import logging
import os
from werkzeug.utils import  secure_filename
from prediction.batch import  batch_prediction
from src.config.configuration import MODEL_FILE_PATH, FEATURE_ENG_OBJ_PATH, PREPROCESSING_OBJ_PATH


UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'
ALLOWED_EXTENSIONS = {'csv'}

feature_engineering_file_path = FEATURE_ENG_OBJ_PATH
transformer_file_path = PREPROCESSING_OBJ_PATH
model_file_path = MODEL_FILE_PATH


app = Flask(__name__, template_folder='templates')



@app.route('/')
def home_page():
    return render_template('index.html')


@app.route("/predict", methods = ["GET", "POST"])
def prediction_data():

    logging.info("defining prediction data")
    if request.method == "GET":
        return render_template("single_prediction_home.html")
    

    else:
        data = CustomClass(
            Age = int(request.form.get("Age")),
            Flight_Distance = int(request.form.get("Flight_Distance")),
            Inflight_wifi_service = int(request.form.get("Inflight_wifi_service")),
            Departure_Arrival_time_convenient = int(request.form.get("Departure_Arrival_time_convenient")),
            Ease_of_Online_booking = int(request.form.get("Ease_of_Online_booking")),
            Gate_location = int(request.form.get("Gate_location")),
            Food_and_drink = int(request.form.get("Food_and_drink")),
            Online_boarding = int(request.form.get("Online_boarding")),
            Seat_comfort = int(request.form.get("Seat_comfort")),
            Inflight_entertainment = int(request.form.get("Inflight_entertainment")),
            On_board_service = int(request.form.get("On_board_service")),
            Leg_room_service = int(request.form.get("Leg_room_service")),
            Baggage_handling = int(request.form.get("Baggage_handling")),
            Checkin_service = int(request.form.get("Checkin_service")),
            Inflight_service = int(request.form.get("Inflight_service")),
            Cleanliness = int(request.form.get("Cleanliness")),
            Departure_Delay_in_Minutes = int(request.form.get("Departure_Delay_in_Minutes")),
            Gender = str(request.form.get("Gender")),
            Customer_Type = str(request.form.get("Customer_Type")),
            Type_of_Travel = str(request.form.get("Type_of_Travel")),
            Class = str(request.form.get("Class")),
            Arrival_Delay_in_Minutes = int(request.form.get("Arrival_Delay_in_Minutes")),

        )

    final_data = data.get_data_DataFrame()
    pipeline_prediction = PredictionPipeline()
    pred = pipeline_prediction.predict(final_data)

    result = pred


    logging.info("returning prediction data")
    # if result == 0:
    #     return render_template("results.html", final_result = "Survey Opinion of the customer is satisfied:{}".format(result) )

    # elif result == 1:
    #         return render_template("results.html", final_result = "Survey Opinion of the customer is dissatisfied or neutral:{}".format(result))


    return render_template("results.html", final_result="Survey Opinion of the customer is: {}".format(result))


@app.route("/batch", methods=['GET','POST'])

def perform_batch_prediction():
    
    logging.info("performing batch prediction")
    if request.method == 'GET':
        return render_template('batch.html')
    else:
        file = request.files['csv_file']  # Update the key to 'csv_file'
        # Directory path
        directory_path = UPLOAD_FOLDER
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)

        # Check if the file has a valid extension
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            # Delete all files in the file path
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Save the new file to the uploads directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(file_path)

            logging.info("CSV received and Uploaded")

            # Perform batch prediction using the uploaded file
            batch = batch_prediction(file_path,
                                    model_file_path,
                                    transformer_file_path,
                                    feature_engineering_file_path)
            batch.start_batch_prediction()

            output = "Batch Prediction Done"

            logging.info("Batch Prediction Done")
            return render_template("batch.html", prediction_result=output, prediction_type='batch')
        else:
            return render_template('batch.html', prediction_type='batch', error='Invalid file type')
        


# @app.route('/train', methods=['GET', 'POST'])
# def train():
#     if request.method == 'GET':
#         return render_template('train.html')
#     else:
#         try:
#             pipeline = Train()
#             pipeline.main()

#             return render_template('train.html', message="Training complete")

#         except Exception as e:
#             logging.error(f"{e}")
#             error_message = str(e)
#             return render_template('index.html', error=error_message)
        

    
if __name__ == "__main__":
     app.run(host = "0.0.0.0", debug = True)