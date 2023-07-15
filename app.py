from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomClass
from src.logger import logging

app = Flask(__name__, template_folder='templates')


@app.route("/",methods = ["GET", "POST"])



def prediction_data():

    logging.info("defining prediction data")
    if request.method == "GET":
        return render_template("home.html")
    

    
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
    
if __name__ == "__main__":
     app.run(host = "0.0.0.0", debug = True)