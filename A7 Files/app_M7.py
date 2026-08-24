from flask import Flask, request, jsonify
from flask_cors import CORS 


import app_M7_hourly
import app_M7_daily
import app_M7_weekly
import app_M7_monthly

app = Flask(__name__)
CORS(app)

@app.route("/predict")
def predict():
    granularity = request.args.get("granularity")

    if granularity == "hourly":
        return app_M7_hourly.predict()
    elif granularity == "daily":
        return app_M7_daily.predict()
    elif granularity == "weekly":
        return app_M7_weekly.predict()
    elif granularity == "monthly":
        return app_M7_monthly.predict()
    else:
        return jsonify({"error": "Invalid granularity"}), 400

if __name__ == "__main__":
    app.run(port=5014)
