from flask import Flask, request, jsonify
from flask_cors import CORS 

import app_M12_hourly
import app_M12_daily
import app_M12_weekly
import app_M12_monthly

app = Flask(__name__)
CORS(app)

@app.route("/predict")
def predict():
    granularity = request.args.get("granularity")

    if granularity == "hourly":
        return app_M12_hourly.predict()
    elif granularity == "daily":
        return app_M12_daily.predict()
    elif granularity == "weekly":
        return app_M12_weekly.predict()
    elif granularity == "monthly":
        return app_M12_monthly.predict()
    else:
        return jsonify({"error": "Invalid granularity"}), 400

if __name__ == "__main__":
    app.run(port=5005)
