from flask import Flask, jsonify, render_template, request

from src.pipelines.predict_pipeline import PredictPipeline

app = Flask(__name__)
predictor = None
predictor_error = None
try:
    predictor = PredictPipeline()
except Exception as exc:  # pragma: no cover
    predictor_error = str(exc)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        print('No predictor')
        return jsonify({"prediction": "", "error": predictor_error}), 500

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"prediction": ""})

    generated = predictor.predict_next_words(seed_text=text, next_words=1)
    generated_tokens = generated.split()
    input_tokens = text.split()
    next_token = generated_tokens[len(input_tokens)] if len(generated_tokens) > len(input_tokens) else ""

    return jsonify({"prediction": next_token, "generated_text": generated})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
