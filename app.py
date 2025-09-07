from flask import Flask, request, jsonify

from service import get_model, predict_probability

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'file is required'}), 400
    file = request.files['file']
    probability = predict_probability(file.stream)
    return jsonify({'probability': probability})


@app.route('/health', methods=['GET'])
def health():
    try:
        get_model()
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
