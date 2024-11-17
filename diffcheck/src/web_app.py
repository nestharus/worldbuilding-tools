from flask import Flask, render_template, request, jsonify
from main import generate_diff_report

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    left_text = data.get('left_text', '')
    right_text = data.get('right_text', '')
    
    report = generate_diff_report(left_text, right_text)
    return jsonify({'report': report})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
