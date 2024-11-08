from flask import Flask, render_template, jsonify, request
import os
from text_comparator.comparator import compare_text

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    left_text = data.get('left_text', '')
    right_text = data.get('right_text', '')
    
    comparison = compare_text(left_text, right_text)
    
    return jsonify({
        'added_words': comparison.added_words,
        'word_count_score': comparison.word_count_score,
        'left_tokens': [{'text': t.text, 'start': t.start} for t in comparison.left_tokens],
        'right_tokens': [{'text': t.text, 'start': t.start} for t in comparison.right_tokens],
        'matches': [{
            'left_start': m.left_start,
            'right_start': m.right_start,
            'length': len(m.source_tokens_left)
        } for m in comparison.matches]
    })

if __name__ == '__main__':
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    app.static_folder = static_folder
    app.run(debug=True, port=5000)
