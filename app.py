from flask import Flask, jsonify, request, send_from_directory
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from wumpus import create_world, agent_step, clause_to_string, key

app = Flask(__name__)

TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

@app.route('/')
def index():
    return send_from_directory(TEMPLATES_DIR, 'index.html')

@app.route('/api/new', methods=['POST'])
def new_game():
    data = request.get_json()
    rows = max(3, min(7, int(data.get('rows', 4))))
    cols = max(3, min(7, int(data.get('cols', 4))))
    pits = max(1, min(rows * cols - 3, int(data.get('pits', 3))))
    world = create_world(rows, cols, pits)
    return jsonify(serialize(world))

@app.route('/api/step', methods=['POST'])
def step():
    data = request.get_json()
    world = deserialize(data['world'])
    world = agent_step(world)
    return jsonify(serialize(world))

def serialize(world):
    return {
        'rows':            world['rows'],
        'cols':            world['cols'],
        'hazards':         world['hazards'],
        'gold':            list(world['gold']),
        'agent':           list(world['agent']),
        'visited':         world['visited'],
        'safe':            world['safe'],
        'kb':              world['kb'],
        'inference_steps': world['inference_steps'],
        'agent_steps':     world['agent_steps'],
        'percepts':        world['percepts'],
        'log':             world['log'][:20],
        'done':            world['done'],
        'outcome':         world['outcome'],
        'kb_display':      [clause_to_string(c) for c in world['kb'][-8:]][::-1],
    }

def deserialize(data):
    return {
        'rows':            data['rows'],
        'cols':            data['cols'],
        'hazards':         data['hazards'],
        'gold':            tuple(data['gold']),
        'agent':           tuple(data['agent']),
        'visited':         data['visited'],
        'safe':            data['safe'],
        'kb':              data['kb'],
        'inference_steps': data['inference_steps'],
        'agent_steps':     data['agent_steps'],
        'percepts':        data['percepts'],
        'log':             data['log'],
        'done':            data['done'],
        'outcome':         data['outcome'],
    }

if __name__ == '__main__':
    app.run(debug=True)
