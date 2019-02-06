from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from array2gif import write_gif
import numpy as np
import cv2
import sys
sys.path.append('../src/')
from mcts import simulate_game
import keras
from keras.models import load_model
from multiprocessing import Process
from time import sleep
import os
import logging
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)
app.config["SECRETE_KEY"] = "secret"
socketio = SocketIO(app)
logging.basicConfig(filename='flask_logging.log', level=logging.DEBUG, format='%(asctime)s %(message)s', filemode='w')

@app.route("/")
def index():
    test = {
        "messaggio" : "ciao"
    }
    socketio.emit("new", test)
    return render_template("index.html")

@socketio.on("ready")
def update():
    test = {
        "messaggio" : "ciao"
    }
    emit("new", test)

@app.route("/get_name", methods=["GET"])
def get_name():
    all_files = [file for file in os.listdir('./static/') if file.endswith('.gif')]
    if len(all_files) == 0:
        return jsonify({'success' : False, 'name' : 'nameplaceholder'})

    idx = np.random.randint(0, len(all_files))
    name = all_files[idx]

    return jsonify({'success' : True, 'name' : os.path.join('/static/', name)})


def sim_to_gif(name, steps, alpha, alphabot):
    states = simulate_game(steps, alpha, alphabot=alphabot, eval_g=True, return_state=True)
    logging.debug('Game was successfully simulated')

    maps = []
    for state in states:
        mapp = state[..., 0]
        mapp += state[..., 2] * 2
        mapp[np.where(state[..., 1] == 1)] = 3
        mapp[np.where(state[..., 3] == 1)] = 4
        mapp = np.expand_dims(mapp, axis=-1)
        mapp = np.tile(mapp, [1, 1, 3])

        idx, cols, c = np.where(mapp == 1)
        mapp[idx, cols, :] = 0
        mapp[idx, cols, 0] = 128

        idx, cols, c = np.where(mapp == 2)
        mapp[idx, cols, :] = 0
        mapp[idx, cols, 1] = 128

        idx, cols, c = np.where(mapp == 3)
        mapp[idx, cols, :] = 0
        mapp[idx, cols, 0] = 255

        idx, cols, c = np.where(mapp == 4)
        mapp[idx, cols, :] = 0
        mapp[idx, cols, 1] = 255

        maps.append(mapp)

    maps = np.array(maps)
    write_gif(maps, os.path.join('./static/', name + '.gif'), fps=5)

def async_sims():
    steps = 25
    alpha = 1.
    runs = 10
    model_path = '../alphabot_best.pickle'
    logging.debug('Loading first model')
    alphabot = load_model(model_path, custom_objects={'categorical_weighted' : keras.losses.categorical_crossentropy})
    last_modifications = os.path.getmtime(model_path)
    for i in range(runs):
        sim_to_gif('run_' + str(i), steps, alpha, alphabot)
        logging.debug('Simulation %d has finished' % i)

    while True:
        logging.debug('Sleeping')
        time.sleep(120)
        if os.path.getmtime(model_path) != last_modifications:
            logging.debug('Model has been modified, running simulation')
            last_modifications = os.path.getmtime(model_path)
            alphabot.load_weights(model_path)
            for i in range(runs):
                sim_to_gif('run_' + str(i), steps, alpha, alphabot)
                logging.debug('Simulation %d has finished' % i)

if __name__ == '__main__' or __name__ == 'application':
    # Run async simulator
    worker = Process(target=async_sims)
    worker.daemon = False
    worker.start()
    app.run()
