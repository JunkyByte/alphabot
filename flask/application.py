from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from array2gif import write_gif
import numpy as np
import cv2
import sys
import keras
from keras.models import load_model
from multiprocessing import Process
from time import sleep
import tensorflow as tf
import os
import logging
sys.path.append('../src/')
from mcts import simulate_game
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)
app.config["SECRETE_KEY"] = "secret"
socketio = SocketIO(app)
logging.basicConfig(filename='flask_logging.log', level=logging.INFO, format='%(asctime)s %(message)s', filemode='w')


@app.route("/")
def index():
    test = {
        "messaggio": "ciao"
    }
    socketio.emit("new", test)
    return render_template("index.html")


@socketio.on("ready")
def update():
    test = {
        "messaggio": "ciao"
    }
    emit("new", test)


@app.route("/get_name", methods=["POST"])
def get_name():
    name = request.values.get('name', '')
    all_files = [f for f in os.listdir('./static/') if f.endswith('.gif') and f not in name]
    if len(all_files) == 0:
        return jsonify({'success': False, 'name': 'nameplaceholder'})

    idx = np.random.randint(0, len(all_files))
    name = all_files[idx]

    return jsonify({'success': True, 'name': os.path.join('/static/', name)})


def round_to_closest(mapp):
    colors = [128, 255]
    mapp[~np.isin(mapp, colors)] = 0
    return mapp
    

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

        mapp = cv2.resize(mapp.astype(np.uint8), (480, 480), interpolation=cv2.INTER_AREA)
        mapp = round_to_closest(mapp)
        
        maps.append(mapp)

    maps = np.array(maps)

    gif_path = os.path.join('./static/', name + '.gif')
    try:
        os.remove(gif_path)
    except Exception:
        pass

    write_gif(maps, gif_path, fps=5)


def async_sims():
    steps = 30
    alpha = 1.
    runs = 5
    model_path = '../alphabot_best.pickle'
    logging.debug('Loading first model')
    alphabot = load_model(model_path, custom_objects={'categorical_weighted': keras.losses.categorical_crossentropy, 'tf': tf})
    last_modifications = -1

    while True:
        if os.path.getmtime(model_path) != last_modifications:
            logging.debug('Model has been modified, running simulation')
            last_modifications = os.path.getmtime(model_path)
            alphabot.load_weights(model_path)
            for i in range(runs):
                sim_to_gif('run_' + str(i), steps, alpha, alphabot)
                logging.debug('Simulation %d has finished' % i)
        sleep(180)


if __name__ == '__main__' or __name__ == 'application':
    # Run async simulator
    worker = Process(target=async_sims)
    worker.daemon = False
    worker.start()
    app.run()
