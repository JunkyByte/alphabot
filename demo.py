import random
import requests
import json
import time

url = 'https://lightningbot.tk/api/test'
# name + a random number so you can launch two bot to play alone. If only one bot is connected the game will not start
botName = 'Python' + str(random.randint(0, 9999))

def moveBot(direction, current_turn):
    res = requests.get('/'.join([url, 'move', connect['token'], str(direction), str(current_turn)]))
    return json.loads(res.text)


def infoPhase():
    res = requests.get('/'.join([url, 'info', connect['token']]))
    return json.loads(res.text)


response = requests.get('/'.join([url, 'connect', botName]))  # Ask the server a token to play the game
connect = json.loads(response.text)  # Get the answer as a json to read it easily
print('Connect Phase: ', connect)
time.sleep(max(0, connect['wait'] / 1000))  # Wait the right time to ask for information

info = infoPhase()
print('Info Phase: ', info)
time.sleep(max(0, info['wait'] / 1000))  # Wait the right time to start making your move

running = True
turn = 1
while running:
    result = moveBot(0, turn)  # Move the bot and store the result of the request in a variable
    running = result['success']
    print('turn: ', turn)
    turn += 1
    time.sleep(max(0, result['wait'] / 1000))  # Wait until the next turn
