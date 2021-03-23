import pandas as pd
import PySimpleGUI as sg
import os
from PIL import Image, ImageTk
import tkinter
import numpy as np

# env_id = 'cloth_v0'
env_id = 'cloth_sewts_exp1'
path_to_csv = os.getcwd() + '/output/' + env_id + "/log.csv"
df = pd.read_csv(path_to_csv)
img_path = []
gameno = []
stepno = []
state = []
action = []
reward = []
print(df)
total = df.shape[0]
for i in range(total):
    img_path.append(df.Image.iloc[i])
    gameno.append(df.Game_no.iloc[i])
    stepno.append(df.Step_no.iloc[i])
    state.append(df.State.iloc[i])
    action.append(df.Action.iloc[i])
    reward.append(df.Reward.iloc[i])

# print(img_path[0])
# img = Image.open(img_path)
# print(img)
# img = img.thumbnail((400, 400))
# img.show()
# image.thumbnail((400, 400))
# root = tkinter.Tk()
# photo_img = ImageTk.PhotoImage(img)
# l = tkinter.Label(root, image = photo_img)
# l.pack()
# print(l)
# print(photo_img)
i = 0
while i<total:
    layout = [  [sg.Text("GAME NO.",font=("Arial", 15)), sg.Text(gameno[i],font=("Arial", 15)),sg.Text("STEP NO.",font=("Arial", 15)) , sg.Text(stepno[i],font=("Arial", 15))],
                [sg.Text("STATE",font=("Arial", 15)),sg.Image(img_path[i]),sg.Text(state[i],font=("Arial", 15))],
                [sg.Text("ACTION",font=("Arial", 15)) , sg.Text(action[i],font=("Arial", 15)), sg.Text("REWARD",font=("Arial", 15)), 
                sg.Text(reward[i],font=("Arial", 15))],
                [sg.Button('NEXT',font=("Arial", 15))]
                ]
    window = sg.Window('Window Title', layout)      # Part 3 - Window Defintion
    event, values = window.read()
    if event == 'NEXT':
        i = i + 1
        window.close()
