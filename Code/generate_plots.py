import pandas as pd
import PySimpleGUI as sg
import os
from PIL import Image, ImageTk
import tkinter
import numpy as np
import subprocess
import re
import ast

# env_id = 'cloth_v0'
env_id = 'cloth_sewts_exp2_2'
n_games = 1000
path_to_csv = os.getcwd() + '/output/' + env_id + "/log_sac.csv"
path_to_output = os.getcwd() + '/output/'
df = pd.read_csv(path_to_csv)
value = []
target_value = []
critic = []
entropy = []
print(df)
total = df.shape[0]
plot_path = []
batchno = []
elementno = []
state = []
next_state = []
action = []
sampled_action = []
reward = []

#def str2array(s):
    # Remove space after [
#    s=re.sub('[', '', s.strip())
#    s=re.sub(']', '', s.strip())
    # s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    #s=re.sub('[,\s]+', ', ', s)
#    return np.array(ast.literal_eval(s))


# for i in range(total):
    # print(df.State.iloc[i])
#    for j in range(len(list(df.State.iloc[i]))):
#        for k in range(len(list(df.State.iloc[i][j]))):
#            print(df.State.iloc[j][k])
#            print("ELEMENT")
            # state[j][k].append(df.State.iloc[j][k])
#df.State.apply(str2array)
#df.Next_State.apply(str2array)
#df.Action.apply(str2array)
#df.Sampled_Action.apply(str2array)
#df.Reward.apply(str2array)
# df.Value.apply(str2array)
#df.Target_Value.apply(str2array)
#df.Critic.apply(str2array)
#df.Entropy.apply(str2array)

# print(state[total][][1])

for i in range(total):
    state.append((df.State.iloc[i]))
    #next_state.append(df.Next_State.iloc[i])
    #action.append(df.Action.iloc[i])
    #sampled_action.append(df.Sampled_Action.iloc[i])
    #reward.append(df.Reward.iloc[i])
    #value.append(df.Value.iloc[i])
    #target_value.append(df.Target_Value.iloc[i])
    #critic.append(df.Critic.iloc[i])
    #entropy.append(df.Entropy.iloc[i])
i = 0
state[320]=state[320].replace("[","")
state[320]=state[320].replace("]","")
state[320]=state[320].replace("  "," ")
state[320]=state[320].replace("  "," ")
state[320]=state[320].replace(" ",",")
state[320]=np.array(state[320])
print(state[320])
#state[0] = np.fromstring(state[0])
#print(state[320][2])
# no_steps = max(elementno)
'''
for j in range(total):
    batch_size = len(state[j])
    print(batch_size)
    while i < batch_size:
        layout = [  [sg.Text("BATCH NO.",font=("Arial", 15)), sg.Text(j,font=("Arial", 10))],
                    [sg.Text("ELEMENT NO.",font=("Arial", 15)), sg.Text(i,font=("Arial", 10))],
                    [sg.Text("STATE",font=("Arial", 15)),sg.Text(state[j][i],font=("Arial", 10))],
                    #sg.Text("NEXT_STATE",font=("Arial", 15)),sg.Text(next_state[i],font=("Arial", 10))],
                    #[sg.Text("ACTION",font=("Arial", 15)),sg.Text(action[i],font=("Arial", 10)),
                    #sg.Text("SAMPLED_ACTION",font=("Arial", 15)),sg.Text((sampled_action[i]),font=("Arial", 10)),
                    #sg.Text("REWARD",font=("Arial", 15)),sg.Text(reward[i],font=("Arial", 10))],
                    #[sg.Text("VALUE",font=("Arial", 15)),sg.Text(value[i],font=("Arial", 10)),
                    #sg.Text("TARGET_VALUE",font=("Arial", 15)),sg.Text(target_value[i],font=("Arial", 10))],
                    #[sg.Text("CRITIC",font=("Arial", 15)),sg.Text(critic[i],font=("Arial", 10))],
                    #[sg.Text("ENTROPY",font=("Arial", 15)),sg.Text(entropy[i],font=("Arial", 10))],
                    [sg.Button('NEXT',font=("Arial", 15)),
                    # sg.Combo(i, size=(10, 5), enable_events=False, key = 'SELECT_BATCH'),
                    sg.Button('GOTO',font=("Arial", 15))]] 
                
        window = sg.Window('Window Title', layout, size = (1200, 1000))      # Part 3 - Window Defintion
        event, values = window.read()
        if event == 'NEXT':
            i = i + 1
        window.close()
        if event == 'GOTO':
        #    if event =='SELECT_BATCH':
        #        step = values['SELECT_BATCH']
            i = 256 
        #    print(i)
        window.close()    


for i in range(total):
    value.append(df.Value.iloc[i])
    target_value.append(df.Target_Value.iloc[i])
    critic.append(df.Critic.iloc[i])
    entropy.append(df.Entropy.iloc[i])

    # Define filenames to store network output plots
    os.chdir(path_to_output)
    path_to_plots = path_to_output + '/Plots/'
    print(path_to_plots)

    if not os.path.exists(path_to_plots):
        subprocess.call(["mkdir","-p", "Plots"])

       
    filename_critic = env_id + '_'+ str(n_games) + 'plot_critic.png'
    figure_file_critic = filename_critic
    figure_file_critic = os.path.join(path_to_plots, figure_file_critic)

    filename_target_value = env_id + '_'+ str(n_games) + 'plot_target_value.png'
    figure_file_target_value = filename_target_value
    figure_file_target_value = os.path.join(path_to_plots, figure_file_target_value)

    filename_entropy = env_id + '_'+ str(n_games) + 'plot_entropy.png'
    figure_file_entropy = filename_entropy
    figure_file_entropy = os.path.join(path_to_plots, figure_file_entropy)

 
    print("PLOTS\n")
    for j in range(len(value)):
        value_plot = []
        value_step = []     
        filename_value = env_id + '_'+ str(j) + 'plot_value.png'
        figure_file_value = filename_value
        figure_file_value = os.path.join(path_to_plots, figure_file_value)

        for i in range(len(value[j])):
            value_plot.append(np.fromstring(value[j], dtype = np.float32))
            value_step.append(i)
        plot_learning_curve(value_plot, value_step , figure_file_value)

    for j in range(len(critic)):
        critic_plot = []
        critic_step = []     
        filename_critic = env_id + '_'+ str(j) + 'plot_critic.png'
        figure_file_critic = filename_critic
        figure_file_critic = os.path.join(path_to_plots, figure_file_critic)

        for i in range(len(critic[j])):
            critic_plot.append(np.fromstring(critic[j],dtype = np.float32))
            critic_step.append(i)
        plot_learning_curve(critic_plot, critic_step , figure_file_critic)

    for j in range(len(value)):
        target_value_plot = []
        target_value_step = []     
        filename_target_value = env_id + '_'+ str(j) + 'plot_value.png'
        figure_file_target_value = filename_target_value
        figure_file_target_value = os.path.join(path_to_plots, figure_file_target_value)

        for i in range(len(target_value[j])):
            target_value_plot.append(np.fromstring(target_value[j], dtype = np.float32))
            target_value_step.append(i)
        plot_learning_curve(target_value_plot, target_value_step , figure_file_target_value)

    for j in range(len(entropy)):
        entropy_plot = []
        entropy_step = []     
        filename_entropy = env_id + '_'+ str(j) + 'plot_entropy.png'
        figure_file_entropy = filename_entropy
        figure_file_entropy = os.path.join(path_to_plots, figure_file_entropy)

        for i in range(len(value[j])):
            entropy_plot.append(np.fromstring(entropy[j], dtype = np.float32))
            entropy_step.append(i)
        plot_learning_curve(entropy_plot, entropy_step , figure_file_entropy)
'''
