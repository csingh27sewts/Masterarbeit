# import gc
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    scores = np.array(scores)
    print("Scores")
    print(scores)
    running_avg = np.zeros(len(scores))
    print(running_avg)
    matplotlib.use('agg')
    plt.close()
    plt.ioff()
    for i in range(running_avg.size):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    fig = plt.figure()
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    fig.savefig(figure_file)
    plt.close('all')
    # gc.collect()
