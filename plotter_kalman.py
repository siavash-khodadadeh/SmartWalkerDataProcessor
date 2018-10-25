import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal

from kalman import kalman


font = {'family': 'normal',
        'weight': 'bold',
        'size': 16}

matplotlib.rc('font', **font)


def get_smooth_data(data):
    return kalman(data, var_estimate=0.01 ** 2)


def cal_walk_loss(leg_data, steps, target_position, leg):
    previous_step = steps[0]
    total_loss = 0
    counter = 0
    for step in steps:
        counter += 1
        step_data = leg_data[previous_step:step - 1]
        if len(step_data) > 0:
            loss = cal_loss(step_data, target_position[leg])
            total_loss += loss
        previous_step = step

    print('loss for leg {}: {}'.format(leg, total_loss / counter))


def cal_loss(step_data, target):
    min_step_data = np.min(step_data)
    return abs(min_step_data - target)


def get_steps(data):
    peakind = signal.find_peaks_cwt(data, np.arange(1, 20), max_distances=np.array(range(20)))
    return sorted(peakind)

tmp_address = 'logs/tmp.txt'


def preprocess(data):
    data[np.where(data < -1000)] = -1000
    data[np.where(data > 1000)] = 1000


def plotter():
    # data: fr, fl, rr, rl
    # fr 600
    # fl 400
    # rr 450
    # rl 300
    target_position = {0: -600, 1: -400, 2: -450, 3: -300}

    legs = [1, 0, 3, 2]

    for address in addresses:
        print(address)
        with open(address) as f:
            f.readline()
            tmp_file = open(tmp_address, 'w')
            for line in f:
                line = line.replace(',', ' ')
                i = line.index(' ', line.index(' ') + 1) + 1
                tmp_file.write(line[i:])

            tmp_file.close()

        data = np.loadtxt(tmp_address)
        preprocess(data)
        fig = plt.figure()

        leg = 1
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Time', fontsize=22)
        ax.set_ylabel('Pressure', fontsize=22)
        ax.set_ylim(-1000, 1000)
        leg_data = data[:, leg]
        ax.plot(leg_data, linestyle='--', color='gray', label='Data')

        smoothed_data = get_smooth_data(leg_data)
        ax.plot(smoothed_data, linewidth=2, color='black', label='Smoothed Data')
        ax.axhline(y=[target_position[leg]], linewidth=3, color='black', label='Target', linestyle='-.')

        steps = get_steps(smoothed_data)

        # for step in steps:
        #     ax.axvline(x=step, color='purple')

        cal_walk_loss(smoothed_data, steps, target_position, leg)
        ax.legend(loc='upper left', shadow=True)
        plt.show()


if __name__ == '__main__':
    # addresses = ['logs/Sharare_Normal.txt', 'logs/Sharare_Blind.txt']
    addresses = ['logs/Siavash_Day2_Normal.txt']
    # addresses = ['logs/Rouhollah_Normal.txt', 'logs/Rouhollah_Blind.txt']
    # addresses = ['logs/Pooya_Normal2.txt', 'logs/Pooya_Blind.txt']
    # addresses = ['logs/Safa_Normal.txt', 'logs/Safa_Blind.txt']
    # addresses = ['logs/Hassam_Normal.txt', 'logs/Hassam_Blind.txt']
    # addresses = ['logs/Jun_Normal.txt', 'logs/Jun_Blind.txt']

    plotter()
