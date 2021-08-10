import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Plot Logs')
    parser.add_argument('-l', '--list', help='delimited list input', type=str)
    args = parser.parse_args()
    tests_list = [[it for it in item.split(': ')] for item in args.list.split(', ')]

    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    LABEL_SIZE = 'medium'
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.legend.html

    for idx in range(len(tests_list)):
        test_num = tests_list[idx][0]
        path = tests_list[idx][1]
        test_path = os.path.join(path + '/history_epoch_last.csv')
        # get number of parameters
        # net_sum_path = os.path.expanduser(os.getcwd() + '/ckpt/' + test_num + '/network_summary.txt')
        # for line in open(net_sum_path):
        #     para = re.findall(r"\[?\d+(?:[\d,.]*\d)+?\]", line)
        #     if para:
        #         '''
        #         # para as string
        #         print(para[0])
        #         # para only numbers
        #         print(re.findall('\d+', para[0]))
        #         # join numbers
        #         print(''.join(re.findall('\d+', para[0])))
        #         # omit last three
        #         print(''.join(re.findall('\d+', para[0]))[:-3])
        #         '''
        #         num_para_k = ''.join(re.findall('\d+', para[0]))[:-3]
        # print(test_path)
        df = pd.read_csv(test_path)

        epoch = df['epoch']
        # train_loss = df['train_loss']
        train_acc = df['train_acc']
        # val_iou = df['val_iou']
        # val_acc = df['val_acc']

        color_num = 'C' + str(idx)
        x_length = len(epoch)

        axs.plot(df.index, train_acc, label=test_num + '_train_acc', color=color_num, linestyle='--') # '_train_acc, para=[' + num_para_k + 'k]', color=color_num, linestyle='--')
        # axs[0][0].plot(df.index, val_acc, label=test_num + '_val_acc', color=color_num) # '_val_acc, para=[' + num_para_k + 'k]', color=color_num)
        axs.set_xlim([0, x_length])
        axs.set(xlabel='Epoch', ylabel='acc',
                   title='Train (Val) Acc Analysis')
        axs.legend(ncol=2, prop={"size":LABEL_SIZE})

        # axs[1][0].plot(df.index, val_iou, label=test_num + '_miou', color=color_num) # ', para=[' + num_para_k + 'k]', color=color_num)
        # axs[1][0].set_xlim([0, x_length])
        # axs[1][0].set(xlabel='Epoch', ylabel='val_miou',
                   # title='Val_miou Analysis')
        # axs[1][0].legend(prop={"size":LABEL_SIZE})


        # axs[0][1].plot(df.index, train_loss, label=test_num + '_train_loss', color=color_num) # ', para=[' + num_para_k + 'k]', color=color_num)
        # axs[0][1].set_xlim([0, x_length])
        # axs[0][1].set(xlabel='Epoch', ylabel='train_loss',
                   # title='train_loss Analysis')
        # axs[0][1].legend(prop={"size":LABEL_SIZE})

    # for i in range(len(axs)):
        # for j in range(len(axs[0])):
    axs.xaxis.label.set_size(10)
    axs.yaxis.label.set_size(10)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
