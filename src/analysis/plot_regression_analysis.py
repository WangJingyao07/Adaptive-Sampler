import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def load_results(f1, f2):
    data1 = np.load(f1, allow_pickle=True)
    data2 = np.load(f2, allow_pickle=True)

    train_data = np.array(data1.item().get('train'))
    test_data = np.array(data1.item().get('test'))

    train_data_other = np.array(data2.item().get('train'))
    test_data_other = np.array(data2.item().get('test'))

    comparison = train_data_other == train_data
    equal_arrays = comparison.all()
    assert equal_arrays

    return train_data, test_data, test_data_other


def get_task_sampler(choice):
    task_sampler = {0: 'uniform',
                    1: 'no_diversity_task',
                    2: 'no_diversity_episode',
                    3: 'no_diversity_tasks_per_episode',
                    4: 'ohtm',
                    5: 'owhtm',
                    6: 's-DPP',
                    7: 'd-DPP',
                    8: 'single_batch_fixed_pool',
                    9: 'single_batch_increased_data',
                    10: 'adaptive_sampler'}
    return task_sampler[choice]


def plot(train, test1, test2, sampler, output_folder):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(test1[:, 0], test1[:, 1], '-r')
    ax.plot(test1[:, 0], test1[:, -1], '--g')
    ax.plot(test2[:, 0], test2[:, -1], '--b')
    plt.scatter(train[:, 0], train[:, 1], marker='^', c='#3c1361', s=(20 * 0.5)**2)
    fig.savefig(f'{output_folder}_{sampler}.pdf', bbox_inches='tight')
    plt.close(fig)


def main(args):
    '''
    Path to filename should be of the directory structure <PATH>/MAML/*.npy or <PATH>/Reptile/*.npy
    '''
    def _parse_filename(filename):
        basename = os.path.basename(filename)
        name, _ = os.path.splitext(basename)
        prefix, sampler_id = name.rsplit('_', 1)
        return (filename, prefix, get_task_sampler(int(sampler_id)))

    filenames_maml = [_parse_filename(filename) for filename
                      in glob.glob(os.path.join(args.folder, "MAML", '*.npy'))]
    filenames_reptile = [_parse_filename(filename) for filename
                         in glob.glob(os.path.join(args.folder, "Reptile", '*.npy'))]

    for s in ["uniform", "no_diversity_task", "no_diversity_episode", "no_diversity_tasks_per_episode", "ohtm", "owhtm"]:
        f1 = [filename for (filename, _, sampler) in filenames_maml
              if sampler == s]
        f2 = [filename for (filename, _, sampler) in filenames_reptile
              if sampler == s]
        train, test1, test2 = load_results(f1[0], f2[0])
        plot(train, test1, test2, s, args.output)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Plot results for regression experiments')
    parser.add_argument('folder', type=str,
                        help='folder containing the results for a specific algorithm/dataset.')
    parser.add_argument('-O', '--output', type=str, default=None,
                        help='output filename to save results (optional)')

    args = parser.parse_args()
    main(args)
