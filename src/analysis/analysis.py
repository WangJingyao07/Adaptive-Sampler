import numpy as np
import json
import os
import math
import glob

from scipy.stats import t, ttest_rel
from collections import namedtuple
# from main import get_task_sampler


Statistics = namedtuple('Statistics', ['mean', 'ci95'])


def paired_difference_ttest(data_uniform, data_sampler, **kwargs):
    """Compute the statistical significance of the difference in mean of
    two populations. If the p-value is lower than a certain threshold (e.g 0.05),
    then the null hypothesis (here, that both populations have the same mean)
    is rejected, showing the statistical significance of the sampler compared
    to uniform sampling.
    """
    sorted_data_uniform = sorted(data_uniform, key=lambda x: x[0])
    tasks_uniform = [task for (task, _) in sorted_data_uniform]

    sorted_data_sampler = sorted(data_sampler, key=lambda x: x[0])
    tasks_sampler = [task for (task, _) in sorted_data_sampler]

    # Make sure that the tasks are the same in both populations
    if tasks_uniform != tasks_sampler:
        raise ValueError('The tasks do not match between the results of the '
                         'uniform sampler and the sampler to be compared against.')

    uniform = np.asarray([value for (_, value) in sorted_data_uniform])
    sampler = np.asarray([value for (_, value) in sorted_data_sampler])

    return ttest_rel(uniform, sampler, **kwargs)


def mean_and_confidence_interval(data):
    """Compute the mean and 95% confidence interval."""
    array = np.asarray([value for (_, value) in data])
    t_95_sqrt_n = t.ppf(1 - 0.05 / 2, df=array.size - 1) / math.sqrt(array.size)
    return Statistics(mean=np.mean(array), ci95=t_95_sqrt_n * np.std(array))


def to_latex(
    stats,
    ttest_result=None,
    threshold=0.05,
    positive_fmt='{value:s} +',  # TODO: Have different formats when the results
    negative_fmt='{value:s} -'   # of sampler are better or worse than uniform


):
    """Format the results to LaTeX. This uses the result of the paired difference
    t-test to assess the statistical significance of the result (i.e. the results
    for the sampler are significantly different from uniform). It then uses
    different string format if the results of the sampler are better (positive_fmt)
    or worse (negative_fmt) than uniform.
    """
    text = rf'{stats.mean * 100:.2f} \pm {stats.ci95 * 100:.2f} %'
    if (ttest_result is not None) and (ttest_result.pvalue < threshold):
        # The result is statistically significant
        fmt = positive_fmt if (ttest_result.statistic < 0) else negative_fmt
        text = fmt.format(value=text)
    return text


def load_results(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        # The data might have been saved as an "odict_items" (iterator returned
        # by the .items() method of OrderedDict)
        if isinstance(data, str):
            data = eval(data[12:-1])  # Remove "odict_items()"
            data = [(eval(task), value) for (task, value) in data
                    if task[0] == '[' and task[-1] == ']']
        else:
            raise IOError(f'Unknown data format: {filename}')

    # Filter out elements that do not correspond to a task
    data = [(tuple(task), value) for (task, value) in data
            if isinstance(task, (list, tuple))]

    print(filename, len(data))
    return data


# TODO: Use the function from main.py
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


def main(args):
    def _parse_filename(filename):
        basename = os.path.basename(filename)
        name, _ = os.path.splitext(basename)
        prefix, sampler_id = name.rsplit('_', 1)
        return (filename, prefix, get_task_sampler(int(sampler_id)))

    filenames = [_parse_filename(filename) for filename
                 in glob.glob(os.path.join(args.folder, '*.json'))]

    # Load the results for the uniform sampler
    uniform = [filename for (filename, _, sampler) in filenames
               if sampler == 'uniform']
    if len(uniform) == 0:
        raise IOError(f'There is no result for uniform sampler: {args.folder}')
    if len(uniform) > 1:
        raise IOError(f'There are multiple results for uniform sampler: {args.folder}')
    data_uniform = load_results(uniform[0])

    results = {}
    for filename, _, sampler in filenames:
        data_sampler = load_results(filename)
        stats = mean_and_confidence_interval(data_sampler)
        if sampler != 'uniform':
            ttest_result = paired_difference_ttest(data_uniform, data_sampler)
        else:
            ttest_result = None

        results[sampler] = {
            'mean': stats.mean,
            'ci95': stats.ci95,
            'latex': to_latex(stats, ttest_result)
        }

        if ttest_result is not None:
            results[sampler].update({
                'pvalue': ttest_result.pvalue,
                'statistic': ttest_result.statistic
            })

    if args.output is not None:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Analysis of task diversity')
    parser.add_argument('folder', type=str,
                        help='folder containing the results for a specific algorithm/dataset.')
    parser.add_argument('-O', '--output', type=str, default=None,
                        help='output filename to save results (optional)')

    args = parser.parse_args()
    main(args)
