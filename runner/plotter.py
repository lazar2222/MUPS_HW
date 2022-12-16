import sys
import copy
import string
import matplotlib
import matplotlib.pyplot as plt

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load(fname):
    path = '../' + fname + '.txt'
    lines = open(path).readlines()
    task = ''
    variant = ''
    testcase = ''
    threads = ''
    res = []
    for line in lines:
        if line.startswith('Running task:'):
            task = line.strip().split(' ')[2]
        elif line.startswith('Running variant:'):
            variant = line.strip().split(' ')[2]
        elif line.startswith('Running test case:'):
            testcase = line[line.index('(') + 1:line.index(')')]
            threads = line.strip().split(' ')[-3][2:]
        elif line.startswith('    AVG ') and 'Speedup' in line:
            time = float(line.strip().split(' ')[2])
            speedup = float(line.strip().split(' ')[5])
            status = line.strip().split(' ')[8]
            res.append([task, variant, testcase,threads, time, speedup, status])
    [print(line) for line in res]
    return res

def makeSeries(table, series):
    res = []
    for serie in series:
        res.append([ent[5] for ent in table if ent[2] == serie])
    return res

def makeCompSeries(table, series, tc,x):
    res = []
    for i in range(len(series)):
        res.append([])
        for var in x: 
            res[i].append([ent[5] for ent in table if ent[3] == series[i] and ent[2] == tc and ent[1][ent[1].index('_') + 2:-2].lstrip(string.digits) == var])
    return res

def normalize(x, ys):
    res = copy.deepcopy(ys)
    for i in range(len(ys)):
        for j in range(len(x)):
            res[i][j] /= int(x[j])
    return res

def simplePlot(x, series, ys, name,xlabel='Broj niti',title = '', pgf = False):
    if pgf:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    fig, ax = plt.subplots()
    ax.grid(which='major',axis='both', linestyle='-', color='k')
    ax.grid(which='minor',axis='y', linestyle='--')
    ax.minorticks_on()
    for i in range(len(series)):
        ax.plot(x, ys[i], label=series[i])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Ubrzanje')
    ax.legend()
    if pgf:
        plt.savefig(f'{name}.pgf')
    else:
        plt.show()

def main(args):
    table = load(args[1])
    if('_' not in args[1]):
        #default plot
        x = list(dict.fromkeys([ent[3] for ent in table]))
        series = list(dict.fromkeys([ent[2] for ent in table]))
        ys = makeSeries(table, series)

        print(x)
        print(series)
        [print(line) for line in ys]

        simplePlot(x, series, ys, args[1])
    else:
        #comparison plot
        x = list(dict.fromkeys([ent[1][ent[1].index('_') + 2:-2].lstrip(string.digits) for ent in table if ent[1].count('_') > 1]))
        x.sort(key=natural_keys)
        series = list(dict.fromkeys([ent[3] for ent in table]))
        tc = list(dict.fromkeys([ent[2] for ent in table]))[-1]
        ys = makeCompSeries(table, series, tc, x)

        print(x)
        print(series)
        print(tc)
        [print(line) for line in ys]

        simplePlot(x, series, ys, args[1],'Varijacija',tc)


if __name__ == '__main__':
    main(sys.argv)