import sys
import copy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


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

def normalize(x, ys):
    res = copy.deepcopy(ys)
    for i in range(len(ys)):
        for j in range(len(x)):
            res[i][j] /= int(x[j])
    return res

def simplePlot(x, series, ys, name):
    fig, ax = plt.subplots()
    ax.grid(which='major',axis='both', linestyle='-', color='k')
    ax.grid(which='minor',axis='y', linestyle='--')
    ax.minorticks_on()
    for i in range(len(series)):
        ax.plot(x, ys[i], label=series[i])
    ax.set_xlabel('Broj niti')
    ax.set_ylabel('Ubrzanje')
    ax.legend()
    plt.savefig(f'{name}.pgf')

def main(args):
    table = load(args[1])
    x = list(dict.fromkeys([ent[3] for ent in table]))
    series = list(dict.fromkeys([ent[2] for ent in table]))
    ys = makeSeries(table, series)
    #nys = normalize(x, ys)

    #print(x)
    #print(series)
    #[print(line) for line in ys]
    #[print(line) for line in nys]

    simplePlot(x, series, ys, args[1])
    #simplePlot(x, series, nys, table[0][0] + ' ' + table[0][1], 'Num threads', 'relative speedup')


if __name__ == '__main__':
    main(sys.argv)