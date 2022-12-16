import os
import subprocess
import sys
import time
import runner.comparers


def parseArgs(args):
    res = {'none_list': []}
    lastkey = 'none_list'
    for par in args:
        if par.startswith('-'):
            res[par[1:]] = 1
            lastkey = par[1:]+'_list'
            if lastkey not in res:
                res[lastkey] = []
        else:
            res[lastkey].append(par)
    return res


def listTasks(argDict):
    tasks = [name for name in os.listdir('.') if os.path.isdir(name) and not name.startswith('.')]
    tasks.remove('runner')
    tasks.sort()
    if 't_list' in argDict:
        tasks = [task for task in tasks if task in argDict['t_list']]
    return tasks


def isValidSet(name, argDict):
    if not (name.endswith('.c') and ('_v' in name)):
        return False
    if '_v0' in name:
        return False
    set = -1
    if 'test_list' in argDict:
        set = 0
        if len(argDict['test_list']) > 0:
            set = argDict['test_list'][0]
    if name[:-3].endswith('_t'):
        return name.endswith(f'_t{set}.c')
    else:
        return set == -1


def listVariants(task, argDict):
    variants = [os.path.splitext(name)[0] for name in os.listdir(task) if isValidSet(name, argDict)]
    variants.sort()
    if 'v_list' in argDict:
        variants = [variant for variant in variants if variant in argDict['v_list']]
    return variants


def getGoldVariant(task):
    variants = [os.path.splitext(name)[0] for name in os.listdir(task) if name.endswith('.c') and ('_v0' in name)]
    return variants[0]


def listCases(task, argDict):
    cases = [text.strip() for text in open(task + '/run').readlines() if not text.startswith('#')]
    if 'c_list' in argDict:
        cases = cases[:int(argDict['c_list'][0])]
    return cases


def listThreads(argDict):
    threads = [2, 4]
    if 'h_list' in argDict:
        threads = [thread for thread in threads if str(thread) in argDict['h_list']]
    return threads


def getRepetitions(argDict):
    repetitions = 1
    if 'r_list' in argDict:
        repetitions = int(argDict['r_list'][0])
    return repetitions


def buildVariant(task, variant):
    print('Building variant:', variant)
    subprocess.run(f'mpicc -O3 -o {task}/{"gold" if "_v0" in variant else "out"}/{variant} {task}/{variant}.c -lm', shell=True)


def compareResults(task, ref, res):
    return runner.comparers.comparers[task](ref, res)


def printResults(task, variant, caseNo, threads, repetitions, verbose):
    goldVariant = getGoldVariant(task)
    reftime = 0
    reflines = ''
    for i in range(repetitions):
        reflines = open(f'{task}/gold/{goldVariant}_{caseNo}_N{1}_R{i}.txt').readlines()
        reftime += float(reflines[-1].split(' ')[1])
    reftime /= repetitions
    totaltime = 0
    passed = True
    for i in range(repetitions):
        lines = open(f'{task}/{"gold" if "_v0" in variant else "out"}/{variant}_{caseNo}_N{threads}_R{i}.txt').readlines()
        time = float(lines[-1].split(' ')[1])
        result = compareResults(task, reflines, lines)
        totaltime += time
        passed = passed and result
        speedup = reftime/time
        if verbose or repetitions == 1:
            if '_v0' in variant:
                print('        Time:', '%.4f' % time)
            else:
                print('        Time:', '%.4f' % time, '    Speedup:', '%.2f' % speedup, '    Test ', 'PASSED' if result else 'FAILED')
    totaltime /= repetitions
    if repetitions > 1:
        if '_v0' in variant:
            print('    AVG Time:', '%.4f' % totaltime)
        else:
            print('    AVG Time:', '%.4f' % totaltime, 'AVG Speedup:', '%.2f' % (reftime/totaltime), 'ALL Tests', 'PASSED' if result else 'FAILED')


def runTestCase(task, variant, case, caseNo, totalCases, threads, repetitions, skip, doPrint, verbose):
    print('Running test case:', caseNo, 'of', totalCases, f'({case})', f'N={threads}', repetitions, 'times')
    hasRun = False
    for i in range(repetitions):
        if os.path.exists(f'{task}/{"gold" if "_v0" in variant else "out"}/{variant}_{caseNo}_N{threads}_R{i}.txt') and skip:
            pass
        else:
            subprocess.run(f'(cd {task}/{"gold" if "_v0" in variant else "out"} && mpirun -np {threads} ./{variant} {case} > {variant}_{caseNo}_N{threads}_R{i}.txt)', shell=True)
            hasRun = True
    if hasRun or doPrint:
        printResults(task, variant, caseNo, threads, repetitions, verbose)


def runVariant(task, variant, cases, threads, repetitions, skip, doPrint, verbose):
    print('Running variant:', variant)
    for i in range(len(cases)):
        for thread in threads:
            runTestCase(task, variant, cases[i], i + 1, len(cases), thread, repetitions, skip, doPrint, verbose)


def runTask(task, threads, repetitions, skip, doPrint, verbose, argDict):
    print('Running task:', task)

    variants = listVariants(task, argDict)
    cases = listCases(task, argDict)

    subprocess.run(f'mkdir -p {task}/gold', shell=True)
    subprocess.run(f'mkdir -p {task}/out', shell=True)

    buildVariant(task, getGoldVariant(task))
    for variant in variants:
        buildVariant(task, variant)

    runVariant(task, getGoldVariant(task), cases, [1], repetitions, True, doPrint, verbose)
    for variant in variants:
        runVariant(task, variant, cases, threads, repetitions, skip, doPrint, verbose)


def run(argDict):
    tasks = listTasks(argDict)
    threads = listThreads(argDict)
    repetitions = getRepetitions(argDict)
    skip = 'skip' in argDict
    doPrint = 'print' in argDict
    verbose = 'verbose' in argDict

    for task in tasks:
        runTask(task, threads, repetitions, skip, doPrint, verbose, argDict)


def clean(argDict):
    tasks = listTasks(argDict)
    for task in tasks:
        if 'out' in argDict['clean_list']:
            print('Cleaning out  for task:', task)
            subprocess.run(f'rm -rf {task}/out', shell=True)
        if 'gold' in argDict['clean_list']:
            print('Cleaning gold for task:', task)
            subprocess.run(f'rm -rf {task}/gold', shell=True)
    if 'logs' in argDict['clean_list']:
        print('Cleaning logs')
        subprocess.run(f'rm -rf runner/logs', shell=True)
    if 'plots' in argDict['clean_list']:
        print('Cleaning plots')
        subprocess.run(f'rm -rf runner/plots', shell=True)


def printHelp():
    print('Arguments:')
    print('-help                              ', 'show this help', sep='\t')
    print('-clean [out] [gold] [logs] [plots] ', 'clean generated files', sep='\t')
    print('-skip                              ', 'dont re-run tests with existing logs', sep='\t')
    print('-print                             ', 'print all tests, even skipped ones', sep='\t')
    print('-verbose                           ', 'print detailed test results', sep='\t')
    print('-test [set]                        ', 'run test variations', sep='\t')
    print('-t <task list>                     ')
    print('-v <variant list>                  ')
    print('-c <number of test cases>          ')
    print('-h <threads to test for>           ')
    print('-r <number of repetitions>         ')


def main(args):
    start_time = time.time()
    argDict = parseArgs(args)

    if 'help' in argDict:
        printHelp()
    elif 'clean' in argDict:
        clean(argDict)
    else:
        run(argDict)

    end_time = time.time()
    print('Total time taken:', '%.4f' % (end_time - start_time))


if __name__ == '__main__':
    main(sys.argv)
