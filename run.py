import os
import subprocess
import sys
import time

threadSettings = [1,2,4,8,16] if 'threads' in sys.argv else [16]
numReps = 3 if 'reps' in sys.argv else 1

refResult = dict()
refTime = dict()

def listTasks():
    tasks = [name for name in os.listdir('.') if os.path.isdir(name)]
    tasks.remove('runner')
    tasks.remove('.git')
    tasks.sort()
    return tasks

def listVariants(task):
    variants = [os.path.splitext(name)[0] for name in os.listdir(task) if (name.endswith('.c') and ('_v' in name))]
    variants.sort()
    return variants

def listCases(task):
    return [text.strip() for text in open(task + '/run').readlines() if not text.startswith('#')]

def buildVariant(task, variant):
    subprocess.run(f'(cd {task} && make {variant})', shell = True)

def runVariant(task, variant, cases, comparers):
    caseNo = 1;
    for case in cases:
        if('_v0' in variant):
            print('Running test case:', caseNo, 'of', len(cases));
            atime = 0
            for i in range(numReps):
                if 'noclean' in sys.argv and os.path.exists(f'{task}/{variant}_{caseNo}.txt'):
                    pass
                else:
                    subprocess.run(f'(cd {task} && ./{variant} {case} > {variant}_{caseNo}.txt)', shell = True)
                time = extractTime(task, variant, caseNo)
                result = extractResult(task, variant, caseNo)
                refTime[str(caseNo)] = time
                atime += time
                refResult[str(caseNo)] = result
                print('Time:', time)
            if numReps > 1:
                print('AVG Time:', "%.4f" % (atime/numReps))
                refTime[str(caseNo)] = atime/numReps
        else:
            for threadset in threadSettings:
                print('Running test case:', caseNo, 'of', len(cases), 'N=', threadset);
                atime = 0
                for i in range(numReps):
                    if 'noclean' in sys.argv and os.path.exists(f'{task}/{variant}_{caseNo}_N{threadset}.txt'):
                        pass
                    else:
                        subprocess.run(f'(cd {task} && OMP_NUM_THREADS={threadset} ./{variant} {case} > {variant}_{caseNo}_N{threadset}.txt)', shell = True)
                    time = extractTime(task, variant, caseNo, threadset)
                    atime += time
                    result = extractResult(task, variant, caseNo, threadset)
                    speedup = refTime[str(caseNo)]/time
                    result = compareResults(task, result, refResult[str(caseNo)], comparers)
                    print('Time:', time, 'Speedup:', "%.2f" % speedup, 'Test ', 'PASSED' if result else 'FAILED')
                if numReps > 1:
                    aspeedup = refTime[str(caseNo)]/(atime/numReps)
                    print('AVG Time:',"%.4f" % (atime/numReps), 'AVG Speedup:', "%.2f" % aspeedup)
        caseNo = caseNo + 1

def extractTime(task, variant, caseNo, threads = None):
    if threads == None:
        lines = open(f'{task}/{variant}_{caseNo}.txt').readlines()
    else:
        lines = open(f'{task}/{variant}_{caseNo}_N{threads}.txt').readlines()            
    if lines[-1].startswith('Time:'):
        return float(lines[-1].split(' ')[1])
    else:
        return None

def extractResult(task, variant, caseNo, threads = None):
    if threads == None:
        lines = open(f'{task}/{variant}_{caseNo}.txt').readlines()
    else:
        lines = open(f'{task}/{variant}_{caseNo}_N{threads}.txt').readlines()   
    return lines

def compareResults(task, res1, res2, comparers):
    return comparers[task](res1,res2)

def comparePrime(res1,res2):
    res1 = res1[8:-6]
    res2 = res2[8:-6]
    for i in range(len(res1)):
        spl1 = list(filter(None,res1[i].split(' ')))
        spl2 = list(filter(None,res2[i].split(' ')))
        if(spl1[0] != spl2[0] or spl1[0] != spl2[0]):
            return False
    return True

def compareFeyman(res1,res2):
    fr1 = float(res1[8].split(' ')[6])
    fr2 = float(res2[8].split(' ')[6])
    return abs(fr1-fr2) < 0.01

def compareMolDyn(res1,res2):
    return res1[:-2] == res2[:-2]

comparers = {'dz1z1': comparePrime, 'dz1z2':comparePrime, 'dz1z3':compareFeyman, 'dz1z4':compareFeyman, 'dz1z5':compareMolDyn}

start = time.time()

if('clean' in sys.argv):
    tasks = listTasks()
    for task in tasks:
        print('Cleaning task:', task)
        subprocess.run(f'(cd {task} && make clean)', shell = True)
        subprocess.run(f'(cd {task} && rm -f *.txt)', shell = True)
        variants = listVariants(task)

        for variant in variants:
            print('Cleaning variant:', variant);
            subprocess.run(f'(cd {task} && rm -f {variant})', shell = True)
else:
    tasks = listTasks()
    for task in tasks:
        print('Running task:', task)

        variants = listVariants(task)
        cases = listCases(task)
        refResult = dict()
        refTime = dict()

        for variant in variants:
            print('Building variant:', variant);
            buildVariant(task, variant)
            print('Running variant:', variant);
            runVariant(task, variant, cases,comparers)

end = time.time()

print('Total time taken:', end-start)