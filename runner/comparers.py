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

comparers = {'dz2z1': comparePrime, 'dz2z2':compareFeyman, 'dz2z3':compareMolDyn, 'dz2z4':compareMolDyn}