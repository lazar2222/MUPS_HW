def comparePrime(res1,res2):
    res1 = res1[8:-6]
    res2 = res2[8:-6]
    for i in range(len(res1)):
        spl1 = list(filter(None,res1[i].split(' ')))
        spl2 = list(filter(None,res2[i].split(' ')))
        if(spl1[0] != spl2[0] or spl1[1] != spl2[1]):
            return False
    return True

def compareFeyman(res1,res2):
    fr1 = float(res1[8].split(' ')[6])
    fr2 = float(res2[8].split(' ')[6])
    return abs(fr1-fr2) < 0.01

def compareMolDyn(res1,res2):
    res1 = res1[-6:-2]
    res2 = res2[-6:-2]
    for i in range(len(res1)):
        r1 = res1[i].split(' ')
        r2 = res2[i].split(' ')
        for j in range(len(r1)):
            if(r1[j] != r2[j]):
                fr1 = float(r1[j])
                fr2 = float(r2[j])
                if(abs(fr1-fr2) > 0.01):
                    return False
    return True

comparers = {'dz4z1': comparePrime, 'dz4z2':compareFeyman, 'dz4z3':compareMolDyn}