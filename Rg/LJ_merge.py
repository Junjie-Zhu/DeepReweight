from numpy import average


def get_LJ(file_name):
    file = open(file_name,"r")
    LJ = []
    file.readline()
    for line in file.readlines():
        data = line.split()
        LJ.append(float(data[6]))
    file.close()
    return LJ
    
    
def get_vdw(file_name):
    file = open(file_name,"r")
    LJ = []
    i = 1
    for line in file.readlines():
        if i == 1:
            data = line.split()
            LJ.append(float(data[-1]))
            i += 1
        elif i == 5:
            i = 1
        else:
            i += 1
    
    file.close()
    return LJ
    
def calc_inter(LJ_VDW,LJ_P,LJ_W):
    LJ_inter = []
    for i in range(0,len(LJ_VDW)):
        tmp = LJ_VDW[i] - LJ_P[i] - LJ_W[i]
        LJ_inter.append(tmp)
        
    return LJ_inter

def get_rg(file_name):
    file = open(file_name,"r")
    rg = []
    file.readline()
    for line in file.readlines():
        data = line.split()
        rg.append(float(data[-1]))
    
    file.close()
    
    return rg

def write_merge(file_name,LJ_inter,rg):
    file = open(file_name,"w")
    file.write("frame\tLJ_inter\trg\n")
    for i in range(0,len(rg)):
        line = str(i) + '\t' + str(LJ_inter[i]) + '\t' + str(rg[i]) + '\n'
        file.write(line)
    file.close()
    return
    
if __name__ == "__main__":
    LJ_P = get_LJ("./LJ_P.dat")
    LJ_W = get_LJ("./LJ_W.dat")
    LJ_VDW = get_vdw("./LJ_VDW.dat")
    LJ_inter = calc_inter(LJ_VDW=LJ_VDW,LJ_P=LJ_P,LJ_W=LJ_W)
    print("done calc")
    rg = get_rg("./Rg_Abeta40.dat")
    print(average(rg))
    write_merge(file_name="./LJ_rg.dat",LJ_inter=LJ_inter,rg=rg)
