import math
import statistics

import numpy as np
def triangle(u1,u2):
    sum1=0;sum2=0;sum3=0;triangle=0;
    if (len(u1)!=len(u2)):
        print("Both vectors must be of same size");
    else:
        for x in range(0,len(u1)):
           #print(u1[x]);
           #print(u2[x]);
            sum1+=(np.power((u1[x]-u2[x]),2));
            sum2+=(np.power(u1[x],2));
            sum3+=(np.power(u2[x],2));
        sum1=np.sqrt(sum1);
        sum2=np.sqrt(sum2);
        sum3=np.sqrt(sum3);
        #print(sum1,sum2,sum3);
        triangle=1-(sum1/(sum2+sum3));

    return triangle
def urp(avgu1,avgu2,SDu1,SDu2):
    urp =abs(avgu1 - avgu2);
    temp2 =    abs(SDu1 - SDu2);
    temp4 = ( - 1 * (urp) * (temp2));
    temp3 = 1 + math.exp(temp4);
    urp = (1 - (1 / temp3));
    #print("  temp2 =" , temp2 , "  temp3 = " , temp3 , " urp = " , urp , " temp4 = " , temp4);
    return urp;
#print("Triangle similarity = "+ str(triangle));
def test() :
    u1 = [4, 5, 4, 2, 4];
    u2 = [3, 1, 2, 1, 2];
    trianglesim=triangle(u1,u2);
    avgu1=3.8;avgu2=1.8;SDu1=0.979;SDu2=0.748
    URP= urp(avgu1,avgu2,SDu1,SDu2)
    ITR=trianglesim*URP
    print( "URP = ",URP);
    print("Triangle similarity = "+ str(trianglesim)," URP = ",URP," ITR sim = ",ITR);
    #m=np.array([[1,5],[2,4],[3,1],[4,5]]);
    m=np.array([[1,5],[2,4]]);
    Mat=[[5.0, 4.0, 5.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0],
         [5.0, 5.0, 4.0, 4.0, 4.0, 2.0, 4.0, 5.0, 5.0, 3.0],
         [5.0, 5.0, 4.0, 3.0, 4.0, 4.0, 5.0, 5.0, 3.0, 4.0],
         [5.0, 2.0, 4.0, 2.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0],
         [5.0, 5.0, 5.0, 4.0, 2.0, 4.0, 5.0, 5.0, 4.0, 5.0],
         [5.0, 5.0, 4.0, 4.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0],
         [5.0, 5.0, 4.0, 5.0, 4.0, 3.0, 5.0, 4.0, 3.0, 4.0]]
    m=np.array(Mat)
    mean=np.mean(m)
    Cmean=np.mean(m,axis=0)
    Rmean=np.mean(m,axis=1)
    print("Matrix Overall average = ",mean, "Column mean = ",Cmean, " Row mean = ",Rmean, "Cmean[0] = ",Cmean[0]);

def triangle2(u1I,u2I,u1R,u2R):
    sim = 0
    R1=[];R2=[]
    u1I=set(u1I)
    u2I=set(u2I)
    union=u1I.union(u2I)
    intersect=u1I.intersection(u2I)
    u1I=list(u1I)
    u2I = list(u2I)
    union=list(union)
    intesect=list(intersect)
    #print("union = ",union," intersection = ",intersect)
    for i in range (0, len(union)):

        #print(union[i])
        if(union[i] in u1I) and (union[i] not in u2I):
            #print(union[i],"only in u1I")
            index1 = u1I.index(union[i])
            R1.append(u1R[index1])
            R2.append(0)
        elif (union[i] in u2I) and (union[i] not in u1I):
            index2 = u2I.index(union[i])
            R2.append(u2R[index2])
            R1.append(0)
            #print(union[i],"only in u2I")
        elif (union[i] in u2I) and (union[i]  in u1I):
            #print(union[i], "in both u1I and u2I")
            index1=u1I.index(union[i])
            index2=u2I.index(union[i])
            R1.append(u1R[index1])
            R2.append(u2R[index2])

    #print("R1= ",R1," R2 = ",R2)
    sim= triangle(R1,R2)
    if (len(u1R) > 0 and len(u2R) > 0 ) :
        meanu1 = sum(u1R) / len(u1R)
        meanu2 = sum(u2R) / len(u2R)
        SDu1 = stddev(u1R)
        SDu2 = stddev(u2R)
        URP = urp(meanu1, meanu2, SDu1, SDu2)
    else :
        URP=0
    sim=URP*sim
    return sim
########################################################For Sidra TMU, Triangle and URP
def triangle4(u1I,u2I,u1R,u2R):
    sim = 0
    R1=[];R2=[]
    u1I=set(u1I)
    u2I=set(u2I)
    union=u1I.union(u2I)
    intersect=u1I.intersection(u2I)
    u1I=list(u1I)
    u2I = list(u2I)

    union=list(union)
    intesect=list(intersect)
    #print("u1I = ",u1I,"u2I =",u2I," intersection = ",intersect)
    for i in range (0, len(intesect)):
            index1 = u1I.index(intesect[i])
            index2 = u2I.index(intesect[i])
            R1.append(u1R[index1])
            R2.append(u2R[index2])
    #print("R1= ",R1," R2 = ",R2)
    sim= triangle(R1,R2)
    if (len(u1R) > 0 and len(u2R) > 0 ) :
        meanu1 = sum(u1R) / len(u1R)
        meanu2 = sum(u2R) / len(u2R)
        SDu1 = stddev(u1R)
        SDu2 = stddev(u2R)
        URP = urp(meanu1, meanu2, SDu1, SDu2)
    else :
        URP=0
    sim=URP*sim
    return sim
#######################333#########33
########################################################For Sidra TMJ
def triangle3(u1I,u2I,u1R,u2R):
    sim = 0;JacSim=0;
    R1=[];R2=[]
    u1I=set(u1I)
    u2I=set(u2I)
    union=u1I.union(u2I)
    intersect=u1I.intersection(u2I)
    if (len(intersect) / len(union) > 0):
        JacSim = len(intersect) / len(union)
    else:
        JacSim = 0
    u1I=list(u1I)
    u2I = list(u2I)

    union=list(union)
    intesect=list(intersect)
    #print("u1I = ",u1I,"u2I =",u2I," intersection = ",intersect)
    for i in range (0, len(intesect)):
            index1 = u1I.index(intesect[i])
            index2 = u2I.index(intesect[i])
            R1.append(u1R[index1])
            R2.append(u2R[index2])
    #print("R1= ",R1," R2 = ",R2)
    #sim=Cosine(R1, R2)
    #sim=euclidean_distance(R1,R2)
    #sim = manhattan(R1, R2)
    sim = triangle(R1, R2)
    if(JacSim>0):
       sim=JacSim*sim
    return sim
def stddev(list):
    #print("The original list : " + str(list))
    mean = sum(list) / len(list)
    variance = sum([((x - mean) ** 2) for x in list]) / len(list)
    SD = variance ** 0.5
    return SD
def Cosine(v1,v2):
    sumxx, sumxy, sumyy=0,0,0
    size = 0;
    # print( "v1 = ", v1, " v2 = ", v2)

    for i in range(len(v1)):
        # print(" i = ",i," v[i] = ",v1[i]," v2[i] = ",v2[i])
        x = v1[i];
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    if((sumxx>0) & (sumyy>0)):
        return sumxy / math.sqrt(sumxx * sumyy)
    else:
        return 0
####################################3
#here mean is bicluster mean
def computePCC(u1I,u2I,u1R,u2R):
    sim = 0;JacSim=0;
    R1=[];R2=[]
    u1I=set(u1I)
    u2I=set(u2I)
    union=u1I.union(u2I)
    intersect=u1I.intersection(u2I)
    if (len(intersect) / len(union) > 0):
        JacSim = len(intersect) / len(union)
    else:
        JacSim = 0
    u1I=list(u1I)
    u2I = list(u2I)

    union=list(union)
    intesect=list(intersect)
    #print("u1I = ",u1I,"u2I =",u2I," intersection = ",intersect)
    for i in range (0, len(intesect)):
            index1 = u1I.index(intesect[i])
            index2 = u2I.index(intesect[i])
            R1.append(u1R[index1])
            R2.append(u2R[index2])
    #print("R1= ",R1," R2 = ",R2)
    #sim= triangle(R1,R2)
    uavg = statistics.mean(u1R)
    #bicavg= statistics.mean(u2R)
    bicavg=2.0
    sim=PCC(R1, R2,uavg,bicavg)
    return sim
#####################################33
def PCC(v1,v2,Uavg,Bicavg):

    num = 0.0; den_a = 0.0; den_b = 0.0;

    #print( "v1 = ", v1, " v2 = ", v2)

    for i in range(len(v1)):
        # print(" i = ",i," v[i] = ",v1[i]," v2[i] = ",v2[i])
        x = v1[i];
        y = v2[i]
        x=x-Uavg;
        y=y-Bicavg
        num += x * y;
        den_a += x * x;
        den_b += y * y;
    if ((den_a > 0) & (den_b > 0)):
        return num / (math.sqrt(den_a)*math.sqrt(den_b))
    else:
        return 0
###################

def euclidean_distance(point1, point2):
    # Calculate the sum of the squared differences between the coordinates
    squared_diffs = [(point1[i] - point2[i]) ** 2 for i in range(len(point1))]
    sum_squared_diffs = sum(squared_diffs)
    # Take the square root of the sum to get the Euclidean distance
    distance = math.sqrt(sum_squared_diffs)
    if distance>0:
        distance=1/distance
    else:
        distance=0
    return distance
def manhattan(a, b):

    distance=sum(abs(val1 - val2) for val1, val2 in zip(a, b))
    if distance > 0:
        distance = 1 / distance
    else:
        distance = 0
    return distance
# u1I=[1,2,3,4,5,7]
# u2I=[3,4,5,6]
# u1R=[1,2,3,4,5,7]
# u2R=[3,4,5,6]
u1I=[11,12,13,14,15]
u2I=[11,13,15]
u1R=[4,5,4,2,4]
u2R=[5,1,2]

#TMJSim=triangle3(u1I,u2I,u1R,u2R)
#print("TMJsim = ",TMJSim)
#ITRsim=Sim*URP
#print("triangle = ",Sim," urp = ",URP, " ITRsim = ",ITRsim)
# v1=[4,5,4,2,4]
# v2=[5,0,1,0,2]
# uavg=3.8;bicavg=1.6;
# PCCsim=PCC(v1,v2,uavg,bicavg)
# print("PCCsim= ",PCCsim)




