import numpy as np
from Triangle import triangle2, triangle3,computePCC
import statistics

# For start state We need 1. bilcuster matrix, bicluster users and items list, 2. Target user rated items list and rating vector
# Define 4 matrices along with users and items list, compute ITR similarity of target user with these matrices.
#store in two lists. One list contain matrix number, 2nd contains ITR similarity values. Store 2nd list in a third list.
# Sort third list from ascending to descending. For highest value of ITR similarity in third list, check its location or index in 2nd list.
# For location/index in 2nd list check same location/index in first list. This will bicluster number, shich will be the start state
GridMat=[[1,4],[2,3]]
Gridlist=[1,2,3,4]
u1I=[11,12,13,14]
u1R=[4,5,4,2]

def computeSim(Dictobj) :
    Simlist = []
    #print(Dictobj)
    for x in range(0, len(Dictobj)):
        mean= np.mean(Dictobj[x].get('Biclust'))
        Cmean = np.mean(Dictobj[x].get('Biclust'), axis=0)
        Rmean = np.mean(Dictobj[x].get('Biclust'), axis=1)
        #print("Matrix ",x," Overall average in computeSim = ", mean, "Column mean = ", Cmean, " Row mean = ", Rmean);
        ITRSim = triangle2(u1I, Dictobj[0].get('Cluster_Items'), u1R, Cmean)
        #print(x,"ITRsim = ", ITRSim)
        Simlist.append(ITRSim)
    return Simlist
#Dictobj contains, cluster number, rows, columns and Biclust ratings matrix
################################33
def computeJaccard(Dictobj,u1I,Biclust):
    Simlist = []
    TH=-1
    #for x in range(0, 36):
    for x in Biclust:
    #for x in range(0, len(Dictobj)):
        # print("Matrix ",x," Overall average in computeSim = ", mean, "Column mean = ", Cmean, " Row mean = ", Rmean);
        u2I=set(Dictobj[x].get('Cluster_Items'))
        #print("u1I in computejaccard = ",u1I)
        u1I = set(u1I)
        unionset=     u1I.union(u2I)
        intersect = u1I.intersection(u2I)
        if(len(intersect) / len(unionset)>0) :
            JacSim= len(intersect)/len(unionset)
        else :
            JacSim=-1
        #print(x,"Jacsim = ", JacSim," u2I = ",u2I," u1I = ",u1I)
        Simlist.append(JacSim)
        if(JacSim==1.0) :
            break;

    #print("Simlist = ", Simlist)
    Simlist2 = []
    Simlist2 = Simlist.copy();  # Original code
    # Simlist2=Simlist;
    Simlist2.sort(reverse=True)
    #print("Simlist2 =", Simlist2, " Simlist2[0] = ", Simlist2[0], )
    if(Simlist2[0]>0):
        StartState = Simlist.index(Simlist2[0])
    else :
        return -1

    #print("index of Simlist2[0] in Simlist and Start state = ", StartState)
    if (StartState>= 0) :
        return StartState
    else :
        return -1;



 #print("Cluster ",x," rowlist = ",rowlist,"columlist = ",columnlist)
    #print("Intersection = ",intersect)
def computeMyReward(StartState,NextState,Dictobj) :
    rewardd=-1;
    Items1 = set(Dictobj[StartState].get('Cluster_Items'))
    Items2 = set(Dictobj[NextState].get('Cluster_Items'))
    unionset =Items1.union(Items2)
    intersect = Items1.intersection(Items2)
    ItemsJacSim = len(intersect) / len(unionset)
    rewardd=ItemsJacSim

    return rewardd

####################################33
#Parameters State = a sate for which we will check whether it is a goal state
#itemset = previous items recommended by visisting different states/biclusters
# I think compute goal input should be itemset, nextstate and Dictobj and  should return Recset. itemset will be the previous recommended items
#Define another function IsGoal() in TwoDGridWorld.py which should have subsetcount;TH; and goal as global variable;
def computeGoal(itemset,NextState,Dictobj,subsetcount):
    itemlist=[];Recset=itemset;#Recset contain items to be recommended
    itemlist = Dictobj[NextState].get('Cluster_Items')
    Biclustitems=set(itemlist)
    if (Biclustitems.issubset(Recset)):#Original code
    #if (Biclustitems == Recset):

            subsetcount += 1
            #print("Biclustitems ", Biclustitems, " is subset of Recset = ", Recset," subsetcount = ",subsetcount)
    else:

            #print("Biclustitems ", Biclustitems, " is not subset of Recset = ", Recset)
            Recset = Recset.union(Biclustitems)

    #print("Recommended items  = ",Recset, " and Bicluster items = ",Biclustitems)

    return Recset,subsetcount#Mubbashir plz note that after returnning Recset in GridWorld, make union of Recset, with previous itemsdef
################################################33
#In this method, If no change in set of previous and new recommended items then it can be a goal state.
#It can be implemented by getting union of both sets. If Size is same as previous then it is no change...
def computeGoal3(itemset,NextState,Dictobj,subsetcount):
    Recset=itemset;#Recset contain items to be recommended, Previous recommended items
    itemlist = Dictobj[NextState].get('Cluster_Items')
    Biclustitems=set(itemlist)#New recommended items
    size1=len(Recset)
    size2=len(Biclustitems.union(Recset))
    if (size1==size2):#Original code
    #if (Biclustitems == Recset):
            subsetcount += 1
            #print("size1= ",size1, "size2= ",size2 ," subsetcount= ",subsetcount )
            #Recset = Recset.union(Biclustitems)
        #print("Biclustitems ", Biclustitems, " is subset of Recset = ", Recset," subsetcount = ",subsetcount)
    else:
            Recset = Recset.union(Biclustitems)
            subsetcount=0
            #print("in else subsetcount= ",subsetcount)
    return Recset,subsetcount#Mubbashir plz note that after returnning Recset in GridWorld, make union of Recset, with previous itemsdef




################################33

def convertSettoList(Set) :
    List=[]
    for i in Set:
        List.append(i)
    return List
# itemset={1,2,3}
# Recitems=computeGoal(itemset,0,Dictobj);
# print("Recommended items = ",Recitems)
#This function returns recommended items rating vector to compute MAE and RMSE
def computeGoal2(itemset,RatingsSet,NextState,Dictobj,subsetcount):
    intersectList = [];
    itemlist3=convertSettoList(RatingsSet)
    itemlist2 = convertSettoList(itemset)
    print("itemset = ",itemset," RatingsSet = ",RatingsSet)
    ratings=[];Recset=itemset;#Recset contain items to be recommended
    #itemlist2 = list(itemset)
    Cmean = np.mean(Dictobj[NextState].get('Biclust'), axis=0)
    print("Column mean = ",Cmean)
    Biclustratings=Cmean;
    itemlist = Dictobj[NextState].get('Cluster_Items')
    Biclustitems=set(itemlist)
    if (Biclustitems.issubset(Recset)):

            subsetcount += 1
            print("Biclustitems ", Biclustitems, " is subset of Recset = ", Recset," subsetcount = ",subsetcount)
    else:

            print("Biclustitems ", Biclustitems, " is not subset of Recset = ", Recset)

            intersectSet=Recset.intersection(Biclustitems)


            intersectList=convertSettoList(intersectSet)


            print("Intersect Set = ", intersectSet, " intersectList = ",intersectList )
            Recset = Recset.union(Biclustitems)
    for i in intersectList:
        index=itemlist.index(i)#itemlist is basically Biclustitems set
        index2=itemlist2.index(i)#itemlist2 is input itemset
        print("itemlist = ",itemlist," and Biclustitems = ",Biclustitems)
        print(" For ", i," index in Bicluster columns Cmean = ",index ," and Bicluster Column Cmean at this index = ",Cmean[index])
        print("ItemList2 = ",itemlist2," i = ",i," index2 = ",index2)
        print("Cmean[index] = ",Cmean[index],"itemlist2[index2]= ",itemlist2[index2],"itemlist3[index2]= ",itemlist3[index2])
        Cmean[index]= float((Cmean[index]+itemlist3[index2])/2)
        #print(" For ", i)#, "itemlist  = ", itemlist)
    #for i in Recset:
    Biclustratings=Cmean
    return Recset,Biclustratings,subsetcount#Mubbashir plz note that after returnning Recset in GridWorld, make union of Recset, with previous items

# itemset={20,5,3}
# RatingsSet=[3,4,3]
# sscount=0
# list=[1,6,12,18,24,30]
# Recitems1,Recratings1,subcount1=computeGoal2(itemset,RatingsSet,list[0],Dictobj,sscount);
# print("Recommended items = ",Recitems1," Ratings = ",Recratings1)