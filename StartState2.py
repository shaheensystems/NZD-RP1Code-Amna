import numpy as np
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
    #Jaccard similarity between the USERS who touched each cluster, not the items.
    #Item sets are disjoint by construction across K-Means clusters (every product belongs
    #to exactly one cluster), so item-Jaccard is structurally always 0 for any transition
    #between different clusters. User sets are not partitioned this way -- a single user's
    #ratings span many item clusters -- so user-Jaccard gives a real, non-degenerate signal.
    #See REWARD_FUNCTION_MATH.md for the full derivation and empirical validation.
    Users1 = set(Dictobj[StartState].get('Cluster_Users'))
    Users2 = set(Dictobj[NextState].get('Cluster_Users'))
    unionset = Users1.union(Users2)
    if not unionset:
        return 0.0
    intersect = Users1.intersection(Users2)
    UsersJacSim = len(intersect) / len(unionset)
    return UsersJacSim

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
