#import ExtractMatrix8
import Amazon_KMeansClustering#SidExtractMatrix3 doesn't sort clusters according to MSR
import AmazonExp1 #Fahad1 contains MSR sorted Bics in 6x6 GridPos, Fahad2 contains non-MSR sortde Bics of 8x8 Grid in GridPos
#FahadExp1 contains 8x8 grid but actions=4
#from ReadFilmTrust import *
from Read_Amazon import *
import pandas as pd
import matplotlib.pyplot as plt
import Listcount
#import SARSA1
rat2=[];frat=[];fitm=[];itm2=[];fusr=[];chkusr=[]

d =  pd.read_csv("F:/Thesis Supervised/Year 2023/NZD/Python NZD1/Amna RL2/amazon test set.csv", sep=',', engine='python')
#d =  pd.read_csv('amazon test set.csv', sep=',', engine='python')
data = np.array(d)
#print(data.shape[0])
firscol=[]
firscol=list(data[:,0]) # Extract first column of the data
fit={}
users= set(firscol)
usersList=(list(users))
usersList.sort()
for i in range(0, len(usersList)):
    usersList[i] = int(usersList[i])
print(" Size of Users list = ",len(usersList))
#print(" Users list = ",usersList)
for z in usersList:
 it=[]
 r=[]
 for x in range(0,data.shape[0]):
  if(data[x][0]==z):
   it.append(int(data[x][1]))
   r.append(data[x][2])
 fit[z]= {
  'User_Number':z,
  'Items' :it,
  'Rating':r
 }

def getItemsandRating(usrid):
 return fit[usrid]['Items'],fit[usrid]['Rating']

def getRating(usrid,itmid):
 rating=0
 for x in range(0,data.shape[0]):
  if(data[x][0]==usrid and data[x][1]==itmid):
   rating=data[x][2]
   break
 return rating
precisioni =[];recalli =[];Fmeasurei=[];coveragei=[];Reti=[]

data, num_users, num_items,list_users,list_items = read_Dataset()

usr, items, scores, inter, user_index, item_index = load_Dataset(data, num_users,num_items)

#Dict = ExtractMatrix8.main(inter, list_users, list_items)# inter is basically user-items rating matrix
filepath = "F:\\Thesis Supervised\\Year 2023\\NZD\\Python NZD1\\Amna Obaid\\amazon cleaned updated amna.csv";
Dictobj_Users,Dictobj_Products =Amazon_KMeansClustering.KMeans_Clusters(filepath)
notPred=0;
StartStates=[]
dicty={}
for testUser in usersList :
    Items = []
    ratings = []
    # print (testUser)
    u1I, u1R = getItemsandRating(testUser)
    # u1I, u1R = getDeficientCLOsMarks(testUser)
    # print('Items rated by test user are', u1I)
    Items.append(u1I)
    ratings.append(u1R)
    #print("Items = ", Items, " u1I = ", u1I)
    #predict, cover, prec, recal, FM = SARSA1.run2(u1I, u1R, testUser, Dict)
    #env = TwoDGridWorld19.TwoDGridWorld(6, Biclust, u1I, u1R, Dict)
    predict, cover, prec, recal, FM,ret,states, states_visited = AmazonExp1.run2(u1I, u1R, testUser, Dictobj_Products)
    dicty[testUser] = {
        'User_ID': testUser,
        'states_visited': states_visited,
        'states_visited size': len(states_visited),
        'Return': ret
    }
    StartStates.append(states)

    if len(predict) > 0:
        coveragei.append(cover)
        precisioni.append(prec)
        recalli.append(recal)
        Fmeasurei.append(FM)
        Reti.append(ret)
    else:
        coveragei.append(cover)
        precisioni.append(prec)
        recalli.append(recal)
        Fmeasurei.append(FM)
        Reti.append(ret)
        notPred += 1
    # i = i + 1;
sumcover = sumprec = sumrecal = sumFM = sumRet = 0
# print("length of cover = ",len(coveragei))
for i in range(0, len(usersList)):
    sumcover += coveragei[i]
    sumprec += precisioni[i]
    sumrecal += recalli[i]
    sumFM += Fmeasurei[i]
    sumRet += Reti[i]
# print("sumcover = ",sumcover)
coverage = float(sumcover / len(usersList));
Precision = float(sumprec / len(usersList));
Recall = float(sumrecal / len(usersList));
Fmeasure = float(sumFM / len(usersList));
Return = float(sumRet / len(usersList))
userCoverage = ((len(usersList) - notPred) / len(usersList)) * 100
print(" RLRecommender Item coverage =", coverage, 'Precision: ', Precision, " Recall =", Recall, " Fmeasure =",
      Fmeasure, "Return =", Return, " userCoverage= ", userCoverage)
print(coverage)
print(Precision)
print(Recall)
print(Fmeasure)
print(Return)
print(userCoverage)

# print("Start states = ",StartStates)
print("unique states visited = ", set(StartStates), "and number of unique states = ", len(set(StartStates)))
Icount, S = Listcount.count_items(StartStates)
print("unique States with count = ", S)
# print("states_visited = ",states_visited)
# print(dicty)
users = [];
statesize = [];
Ret = []
for testUser in usersList:
    # print(dicty[testUser].get('states_visited size'))
    users.append(testUser)
    statesize.append(dicty[testUser].get('states_visited size'))
    Ret.append(dicty[testUser].get('Return'))
plt.title("States Visited By Each User")
plt.xlabel("User id")
plt.ylabel("Visited States Count")
plt.bar(users, statesize, alpha=0.6, color='blue', width=5)
plt.show()
plt.title("Return Earned By Each User")
plt.xlabel("User id")
plt.ylabel("Return Earned")
plt.bar(users, Ret, alpha=0.6, color='red', width=5)
plt.show()


