#import ExtractMatrix8
import matplotlib
matplotlib.use('Agg')  # must be set before any submodule imports pyplot, so plt.show() never blocks headless runs
import Amazon_KMeansClustering#SidExtractMatrix3 doesn't sort clusters according to MSR
import AmazonExp1 #Fahad1 contains MSR sorted Bics in 6x6 GridPos, Fahad2 contains non-MSR sortde Bics of 8x8 Grid in GridPos
#FahadExp1 contains 8x8 grid but actions=4
#from ReadFilmTrust import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Listcount
#import SARSA1

def getItemsandRating(fit, usrid):
    return fit[usrid]['Items'], fit[usrid]['Rating']


def run_full_population():
    """Runs the RL recommender over every user in the Amazon dataset (not a sample) and
    reproduces the paper's States-Visited / Return-Earned per-user figures. This is the
    full-population counterpart to MultiUserEval.py's 40-user sample."""
    d = pd.read_csv("amazon test set.csv", sep=',', engine='python',
                     names=['User No.', 'Product ID code', 'Rating'])
    data = np.array(d)
    firscol = list(data[:, 0])  # Extract first column of the data
    fit = {}
    usersList = sorted(int(u) for u in set(firscol))
    print(" Size of Users list = ", len(usersList))
    for z in usersList:
        it = []
        r = []
        for x in range(0, data.shape[0]):
            if data[x][0] == z:
                it.append(int(data[x][1]))
                r.append(data[x][2])
        fit[z] = {
            'User_Number': z,
            'Items': it,
            'Rating': r
        }

    precisioni = []; recalli = []; Fmeasurei = []; coveragei = []; Reti = []

    filepath = "amazon test set.csv"
    Dictobj_Users, Dictobj_Products = Amazon_KMeansClustering.KMeans_Clusters(filepath)
    notPred = 0
    StartStates = []
    dicty = {}
    per_user_rows = []
    for i, testUser in enumerate(usersList):
        u1I, u1R = getItemsandRating(fit, testUser)
        predict, cover, prec, recal, FM, ret, states, states_visited = AmazonExp1.run2(u1I, u1R, testUser, Dictobj_Products)
        dicty[testUser] = {
            'User_ID': testUser,
            'states_visited': states_visited,
            'states_visited size': len(states_visited),
            'Return': ret
        }
        StartStates.append(states)

        coveragei.append(cover)
        precisioni.append(prec)
        recalli.append(recal)
        Fmeasurei.append(FM)
        Reti.append(ret)
        feasible = len(predict) > 0
        if not feasible:
            notPred += 1
        per_user_rows.append({
            'user': testUser, 'n_items': len(u1I), 'start_state': states,
            'states_visited_count': len(states_visited), 'coverage': cover,
            'precision': prec, 'recall': recal, 'f_measure': FM, 'return': ret,
            'feasible': feasible,
        })
        if (i + 1) % 100 == 0 or (i + 1) == len(usersList):
            print(f"[Amazon full-population] {i + 1}/{len(usersList)} users done")

    pd.DataFrame(per_user_rows).to_csv('fullpop_eval_Amazon.csv', index=False)

    sumcover = sumprec = sumrecal = sumFM = sumRet = 0
    for i in range(0, len(usersList)):
        sumcover += coveragei[i]
        sumprec += precisioni[i]
        sumrecal += recalli[i]
        sumFM += Fmeasurei[i]
        sumRet += Reti[i]
    coverage = float(sumcover / len(usersList))
    Precision = float(sumprec / len(usersList))
    Recall = float(sumrecal / len(usersList))
    Fmeasure = float(sumFM / len(usersList))
    Return = float(sumRet / len(usersList))
    userCoverage = ((len(usersList) - notPred) / len(usersList)) * 100
    print(" RLRecommender Item coverage =", coverage, 'Precision: ', Precision, " Recall =", Recall, " Fmeasure =",
          Fmeasure, "Return =", Return, " userCoverage= ", userCoverage)

    pd.DataFrame([{
        'dataset': 'Amazon', 'n_users': len(usersList), 'n_infeasible': notPred,
        'coverage_mean': coverage, 'precision_mean': Precision, 'recall_mean': Recall,
        'f_measure_mean': Fmeasure, 'return_mean': Return, 'user_coverage_pct': userCoverage,
    }]).to_csv('fullpop_eval_summary_Amazon.csv', index=False)

    print("unique states visited = ", set(StartStates), "and number of unique states = ", len(set(StartStates)))
    Icount, S = Listcount.count_items(StartStates, savepath='Figure_StartStateCounts_Amazon.png')
    print("unique States with count = ", S)

    users_out = []
    statesize = []
    Ret = []
    for testUser in usersList:
        users_out.append(testUser)
        statesize.append(dicty[testUser].get('states_visited size'))
        Ret.append(dicty[testUser].get('Return'))

    plt.figure()
    plt.title("States Visited By Each User")
    plt.xlabel("User id")
    plt.ylabel("Visited States Count")
    plt.bar(users_out, statesize, alpha=0.6, color='blue', width=5)
    plt.savefig('Figure_StatesVisitedByUser_Amazon.png', dpi=150)
    plt.close()

    plt.figure()
    plt.title("Return Earned By Each User")
    plt.xlabel("User id")
    plt.ylabel("Return Earned")
    plt.bar(users_out, Ret, alpha=0.6, color='red', width=5)
    plt.savefig('Figure_ReturnByUser_Amazon.png', dpi=150)
    plt.close()

    return coverage, Precision, Recall, Fmeasure, Return, userCoverage


if __name__ == '__main__':
    run_full_population()
