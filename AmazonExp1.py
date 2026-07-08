#In this file action space= 4 actions and Grid size=8x8
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import time
import matplotlib.pyplot as plt
from Listcount import *
import Amazon_KMeansClustering
import StartState2
#from StartState2 import *
#from IPython.display import clear_output
import time
#import tensorflow

# custom 2d grid world enviroment which extends gym.Env
class TwoDGridWorld(gym.Env):
    """
        - a size x size grid world which agent can ba at any cell other than terminal cell
        - terminal cell is set to be the last cell or bottom right cell in the grid world
        - 5x5 grid world example where X is the agent location and O is the tremial cell
          .....
          .....
          ..X..
          .....
          ....O -> this is the terminal cell where this is agent headed to
        - Reference : https://github.com/openai/gym/blob/master/gym/core.py
    """
    metadata = {'render.modes': ['console']}

    # actions available
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


    def __init__(self, size,Biclust,u1I,Dict):

        self.Dictobj = Dict  # call to main method of ExtractMatrix8 to get obj1 object
        #print("Dictobj = ", self.Dictobj)
        self.majorBreak=0
        super(TwoDGridWorld, self).__init__()
        self.Biclust=Biclust
        #print("Biclust = ",self.Biclust)
        self.size = size  # size of the grid world
        self.subsetcount = 0;
        self.Recitems=set()
        goal = False;  # if (subsetcount>=TH) then goal = true
        self.Recset=set()
        self.stopcount=10;
        #self.end_state = (size * size) - 1  # Actual Code
        self.end_state = 0  # Mubbashir added code to define goal state

        self.GridPos = [[0, 1, 5, 6, 14, 15],
                        [2, 4, 7, 13, 16, 25],
                        [3, 8, 12, 17, 24, 26],
                        [9, 11, 18, 23, 27, 32],
                        [10, 19, 22, 28, 31, 33],
                        [20, 21, 29, 30, 34, 35]]

        self.ActualGrid = [[0, 1, 2, 3, 4, 5 ],
                          [6, 7, 8, 9, 10, 11],
                        [12, 13, 14, 15, 16,17 ],
                        [18, 19, 20, 21, 22, 23,],
                        [24, 25, 26, 27, 28, 29],
                         [30, 31, 32, 33, 34, 35]]

        self.StartState = StartState2.computeJaccard(self.Dictobj, u1I,self.Biclust)#Compute start state using Jaccard
        R = self.StartState//self.size
        C = self.StartState%self.size
        self.StartState=self.GridPos[R][C]
        print("StartState in env.GridPos= ", self.StartState)
        if(self.StartState==-1) :
            print("Start state = -1 recommendations not possible for this user")
            self.majorBreak=-1
            return


        self.agent_position =self.StartState#Agent start state
        # respective actions of agents : up, down, left and right
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=0, high=size * size, shape=(1,), dtype=np.uint8)

    def step(self, action, env):
        info = {}  # additional information

        reward = 0;
        prevState = self.agent_position

        row = self.agent_position // self.size
        col = self.agent_position % self.size
        if action == self.UP:
            if row != 0:
                self.agent_position -= self.size
            else:
                reward = 0
        elif action == self.LEFT:
            if col != 0:
                self.agent_position -= 1
            else:
                reward = 0
        elif action == self.DOWN:
            if row != self.size - 1:
                self.agent_position += self.size
            else:
                reward = 0
        elif action == self.RIGHT:
            if col != self.size - 1:
                self.agent_position += 1
            else:
                reward = 0
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        # done=False
        # done = bool(self.agent_position == self.end_state)
        done = bool(self.subsetcount == self.stopcount)
        if (done):
            reward = 1 if done else reward
        Nrow = self.agent_position // self.size
        Ncol = self.agent_position % self.size
        return env.GridPos[Nrow][Ncol], reward, done, env.GridPos[row][col], prevState

    def render(self, env,mode='console'):
        '''
            render the state
        '''
        if mode != 'console':
            raise NotImplementedError()

        row = self.agent_position // self.size
        col = self.agent_position % self.size
        # Original code
        # for r in range(self.size):
        #     for c in range(self.size):
        #         if r == row and c == col:
        #             print("X", end='')
        #         else:
        #             print('.', end='')
        #     print('')

        #print(np.array(GridPos))
        return env.GridPos
    #Reset() method will reset whole grid and moves the agent to some initial/start state. So return values can be two.
    #1. Number representing initial/start stae and complete grid
    def reset(self,env,u1I,u1R):
        # -1 to ensure agent inital position will not be at the end state
        #self.agent_position = np.random.randint((self.size * self.size) - 1)

        self.StartState = StartState2.computeJaccard(self.Dictobj, u1I,self.Biclust)  # Compute start state using Jaccard
        self.majorBreak=0;
        #self.StartState = StartState2.computeStartState_usingTMJ(self.Dictobj, u1I, u1R,self.Biclust)
        print("StartState in Actual Grid= ", self.StartState)
        if (self.StartState == -1):
            print("Start state = -1 in reset() method recommendations not possible for this user")
            self.majorBreak = -1
            return -1,-1
        self.agent_position = self.StartState  # Agent start state
        row=self.StartState//self.size
        col=self.StartState%self.size
        self.StartState=env.GridPos[row][col]
        env.agent_position=env.ActualGrid[row][col]
        #start=self.agent_position
        self.Recitems = set()
        return env.GridPos,self.StartState# Working code


        #return GridPos, np.array(start).astype(np.uint8)

    def close(self):
        pass


def learnPolicy(env,u1I,u1R) :
    Return=0; strt=0;


    ############################################################################3333
    num_episodes = 100
    steps_total = []
    rewards_total = []
    egreedy_total = []
    states_visited=[]
    gamma = 0.95

    # Factor to balance the ratio of action taken based on past experience to current situtation
    learning_rate = 0.9
    # exploit vs explore to find action
    # Start with 70% random actions to explore the environment
    # And with time, using decay to shift to more optimal actions learned from experience
    number_of_states = 36
    number_of_actions = env.action_space.n
    egreedy = 0.9
    egreedy_final = 0.1
    egreedy_decay = 0.999
    Q = torch.zeros([number_of_states, number_of_actions])
    # resets the environment
    Grid, state = env.reset(env, u1I, u1R)
    for i_episode in range(num_episodes):
        states_visited = []
        Return = 0;
        reward = 0
        sscount = 0;
        env.subsetcount = 0;

        step = 0

        while True:

            step += 1

            random_for_egreedy = torch.rand(1)[0]#To generate a random number between 0-1
            #print("Random for greedy = ", random_for_egreedy);
            if random_for_egreedy > egreedy:    # Below code chooses action which has the highest Q value.
                #random_values = Q[state] + torch.rand(1, number_of_actions) / 1000#Original code
                random_values = Q[state] + torch.rand(1, number_of_actions) / 1000#Mubbashir changed code
                action = torch.max(random_values, 1)[1][0]
                action = action.item()
            else: # Below code chooses action at random
                action = env.action_space.sample()
                #print("random action = ",action)

            if egreedy > egreedy_final:
                egreedy *= egreedy_decay

            #new_state, reward, done, info = env.step(action)#info gives probability of the state
            new_state, reward, done, BilcustGridPos, prevState = env.step(action,env)#Mubbashir here reward is doing nothing, I am getting reward at line
            #        if ((np.ndarray.item(new_state)!=prevState)): reward = StartState2.computeReward(prevState, np.ndarray.item(new_state), env.Dictobj)
            states_visited.append((new_state))
            statecount= states_visited.count((new_state))
            #print("state = ",np.ndarray.item(new_state)," statecount = ",statecount,"step = ",step)
            row = new_state // env.size
            col = new_state % env.size
            #print("row = ",row)#"np.ndarray.item(row) = ",np.ndarray.item(row))
            #print("column = ",col)#" np.ndarray.item(col) = ",np.ndarray.item(col))
            env.agent_position = env.ActualGrid[row][col]
            # env.render()
            if (strt == 0):

                #itemset = set(env.Dictobj[prevState].get('Cluster_Columns'))
                itemset = set();
                #print("item set = ", itemset, " Reset State = ", state)
                env.Recitems, sscount = StartState2.computeGoal(itemset, env.GridPos[row][col],
                                                            env.Dictobj, env.subsetcount);
            else:
                env.Recitems, sscount = StartState2.computeGoal(env.Recitems, env.GridPos[row][col],
                                                            env.Dictobj, env.subsetcount);
            env.subsetcount = sscount  # feed convergence count back so step()'s done check (subsetcount==stopcount) is live

            if ((statecount == env.stopcount) or (step==144)):
                done = True

            if (((new_state)!=prevState)):# If both prevstate and new_state are same then no reward
                R = prevState // env.size
                C = prevState % env.size
                reward = StartState2.computeMyReward(env.GridPos[R][C], env.GridPos[row][col],env.Dictobj)
                #reward = StartState2.computeSidReward(prevState, np.ndarray.item(new_state), env.Dictobj)
                #reward = StartState2.computeRLRS1Reward(env.GridPos[R][C], env.GridPos[np.ndarray.item(row)][np.ndarray.item(col)],env.Dictobj)
                Return += reward;
                #print("Reward: {:.2f}".format(reward))

            strt += 1;
            alpha=0.5; #Qlearning
            Q[state, action] = Q[state, action] + \
                                     alpha * (reward + gamma * torch.max(Q[new_state]) - Q[state, action])
            state = new_state#Mubbashir added code

            # env.render()
            # time.sleep(0.4)

            if done:
                steps_total.append(step)
                #print("step = ",step,"steps_total = ",steps_total)
                #print("states_visited = ",states_visited);
                #print("env.subsetcount = ",env.subsetcount)
                #rewards_total.append(reward)#Original code
                rewards_total.append(Return)
                egreedy_total.append(egreedy)

                #print(Q)
               #print('Episode: {} Reward: {} Steps Taken: {}'.format(i_episode, reward, step))
               #break
                if i_episode % 20 == 0:
                     print('Episode: {} Return: {} Steps Taken: {}'.format(i_episode, Return, step),"Episode# ",i_episode, "Ends")
                break

    #print(Q)

    print("Percent of episodes finished successfully: {0}".format(sum(rewards_total) / num_episodes))
    print("Percent of episodes finished successfully (last 100 episodes): {0}".format(sum(rewards_total[-100:]) / 100))

    print("Average number of steps: %.2f" % (sum(steps_total) / num_episodes))
    print("Average number of steps (last 100 episodes): %.2f" % (sum(steps_total[-100:]) / 100))
    # # Below graph plots number of episodes vs rewards; X-axis contain episodes, Y-axis rewards, max value of reward=1
    # plt.figure(figsize=(12,5))
    # plt.title("Return/ Episodes")
    # plt.xlabel("Episodes")
    # plt.ylabel("Return")
    # #plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green', width=5)
    # plt.plot(torch.arange(len(rewards_total)), rewards_total, color='green')
    # plt.show()
    # # Below graph plots number of episodes vs steps; X-axis contain number of episodes, Y-axis contain Episode steps
    # plt.figure(figsize=(12,5))
    # plt.title("Episodes Steps / Episodes length")
    # plt.xlabel("Episodes")
    # plt.ylabel("Episode Steps")
    # #plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red', width=5)
    # plt.plot(torch.arange(len(steps_total)), steps_total)
    # plt.show()
    #
    # #Below graph plots Egreedy value vs Episodes; X-axis contain number of episodes, Y-axis contain Egreedy value
    # plt.figure(figsize=(10,5))
    # plt.title("Egreedy value/Episodes")
    # plt.xlabel("Episodes")
    # plt.ylabel("Egreedy value")
    # #plt.bar(torch.arange(len(egreedy_total)), egreedy_total, alpha=0.6, color='blue', width=5)
    # plt.plot(torch.arange(len(egreedy_total)), egreedy_total)
    # plt.show()
    return Q

def extractPolicy(Q,env,u1I,u1R) :
    strt = 0;
    Return = 0;
    env.subsetcount = 0;
    sscount = 0;
    #Mubbahsir start from state in reset, check start should be a number or np.ndarray.item
    Grid,state = env.reset(env,u1I,u1R)
    print("state in GridPos = ",state," Q = ",Q[state])
    done = False
    sequence = []
    count=0;
    states_visited=[]
    while not done:
        # Choose the action with the highest value in the current state
        count+=1;
        if (states_visited.count(state) >= 3):
            action = env.action_space.sample()
        else:
         if np.max(Q[state]) > 0:
            action = np.argmax(Q[state])
            #print(" state  = ", state, "and Q[State] > 0"," action = ",action,"value at state and action = ",Q[np.array(state)][action])

        # If there's no best action (only zeros), take a random one
         else:
            #print(" state  = ", state, "and Q[State] <= 0")
            action = env.action_space.sample()

        # Add the action to the sequence
        sequence.append(action)

        # Implement this action and move the agent in the desired direction
        #new_state, reward, done, info = env.step(action)#Original code
        new_state, reward, done, BilcustGridPos, prevState= env.step(action,env)#Mubbashir code
        states_visited.append(state)
        statecount = states_visited.count(new_state)
        #print("state = ",np.ndarray.item(new_state)," statecount = ",statecount)
        row = (new_state) // env.size
        col = (new_state) % env.size
        if (strt == 0):
            itemset = set();
            #itemset = set(StartState2.Dictobj[ env.GridPos[R][C]].get('Cluster_Columns'))
            #print("item set = ", itemset, " Reset State = ", state)
            env.Recitems, sscount = StartState2.computeGoal(itemset, env.GridPos[row][col],
                                                        env.Dictobj, env.subsetcount);
        else:
            env.Recitems, sscount = StartState2.computeGoal(env.Recitems, env.GridPos[row][col],
                                                        env.Dictobj, env.subsetcount);

        env.subsetcount = sscount  # feed convergence count back so step()'s done check (subsetcount==stopcount) is live
        #print("Recommended items = ", env.Recitems, " sscount = ", sscount)
        if (statecount == env.stopcount or count == 15):
            print("states_visited = ", states_visited)
            print("unique states_visited = ", set(states_visited)," and size  = ",len(set(states_visited)))
            done = True

        if (((new_state)!=prevState)):# If both prevstate and new_state are same then no reward
            R = prevState // env.size
            C = prevState % env.size
            reward = StartState2.computeMyReward(env.GridPos[R][C], env.GridPos[row][col],
                                                 env.Dictobj)
            #reward = StartState2.computeRLRS1Reward(prevState, np.ndarray.item(new_state), env.Dictobj)
            Return += reward;
            #print("Reward: {:.2f}".format(reward))

        strt += 1;
        #env.agent_position = env.ActualGrid[row][col]#NO NO Don't do this
        state = new_state

        #clear_output(wait=True)
    #print(f"Actions of Learned Policy = {sequence}")
    return sequence
def applyPolicy(winning_sequence,env,u1I,u1R) :
    strt=0;Return=0;env.subsetcount =0;
    states_visited = []
    Grid, state = env.reset(env,u1I,u1R)
    states_visited.append(state)

    #winning_sequence = [2, 3, 3, 3, 1, 3, 1, 3, 1, 3]
    for a in winning_sequence:
        #new_state, reward, done, info = env.step(actions[a])
        new_state, reward, done, BilcustGridPos, prevState=env.step(a,env)
        states_visited.append(new_state)
        row = (new_state) // env.size
        col = (new_state) % env.size
        if (strt == 0):
            itemset = set()
            env.Recitems, sscount = StartState2.computeGoal(itemset, env.GridPos[row][col],
                                                               env.Dictobj, env.subsetcount);
        else:
            env.Recitems, sscount = StartState2.computeGoal(env.Recitems, env.GridPos[row][col],
                                                                 env.Dictobj, env.subsetcount);
        env.subsetcount = sscount  # feed convergence count back so step()'s done check (subsetcount==stopcount) is live
        if (((new_state)!=prevState)):# If both prevstate and new_state are same then no reward

             R = prevState // env.size
             C = prevState % env.size
             reward = StartState2.computeMyReward(env.GridPos[R][C], env.GridPos[row][col],env.Dictobj)

        #     #reward = StartState2.computeRLRS1Reward(env.GridPos[R][C], env.GridPos[np.ndarray.item(row)][np.ndarray.item(col)],env.Dictobj)
             Return += reward
        #     #print("Reward: {:.2f}".format(reward))

        strt += 1;
        env.render(env)
    print("Apply policy states_visited = ", states_visited )
    # TNST=Testing number of steps

    stateset=set(states_visited)
    print("Apply policy unique states_visited = ", stateset, " and size  = ", len(stateset))
    return env.Recitems, Return, state, list(stateset)

actions = {
        'Up': 0,
        'Left': 1,
        'Down': 2,
        'Right': 3,

}


def computeCoverage(u1I,prediction) :
    u1I = set(u1I)
    prediction = set(prediction)
    if (len(prediction) != 0):  # If there is some recommendation only then coverage can be computed
        #diff = prediction.difference(u1I)#Previous code
        diff = u1I.difference(prediction)
        if (len(diff) > 0):
            coverage = float(((len(u1I) - len(diff)) / len(u1I)) * 100)
            #coverage = float((len(diff) / len(prediction)) * 100)#Previous code
            # print("coverage = ",coverage, "%")
        else:
            coverage = 100;
            print("Coverage = ", coverage, "%")
    else:
        coverage = 0
        print("coverage = ", 0, "%")
    return coverage

def computeCoverage2(u1I,prediction) :
    u1I = set(u1I)
    prediction = set(prediction)
    if (len(prediction) != 0):  # If there is some recommendation only then coverage can be computed
        diff = prediction.difference(u1I)#Actual code
        #diff = u1I.difference(prediction)#New code but not finalized
        if (len(diff) > 0):
            coverage = float((len(diff) / len(prediction)) * 100)#Actual code
            #print(" len(diff) = ",len(diff), "len(u1I) = ",len(u1I))
            #coverage =(1- float((len(diff) / len(u1I))) * 100)#New code but not finalized
            # print("coverage = ",coverage, "%")
        else:
            coverage = 100;
            print("Coverage = ", coverage, "%")
    else:
        coverage = 0
        print("coverage = ", 0, "%")
    return coverage
def main(u1I,u1R,Dict) :
    Biclust = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,  25,  26, 27,28,  29,
               30, 31,  32, 33,  34, 35 ]  # , 39, 33, 35, 37]

    env = TwoDGridWorld(6, Biclust,u1I,Dict)
    # check_env(env, warn=True)
    #print("env = ", np.asarray(env.GridPos))
    if(env.majorBreak==-1):
        print("In main(), Recommendations not possible for this user")
        return set(),0,0,0,0,0,0,-1,list()
    else :
        Reset, start = env.reset(env,u1I,u1R)
        if (env.majorBreak == -1):
            print("In main(), Recommendations not possible for this user")
            return set(), 0, 0, 0, 0, 0, 0, -1,list()
        else:
         print("Reset applied Successfully ")
        rend = env.render(env)
        #print("render = ", rend)

        print("Learnning policy ")
        starttime = time.time()
        QTable=learnPolicy(env,u1I,u1R)
        LPtime = round((time.time() - starttime), 2)
        print("Learning policy and Time taken to learn policy = ",LPtime)
        ExPtime = round((time.time() - LPtime), 2)
        Policy=extractPolicy(np.asarray(QTable),env,u1I,u1R)
        print("Applying Actions of Learned policy = ",Policy," Time taken to extract policy = ",ExPtime)
        Apptime = round((time.time() - ExPtime), 2)
        Preditems,Return,state, states_visited=applyPolicy(Policy,env,u1I,u1R)
        print("Time taken to apply policy = ",Apptime)
        TotalTime=LPtime+ExPtime+Apptime

        #coverage = computeCoverage(u1I, Preditems)
        coverage = computeCoverage2(u1I, Preditems)
        #print("coverage for user", u1I, " = ", coverage, "%")
        intersectP = set(Preditems).intersection(set(u1I))
        if (len(intersectP) > 0):
            print(" intersectP = ", intersectP)
            precision = (len(intersectP) / len(set(Preditems))) * 100
            recall = (len(intersectP) / len(set(u1I))) * 100
            Fmeasure = float(2 * precision * recall) / (precision + recall)
            #print("For user ", u1I, 'Precision: ', precision, " Recall = ", recall, " Fmeasure = ", Fmeasure)
        else:
            precision= 0;
            recall = 0;
            Fmeasure = 0
            #print("For user ", u1I, 'Precision: ', precision, " Recall = ", recall, " Fmeasure = ", Fmeasure)
    #print("In main StartSates = ",state)
    return Preditems,coverage,precision,recall,Fmeasure,TotalTime,Return, state, states_visited
#Call to main method which will then call
u1I = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 206, 241, 243, 248, 249, 250, 251, 254, 256, 257]#Biclustcolums0
u2I=[11, 12, 13, 17, 211, 213, 215, 217, 219, 235, 236, 239, 244, 245, 247, 252, 253, 255]#Biclustcolums10

u1R=[2.0, 4.5, 4.0, 3.0, 4.0, 3.0, 4.5, 4.5, 4.0, 4.0, 0.5, 1.0, 2.0, 2.0, 2.5, 3.5, 3.5, 4.0, 4.5, 1.5]
u2R=[1.0, 2.0, 4.5, 3.0, 4.0, 3.0, 1.0, 2.0, 3.0, 1.5, 2.0, 1.0, 2.0, 2.0, 2.5, 3.5, 3.0, 4.5, 4.0, 2.5]
#u3R=[1.0, 2.0, 5.0, 3.0, 4.0, 5.0, 1.0,4.0, 5.0, 1.0]
users=[]
ratings=[]
users.append(u1I)
users.append(u2I)
#users.append(u3I)
ratings.append(u1R)
ratings.append(u2R)
#ratings.append(u3R)

precisioni =[];recalli =[];Fmeasurei=[];coveragei=[];Return=[]


def run (users,ratings) :
    filepath = "amazon test set.csv";#Amazon dataset (User No., Product ID code, Rating)
    #filepath = "F:/Thesis Supervised/Year 2023/NZD/Python NZD1/Amna RL/movielens1.csv";#ML100K dataset
    Dict2,Dict = Amazon_KMeansClustering.KMeans_Clusters(filepath)# For Amazon
    #Dict2, Dict = ML100K_KMeansClustering.KMeans_Clusters(filepath)  # For ML100K
    for i in range(0,len(users)):
        u1I=users[i];u1R=ratings[i]
        prediction,cover,prec,recal,FM,TotalTime,ret,state,states_visited=main(u1I,u1R,Dict);
        coveragei.append(cover)
        precisioni.append(prec)
        recalli.append(recal)
        Fmeasurei.append(FM)
        Return.append(ret)


        print("Recommended or predicted items = ",prediction)
        print("For user with items", u1I)
        print(" coverage = ",cover, 'Precision: ', prec, " Recall = ", recal, " Fmeasure = ", FM,f"Elapsed time: {TotalTime:0.4f} seconds")
    sumcover=sumprec=sumrecal=sumFM=sumReturn=0
    for i in range(0, len(users)):
        sumcover+=coveragei[i]
        sumprec+=precisioni[i]
        sumrecal+=recalli[i]
        sumFM+=Fmeasurei[i]
        sumReturn+=Return[i]
    coverage= float(sumcover/len(users));
    Precision=float(sumprec/len(users));
    Recall=float(sumrecal/len(users));
    Fmeasure=float(sumFM/len(users));
    Rett=float(sumReturn/len(users))
    print(" Final coverage = ",coverage, 'Precision: ', Precision, " Recall = ", Recall, " Fmeasure = ", Fmeasure,"Retrun = ",Rett)
    return coverage,Precision,Recall,Fmeasure,Rett,state,states_visited
def run2 (items,ratings,userid,Dict) :

    #print("items = ",users,"ratings = ",ratings)
    prediction, cover, prec, recal, FM, TotalTime,ret,state, states_visited = main(items, ratings, Dict);
    #prec+=70;
    #recal+=10;
    #FM+=68;
    print("run2 states = ",state )
    print("For user id=", userid, " with items", items)
    #print("Predicted items = ",prediction)
    print(" coverage = ",cover, 'Precision: ', prec, " Recall = ", recal, " Fmeasure = ", FM,f"Elapsed time: {TotalTime:0.4f} seconds")
    return prediction, cover, prec, recal, FM,ret,state,states_visited

#Execution will begin from here, If execution to be started from RLRecommender1.py then must comment
#Comment this if to start from RLRecommender4.py, #Otherwise will double run
#Cover,Prec,Rec,FM,Ret,state,states_visited=run (users,ratings)
#Output on 01/03/2025
# RLRecommender Item coverage = 98.32361782498181 Precision:  1.6763821750182242  Recall = 24.88148973962057  Fmeasure = 2.818529088948922 Return = 0.0  userCoverage=  100.0
# 98.32361782498181
# 1.6763821750182242
# 24.88148973962057
# 2.818529088948922
# 0.0
# 100.0