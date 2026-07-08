import os
import numpy as np
import pandas as pd

def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
def read_Dataset():
    #data_dir = d2l.download_extract('ml-100k')
    names = ['Product_ID', 'Product ID', 'Category','Discounted price','Rating','Rating count','UID','User ID','product_id_encoded','product_cluster_36','comma_separated_user_ids','comma_separated_product_ids_code']

    data_dir = "F:\\Thesis Supervised\\Year 2023\\NZD\\Python NZD1\\Amna Obaid";
    #data_dir="amazon cleaned updated.csv";
    data = pd.read_csv(os.path.join(data_dir, 'amazon cleaned updated amna.csv'), sep=',', names=names,engine='python', header=0)# 'my_data.data' 'Ratings.data'
    #print(data)
    num_users = data.UID.unique().shape[0]
    list_users = data.UID.unique()
    num_items = data.Product_ID.unique().shape[0]
    list_items=data.Product_ID.unique()


    return data, num_users, num_items,list_users,list_items
def load_Dataset(data, num_users, num_items):
    print("in load_Dataset method")
    users, items, scores = [], [], []
    UIMat = np.zeros((num_users,num_items ))  #BinaryMat will be a matrix having rows=num_users and columns=num_items, all filled with zeros

    for line in data.itertuples():
        user_index, item_index = int(line[7]-1), int(line[1]-1)
        score = float(line[5])
        users.append(user_index)
        items.append(item_index)
        #print(user_index, item_index)
        scores.append(score)
        UIMat[user_index, item_index] = score
        # if score>=1.0:
        #     BinaryMat[user_index,item_index ] = 1
        # else:
        #     BinaryMat[user_index,item_index ] = 0 #BinaryMat here is user-item ratings matrix with non-rated movies filled as zeros
        #print(user_index+1,item_index+1,score)
        #print(users, items)
    #print(BinaryMat)
    return users, items, scores, UIMat,user_index,item_index

#Demo block below only runs when this file is executed directly, not on import
#(previously ran unconditionally at import time). It also requires the full-featured
#Amazon file (Category, Discounted price, Rating count, etc.) which isn't part of this
#repo -- only 'amazon test set.csv' (User No., Product ID code, Rating) is available, so
#read_Dataset()'s hardcoded path/schema below won't resolve here without that file.
if __name__ == '__main__':
    data, num_users, num_items, list_users, list_items = read_Dataset()
    users, items, scores, BinaryMat, user_index, item_index = load_Dataset(data, num_users, num_items)
