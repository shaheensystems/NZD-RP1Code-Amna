from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import FeatureEngineering as FE
#Amazon dataset contains 1191 users and 1348 distinct product ids
def KMeans_Clusters(filepath):
    data = pd.read_csv(filepath, header=0, names=['User No.', 'Product ID code', 'Rating'])
    df = pd.DataFrame(data)

    df_cluster = df[['User No.', 'Product ID code']].copy()

    label_encoder_user = LabelEncoder()
    label_encoder_product = LabelEncoder()

    def space_to_comma(val):
        if isinstance(val, str):
            return val.replace(" ", ",")
        return str(val)
    #Behavioral features: cluster on rating statistics (mean/count/std) per user and
    #per product instead of the arbitrary label-encoded ID, so clusters reflect
    #interaction behavior rather than ID order.
    X_users, user_ids, X_products, product_ids = FE.amazon_features(df, rating_col='Rating')

    df['comma_separated_User_No.'] = df['User No.'].apply(space_to_comma)
    df['comma_separated_Product_ID_code'] = df['Product ID code'].apply(space_to_comma)

    kmeans_36_users = KMeans(n_clusters=36, random_state=42, n_init=10)
    user_labels = kmeans_36_users.fit_predict(X_users)
    user_cluster_map = dict(zip(user_ids, user_labels))
    df_cluster['user_cluster_36'] = df_cluster['User No.'].map(user_cluster_map)

    kmeans_36_products = KMeans(n_clusters=36, random_state=42, n_init=10)
    product_labels = kmeans_36_products.fit_predict(X_products)
    product_cluster_map = dict(zip(product_ids, product_labels))
    df_cluster['product_cluster_36'] = df_cluster['Product ID code'].map(product_cluster_map)
    df['product_cluster_36'] = df['Product ID code'].map(product_cluster_map)

    #print(df_cluster[['User No.', 'user_cluster_36']].head(1462))

    #print(df[['comma_separated_User_No.', 'comma_separated_Product_ID_code']].head(1462))

    cluster_counts = df_cluster.groupby('user_cluster_36')['User No.'].nunique().reset_index()
    cluster_counts.columns = ['Cluster', 'Number of Users']

    #print(cluster_counts)
    #######################################3current commented
    clusters = {}
    user_list = []
    user_labels = df_cluster['user_cluster_36']

    for idx, label in enumerate(user_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(int(df_cluster['User No.'].iloc[idx]))


    for cluster_label, points in clusters.items():
        comma_separated_points = ",".join(map(str, points))
        #print(f"Cluster {cluster_label}:{comma_separated_points}")
        lst=[comma_separated_points]
        lst = lst[0].split(',')
        lst = [int(x) for x in lst]
        user_list.extend(lst)

    cluster_product_counts = df.groupby('product_cluster_36')['Product ID code'].nunique().reset_index()
    cluster_product_counts.columns = ['Cluster', 'Number of Products']
    #print(cluster_product_counts)

    product_labels = df_cluster['product_cluster_36']

    Dictobj_Products = {}
    Dictobj_Users = {}
    x = 0
    sorted_clusters = sorted(df['product_cluster_36'].unique())
    for cluster in sorted_clusters:#This loop adds priduct cluster to Dictobj_Products
        products_in_cluster = df_cluster[df['product_cluster_36'] == cluster]['Product ID code'].unique()

        Dictobj_Products[x] = {
            'Cluster_Number': x,
            'Cluster_Items': products_in_cluster,
            'Number of Products': len(products_in_cluster)
        }
        x += 1
        print(f"Number of Products in Cluster {cluster}: {len(products_in_cluster)}\n")
    x=0
    sorted_clusters = sorted(df_cluster['user_cluster_36'].unique())
    for cluster in sorted_clusters:
        users_in_cluster = df_cluster[df_cluster['user_cluster_36'] == cluster]['User No.'].unique()
        #print(f"Cluster {cluster},'users',: {users_in_cluster}")

        Dictobj_Users[x] = {
            'Cluster_Number': x,
            'Cluster_Users': users_in_cluster,
            'Number of Users': len(users_in_cluster),

        }
        x += 1
        #print(f"Number of Users in Cluster {cluster}: {len(users_in_cluster)}\n")
    return Dictobj_Users,Dictobj_Products
filepath='amazon cleaned updated amna.csv'
#Dictobj_Users,Dictobj_Products =KMeans_Clusters(filepath)
#Below code is to just print both dictioneries
# list_users=[];
# list_items=[]
# for cluster_label, cluster_data in Dictobj_Products.items():
#     print(f"Product Cluster {cluster_label}:")
#     print(f"Cluster_Items:\n[{','.join(map(str, cluster_data['Cluster_Items']))}]")
#     print(f"Number of Products: {cluster_data['Number of Products']}")
#     list_items.append(cluster_data['Number of Products'])
#     print("\n")
# print("Number of products in each cluster = ",list_items," and sum of ietms = ",sum(list_items))
# #############################################3
#
# for cluster_label, cluster_data in Dictobj_Users.items():
#     print(f"User Cluster {cluster_label}:")
#     print(f"Cluster_Users:\n[{','.join(map(str, cluster_data['Cluster_Users']))}]")
#     print(f"Number of Users: {cluster_data['Number of Users']}")
#     list_users.append(cluster_data['Number of Users'])
#     print("\n")
#
# print("Number of users in each cluster = ",list_users," and sum of users = ",sum(list_users))
#End of Below code to just print both dictioneries
#######################currnet commented
#########################################################3
########################################################33
#Plot Clusters
#
# plt.figure(figsize=(18, 6))
#
# df_cluster['user_cluster_36'].value_counts().sort_index().plot(kind='bar', color='skyblue')
#
#
# plt.xlabel('Cluster')
# plt.ylabel('Number of Users')
# plt.title('User Distribution Across 36 Clusters')
#
# plt.show()
#
# cluster_counts = df_cluster.groupby('user_cluster_36')['User No.'].nunique().reset_index()
#
# cluster_counts.columns = ['Cluster', 'Number of Users']
#
# print(cluster_counts)
#
# for cluster in df_cluster['user_cluster_36'].unique():
#     users_in_cluster = df_cluster[df_cluster['user_cluster_36'] == cluster]['User No.'].unique()
#     users_in_cluster=list(users_in_cluster)#Mubbashir added code
#
#     print(f"Cluster {cluster}:")
#     print(users_in_cluster)
#     print(f"Number of Users in Cluster {cluster}: {len(users_in_cluster)}\n")
#
