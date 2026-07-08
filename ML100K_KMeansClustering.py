from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import Listcount
#In this file in for loop at line 72 I am getting movieid, title, userid, rating for respective product cluster
def KMeans_Clusters(filepath):

    data = pd.read_csv(filepath, sep=',', header=0)  # 'my_data.data' 'Ratings.data'
    df = pd.DataFrame(data)

    df_cluster = df[['userId', 'movieId','title','rating']].copy()
    #print("df_cluster = ",df_cluster)
    label_encoder_user = LabelEncoder()
    label_encoder_product = LabelEncoder()

    #Why we need user_id_encode
    df_cluster['user_id_encoded'] = label_encoder_user.fit_transform(df_cluster['userId'].astype(str))
    X_users = df_cluster[['user_id_encoded']]
    df_cluster['product_id_encoded'] = label_encoder_product.fit_transform(df_cluster['movieId'].astype(str))

    X_products = df_cluster[['product_id_encoded']]
    #what is 'user_cluster_36'
    kmeans_36_users = KMeans(n_clusters=36, random_state=42)
    #print("kmeans_36_users = ",kmeans_36_users)
    df_cluster['user_cluster_36'] = kmeans_36_users.fit_predict(X_users)
    #print(" df_cluster['user_cluster_36'] = ",df_cluster['user_cluster_36'])

    kmeans_36_products = KMeans(n_clusters=36, random_state=42)
    df_cluster['product_cluster_36'] = kmeans_36_products.fit_predict(X_products)
    #print("df_cluster['product_cluster_36'] = ", df_cluster['product_cluster_36'])
    #print("df_cluster['product_cluster_36'].values[0] = ",df_cluster['product_cluster_36'] )
    #Find all product ids whose cluster number=23

    #FRating = data.loc[(data['Student_ID'] == y) & (data['Feedback_ID'] == z), 'Feedback_Rating'].values[0]  # Get Feedback Rating for student_ID==1 and Feedback_ID==1

    cluster_product_counts = df_cluster.groupby('product_cluster_36')['movieId'].nunique().reset_index()
    cluster_product_counts.columns = ['Cluster', 'Number of Products']

    cluster_counts = df_cluster.groupby('user_cluster_36')['userId'].nunique().reset_index()
    cluster_counts.columns = ['Cluster', 'Number of Users']

    #print(cluster_counts)
    #######################################3current commented
    clusters = {}
    user_list = []
    user_labels = df_cluster['user_cluster_36']

    for idx, label in enumerate(user_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(int(df_cluster['userId'].iloc[idx]))

    product_labels = df_cluster['product_cluster_36']

    Dictobj_Products = {}
    Dictobj_Users = {}
    x = 0
    sorted_clusters = sorted(df_cluster['product_cluster_36'].unique())
    for cluster in sorted_clusters:#This loop adds priduct cluster to Dictobj_Products
        #products_in_cluster = df_cluster[df_cluster['product_cluster_36'] == cluster]['movieId'].unique()#Original code
        products_in_cluster = df_cluster[df_cluster['product_cluster_36'] == cluster]['movieId']
        title_in_cluster = df_cluster[df_cluster['product_cluster_36'] == cluster]['title']
        userId_in_cluster = df_cluster[df_cluster['product_cluster_36'] == cluster]['userId']
        ratings_in_cluster = df_cluster[df_cluster['product_cluster_36'] == cluster]['rating']
        # here aim is to get items that have more than on repetitions/occurence in products_in_cluster with uniqitems in products_in_cluster and their count
        GreatItemsCount, Uniqitems, Uniqcounts = Listcount.count_items2(products_in_cluster, 3)#Purpose of Listcount.count_items2() function
        #print("movieId,title,userId,rating")
        #In this for loop at line 72 I am getting movieid, title, userid, rating for respective product cluster
        #for i in range (0,len(products_in_cluster)):
        #    print(products_in_cluster.values[i],title_in_cluster.values[i],userId_in_cluster.values[i],ratings_in_cluster.values[i])

        Dictobj_Products[x] = {
            'Cluster_Number': x,
            'Cluster_Items':products_in_cluster.tolist(),
            'Title':title_in_cluster.tolist(),
            'UserId':userId_in_cluster.tolist(),
            'Ratings':ratings_in_cluster.tolist(),
            'Cluster_UniqItems': Uniqitems,
            'GreatItemsCount':GreatItemsCount,
            'Number of Products': len(Uniqitems)
        }
        #print("x=",x,"Cluster_Items = ",Dictobj_Products[x].get('Cluster_Items'));
        #print("x=",x,"Cluster_UniqItems = ",Dictobj_Products[x].get('Cluster_UniqItems'));
        #print("x=",x,"Ratings = ",Dictobj_Products[x].get('Ratings'));
        x += 1
        #print(f"Number of Movies in Cluster {cluster}: {len(products_in_cluster)}\n")

    x=0
    sorted_clusters = sorted(df_cluster['user_cluster_36'].unique())
    for cluster in sorted_clusters:
        users_in_cluster = df_cluster[df_cluster['user_cluster_36'] == cluster]['userId'].unique()
        #print(f"Cluster {cluster},'users',: {users_in_cluster}")

        Dictobj_Users[x] = {
            'Cluster_Number': x,
            'Cluster_Users': users_in_cluster,
            'Number of Users': len(users_in_cluster),

        }
        x += 1
        print(f"Number of Users in Cluster {cluster}: {len(users_in_cluster)}\n")
    return Dictobj_Users,Dictobj_Products,df_cluster
#names = ['movieId',	'title', 'year',	'genres',	'userId',	'rating']
filepath = "F:/Thesis Supervised/Year 2023/NZD/Python NZD1/Amna RL2/Movielens100k.csv";
Dictobj_Users,Dictobj_Products,df_cluster =KMeans_Clusters(filepath)
#Below code is to just print both dictioneries
list_users=[];
list_items=[]
for cluster_label, cluster_data in Dictobj_Products.items():
    print(f"Product Cluster {cluster_label}:")
    print(f"Cluster_Items:\n[{','.join(map(str, cluster_data['Cluster_Items']))}]")
    print(f"Number of Products: {cluster_data['Number of Products']}")
   # 'Title': title_in_cluster.tolist(),
    print(f"Title: {cluster_data['Title']}")
    print(f"'UserId':{cluster_data['UserId']}")
    print(f"'Ratings':{cluster_data['Ratings']}")
    list_items.append(cluster_data['Number of Products'])
    print("\n")
print("Number of products in each cluster = ",list_items," and number of distinct movies = ",sum(list_items))
#############################################3

for cluster_label, cluster_data in Dictobj_Users.items():
    print(f"User Cluster {cluster_label}:")
    print(f"Cluster_Users:\n[{','.join(map(str, cluster_data['Cluster_Users']))}]")
    print(f"Number of Users: {cluster_data['Number of Users']}")
    list_users.append(cluster_data['Number of Users'])
    print("\n")

print("Number of users in each cluster = ",list_users," and sum of users = ",sum(list_users))
#End of Below code to just print both dictioneries
######################currnet commented
#Plot Clusters

plt.figure(figsize=(18, 6))

df_cluster['user_cluster_36'].value_counts().sort_index().plot(kind='bar', color='skyblue')
#Dictobj_Users['Cluster_Users'].value_counts().sort_index().plot(kind='bar', color='skyblue')

plt.xlabel('Cluster')
plt.ylabel('Number of Users')
plt.title('User Distribution Across 36 Clusters')

plt.show()
# Plot Product Clusters
plt.figure(figsize=(18, 6))

df_cluster['product_cluster_36'].value_counts().sort_index().plot(kind='bar', color='black')

plt.xlabel('Cluster')
plt.ylabel('Number of Products')
plt.title('Product Distribution Across 36 Clusters')

plt.show()
cluster_counts =df_cluster.groupby('user_cluster_36')['userId'].nunique().reset_index()
cluster_counts.columns = ['Cluster', 'Number of Users']

print(cluster_counts)

for cluster in df_cluster['user_cluster_36'].unique():
    users_in_cluster = df_cluster[df_cluster['user_cluster_36'] == cluster]['userId'].unique()
    users_in_cluster=list(users_in_cluster)#Mubbashir added code

    print(f"Cluster {cluster}:")
    print(users_in_cluster)
    print(f"Number of Users in Cluster {cluster}: {len(users_in_cluster)}\n")

