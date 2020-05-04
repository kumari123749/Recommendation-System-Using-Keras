# Recommendation-System-Using-Keras
1. importing the required libraries
import numpy as np
import pandas as pd
import pickle
import matrix_factorization_utilities
import scipy.sparse as sp
from scipy.sparse.linalg import svds
2. Reading the ratings data
ratings = pd.read_csv(r'C:\Users\kumari\Desktop\fintech\AI & ML\Movie-Recommendation-System-master\Dataset\ratings.csv')
3. Checking if the user has rated the same movie twice, in that case we just take max of them
ratings_df = ratings.groupby(['userId','movieId']).aggregate(np.max)
4. Getting the percentage count of each rating value 
count_ratings = ratings.groupby('rating').count()
count_ratings['perc_total']=round(count_ratings['userId']*100/count_ratings['userId'].sum(),1)
count_ratings
5. Visualising the percentage total for each rating
count_ratings['perc_total'].plot.bar()
6. reading the movies dataset
movie_list = pd.read_csv(r'C:\Users\kumari\Desktop\fintech\AI&ML\Movie-Recommendation-System-master\Dataset\movies.csv')
7. reading the tags datast
tags = pd.read_csv(r'C:\Users\kumari\Desktop\fintech\AI&ML\Movie-Recommendation-System-master\Dataset\tags.csv')
8. inspecting various genres
genres = movie_list['genres']
genre_list = ""for index,row in movie_list.iterrows(): genre_list += row.genres + "|"
9. split the string into a list of values
genre_list_split = genre_list.split('|')
10. join the movie indices
df_with_index = pd.merge(ratings,item_indices,on='movieId')
11. join the user indices
df_with_index=pd.merge(df_with_index,user_indices,on='userId')
12. inspec the data frame
df_with_index.head()

13. import train_test_split module
from sklearn.model_selection import train_test_split
14. take 80% as the training set and 20% as the test set
df_train, df_test= train_test_split(df_with_index,test_size=0.2)
print(len(df_train))
print(len(df_test))

df_train.head()
df_test.head()
n_users = ratings.userId.unique().shape[0]
n_items = ratings.movieId.unique().shape[0]
print(n_users)
print(n_items)
15. Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
#for every line in the data
for line in df_train.itertuples():
#set the value in the column and row to 
#line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    train_data_matrix[line[5], line[4]] = line[3]
train_data_matrix.shape

16. Create two user-item matrices, one for training and another for testing
test_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_test[:1].itertuples():
    #set the value in the column and row to 
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    #print(line[2])
    test_data_matrix[line[5], line[4]] = line[3]
    #train_data_matrix[line['movieId'], line['userId']] = line['rating']
test_data_matrix.shape

pd.DataFrame(train_data_matrix).head()
df_train['rating'].max()
17 Calculate the rmse sscore of SVD using different values of k (latent features)
rmse_list = []
#for i in [1,2,5,20,40,60,100,200]:
    #apply svd to the test data
    u,s,vt = svds(train_data_matrix,k=i)
    #get diagonal matrix
    s_diag_matrix=np.diag(s)
    #predict x with dot product of u s_diag and vt
    X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
#calculate rmse score of matrix factorisation predictions
18. Convert predictions to a DataFrame
mf_pred = pd.DataFrame(X_pred)
mf_pred.head()
df_names = pd.merge(ratings,movie_list,on='movieId')
df_names.head()
19. choose a user ID
user_id = int(input('Enter User Id'))
#get movies rated by this user id
users_movies = df_names.loc[df_names["userId"]==user_id]
20. print how many ratings user has made 
print("User ID : " + str(user_id) + " has already rated " + str(len(users_movies)) + " movies")
#list movies that have been rated
users_movies

