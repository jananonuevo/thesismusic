from flask import Flask, session, render_template, request
import pandas as pd
import numpy as np
import pickle
import pandas as pd

import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm import cross_validation

import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

import string
import random

def getSessionKey():
    sessionkey = ''.join(random.choices(string.ascii_letters,k=7))
    return sessionkey

app = Flask(__name__)  # Create the Flask app instance
app.config['SECRET_KEY'] = getSessionKey()

#Load excel DATA
xls = pd.ExcelFile('surveyData(800 ver).xlsx')
songsDF = pd.read_excel(xls, 'songs2')
usersDF = pd.read_excel(xls, 'users2')
interactionsDF = pd.read_excel(xls, 'interactions2')
userpersonalityDF = pd.read_excel(xls, 'userPersonality')
NEW_USER_ID = 'new_user'

usersDF=usersDF.fillna(0)
songsDF=songsDF.fillna(0)
interactionsDF=interactionsDF.fillna(0)

user_features_columns = [
        'country', 'edm', 'indie', 'metal', 'pop', 'pop punk', 'rap',
       'rock', 'singer songwriter', 'soul',
           'extraversion_high',
      'extraversion_average', 'extraversion_low', 'agreeableness_high',
      'agreeableness_average', 'agreeableness_low', 'conscientiousness_high',
      'conscientiousness_average', 'conscientiousness_low',
      'neuroticism_high', 'neuroticism_average', 'neuroticism_low',
      'openness_high', 'openness_average', 'openness_low'
]
item_features_columns = [
        'danceability',
        'energy', 'acousticness', 'instrumentalness', 'liveness', 'valence',
         'edm', 'country', 'indie', 'metal', 'pop', 'pop punk', 'rap', 'rock',
        'singer songwriter', 'soul',
        'extraversion_high',
        'extraversion_average', 'extraversion_low', 'agreeableness_high',
        'agreeableness_average', 'agreeableness_low', 'conscientiousness_high',
        'conscientiousness_average', 'conscientiousness_low',
        'neuroticism_high', 'neuroticism_average', 'neuroticism_low',
        'openness_high', 'openness_average', 'openness_low'
]

def item_feature_generator():
    for i, row in songsDF.iterrows():
        #features =  (pd.Series(row.values[5:-1]) ) .fillna(0)
        features =  (pd.Series(row.values[item_features_columns]) ) .fillna(0)
        yield (row['songID'], features)

def user_feature_generator():
    for i, row in usersDF.iterrows():
        #features =  (pd.Series(row.values[1:]) ) .fillna(0)
        features =  (pd.Series(row.values[user_features_columns]) ) .fillna(0)
        yield (row['userID'], features)

IPPdataset=Dataset()
IPPdataset.fit(
            users=np.append(usersDF['userID'].unique(), NEW_USER_ID),
            items=songsDF['songID'].unique(),
            item_features = item_features_columns,
            user_features= user_features_columns)

user_mappings = IPPdataset.mapping()[0]
user_features_mappings = IPPdataset.mapping()[1] #  user feature map
item_mappings = IPPdataset.mapping()[2] # item id map
item_features_mappings = IPPdataset.mapping()[3] # item feature map

# Create inverse mappings
inv_user_mappings = {v:k for k, v in user_mappings.items()}
inv_user_features_mappings = {v:k for k, v in user_features_mappings.items()}
inv_item_mappings = {v:k for k, v in item_mappings.items()}
inv_item_features_mappings = {v:k for k, v in item_features_mappings.items()}
len(item_mappings),len(item_features_mappings)

item_features_lookup = [(song_id, {column: value for column, value in zip(item_features_columns, features)}) for song_id, *features in songsDF[['songID'] + item_features_columns].itertuples(index=False)]
user_features_lookup = [(user_id, {column: value for column, value in zip(user_features_columns, features)}) for user_id, *features in usersDF[['userID'] +user_features_columns].itertuples(index=False)]
user_features_lookup = user_features_lookup + [(NEW_USER_ID, {feature: 0 for feature in user_features_columns})]

item_features_list = IPPdataset.build_item_features(item_features_lookup,normalize=True)
user_features_list = IPPdataset.build_user_features(user_features_lookup,normalize=True)

#(interactions, weights) = IPPdataset.build_interactions(interactionsDF[['userID','songID','weight']].values)

noIPPdataset=Dataset()
noIPPdataset.fit(
            users=np.append(usersDF['userID'].unique(), NEW_USER_ID),
            items=songsDF['songID'].unique())

#(interactions, weights) = dataset.build_interactions(interactionsDF[['userID','songID']].values)

#Model Setup
with open("lightFM_hybrid.pickle", "rb") as file:
    IPPmodel = pickle.load(file)

with open("lightFM_CF.pickle", "rb") as file:
    noIPPmodel = pickle.load(file)

connect_str = "DefaultEndpointsProtocol=https;AccountName=musicrecommender;AccountKey=xuY+OSYHySZDuIiMnbR5n6u0RvY7cjfqrkUupdG+KVJ5bFmR8N+PSyQPzxktS4SFFfW49mhGu/k0+AStYg8XXw==;EndpointSuffix=core.windows.net"
container_name = "csvs"

def getCSV(connect_str, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)

    blob_client = container_client.get_blob_client(blob_name)

    download_stream = blob_client.download_blob()
    df = pd.read_csv(download_stream)

    return df

def append_row_to_csv_blob(connect_str, container_name, blob_name, new_row):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)

    blob_client = container_client.get_blob_client(blob_name)

    download_stream = blob_client.download_blob()
    df = pd.read_csv(download_stream)
    new_row = pd.DataFrame([new_row])
    new_df = pd.concat([df, new_row], ignore_index=True)

    csv_data = new_df.to_csv(index=False)

    blob_client.upload_blob(csv_data, overwrite=True)


def reverseScore(r):
    if(r == 5):
        r = 1
    elif(r == 4):
        r = 2
    elif(r == 3):
        r = 3
    elif(r == 2):
        r = 4
    elif(r == 1):
        r = 5
    return r    

def computePersonalityScore(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, top_genres_user):
    session['df_new_users']['extraversion'] = (reverseScore(q1) + q6) / 2
    session['df_new_users']['agreeableness'] = (q2 + reverseScore(q7)) / 2
    session['df_new_users']['openness'] = (reverseScore(q5) + q10) / 2
    session['df_new_users']['conscientiousness'] = (reverseScore(q3) + q8) / 2
    session['df_new_users']['neuroticism'] = (reverseScore(q4) + q9) / 2

    session['df_new_users']['selfrating_E'] = (((((session['df_new_users']['extraversion']) - pd.concat([userpersonalityDF['x(Extraversion)'], pd.Series([session['df_new_users']['extraversion']])]).mean()) / (pd.concat([userpersonalityDF['x(Extraversion)'], pd.Series([session['df_new_users']['extraversion']])]).std()) * 10) + 50))
    session['df_new_users']['selfrating_A'] = (((((session['df_new_users']['agreeableness']) - pd.concat([userpersonalityDF['x(Agreeableness)'], pd.Series([session['df_new_users']['agreeableness']])]).mean()) / (pd.concat([userpersonalityDF['x(Agreeableness)'], pd.Series([session['df_new_users']['agreeableness']])]).std()) * 10) + 50))
    session['df_new_users']['selfrating_O'] = (((((session['df_new_users']['openness']) - pd.concat([userpersonalityDF['x(Openness)'], pd.Series([session['df_new_users']['openness']])]).mean()) / (pd.concat([userpersonalityDF['x(Openness)'], pd.Series([session['df_new_users']['openness']])]).std()) * 10) + 50))
    session['df_new_users']['selfrating_C'] = (((((session['df_new_users']['conscientiousness']) - pd.concat([userpersonalityDF['x(Conscientiousness)'], pd.Series([session['df_new_users']['conscientiousness']])]).mean()) / (pd.concat([userpersonalityDF['x(Conscientiousness)'], pd.Series([session['df_new_users']['conscientiousness']])]).std()) * 10) + 50))
    session['df_new_users']['selfrating_N'] = (((((session['df_new_users']['neuroticism']) - pd.concat([userpersonalityDF['x(Neuroticism)'], pd.Series([session['df_new_users']['neuroticism']])]).mean()) / (pd.concat([userpersonalityDF['x(Neuroticism)'], pd.Series([session['df_new_users']['neuroticism']])]).std()) * 10) + 50))

    if session['df_new_users']['selfrating_E'] < 45:
        session['df_new_users']['extraversion_low'] = 1
    elif session['df_new_users']['selfrating_E'] > 55:
        session['df_new_users']['extraversion_high'] = 1
    else: 
        session['df_new_users']['extraversion_average'] = 1

    if session['df_new_users']['selfrating_A'] < 45:
        session['df_new_users']['agreeableness_low'] = 1
    elif session['df_new_users']['selfrating_A'] > 55:
        session['df_new_users']['agreeableness_high'] = 1
    else: 
        session['df_new_users']['agreeableness_average'] = 1

    if session['df_new_users']['selfrating_O'] < 45:
        session['df_new_users']['openness_low'] = 1
    elif session['df_new_users']['selfrating_O'] > 55:
        session['df_new_users']['openness_high'] = 1
    else:
        session['df_new_users']['openness_average'] = 1

    if session['df_new_users']['selfrating_C'] < 45:
        session['df_new_users']['conscientiousness_low'] = 1
    elif session['df_new_users']['selfrating_C'] > 55:
        session['df_new_users']['conscientiousness_high'] = 1
    else:
        session['df_new_users']['conscientiousness_average'] = 1

    if session['df_new_users']['selfrating_N'] < 45:
        session['df_new_users']['neuroticism_low'] = 1
    elif session['df_new_users']['selfrating_N'] > 55:
        session['df_new_users']['neuroticism_high'] = 1
    else:
        session['df_new_users']['neuroticism_average'] = 1

    for i in range(len(top_genres_user)):
        if top_genres_user[i] == 'edm':
            session['df_new_users']['edm'] = 1
        elif top_genres_user[i] == 'country':
            session['df_new_users']['country'] = 1
        elif top_genres_user[i] == 'indie':
            session['df_new_users']['indie'] = 1
        elif top_genres_user[i] == 'metal':
            session['df_new_users']['metal'] = 1
        elif top_genres_user[i] == 'pop':
            session['df_new_users']['pop'] = 1
        elif top_genres_user[i] == 'poppunk':
            session['df_new_users']['pop punk'] = 1
        elif top_genres_user[i] == 'rap':
            session['df_new_users']['rap'] = 1
        elif top_genres_user[i] == 'rock':
            session['df_new_users']['rock'] = 1
        elif top_genres_user[i] == 'singersongwriter':
            session['df_new_users']['singer songwriter'] = 1
        elif top_genres_user[i] == 'soul':
            session['df_new_users']['soul'] = 1

    session['df_personality_profile'] =  {
            'extraversion_high': session['df_new_users']['extraversion_high'],
            'extraversion_average': session['df_new_users']['extraversion_average'],
            'extraversion_low': session['df_new_users']['extraversion_low'],
            'agreeableness_high': session['df_new_users']['agreeableness_high'],
            'agreeableness_average': session['df_new_users']['agreeableness_average'],
            'agreeableness_low': session['df_new_users']['agreeableness_low'],
            'neuroticism_high': session['df_new_users']['neuroticism_high'],
            'neuroticism_average': session['df_new_users']['neuroticism_average'],
            'neuroticism_low': session['df_new_users']['neuroticism_low'],
            'conscientiousness_high': session['df_new_users']['conscientiousness_high'],
            'conscientiousness_average': session['df_new_users']['conscientiousness_average'],
            'conscientiousness_low': session['df_new_users']['conscientiousness_low'],
            'openness_high': session['df_new_users']['openness_high'],
            'openness_average': session['df_new_users']['openness_average'],
            'openness_low': session['df_new_users']['openness_low'],
            'edm': session['df_new_users']['edm'],
            'country': session['df_new_users']['country'],
            'indie': session['df_new_users']['indie'],
            'metal': session['df_new_users']['metal'],
            'pop': session['df_new_users']['pop'],
            'pop punk': session['df_new_users']['pop punk'],
            'rap': session['df_new_users']['rap'],
            'rock': session['df_new_users']['rock'],
            'singer songwriter': session['df_new_users']['singer songwriter'],
            'soul': session['df_new_users']['soul']
            }
    print (session['df_personality_profile'])   

    return session['df_personality_profile']

def getLargestNumber():
    new_users = getCSV(connect_str, container_name, "new_users.csv")
    userID = new_users['userID'].tolist()
    userID_remove_duplicates = list(set(userID))
    userID_remove_U = [element.replace('U', '') for element in userID_remove_duplicates]
    userID_convert_int = [int(item) for item in userID_remove_U if item.isdigit()]
    userID_largest_number = max(userID_convert_int)
    return userID_largest_number

def newUserRecommendation(model, model_type, dataset, userID=None, new_user_feature=None, k=10):
    userID = 'new_user'
    mapper_to_internal_ids = dataset.mapping()[2]
    mapper_to_external_ids = {v: k for k, v in mapper_to_internal_ids.items()}

    userID_map = dataset.mapping()[0][userID]
    item_ids = list(mapper_to_internal_ids.values())

    if model_type == 'A':
        new_user_features_lookup = [(userID, new_user_feature)]
        new_user_feature_matrix = dataset.build_user_features(new_user_features_lookup, normalize=False)
        scores = model.predict(userID_map, item_ids, user_features=new_user_feature_matrix, item_features=item_features_list)
        print("Model A Scores\n")
        print(scores)
    else:
        scores = model.predict(userID_map, list(mapper_to_internal_ids.values()))
        print("Model B Scores\n")
        print(scores)


    top_k_indices = np.argsort(-scores)[:k]
    recommended_song_ids = np.vectorize(mapper_to_external_ids.get)(top_k_indices)
    songs_recommended = songsDF[songsDF['songID'].isin(recommended_song_ids)]

    print(songs_recommended[['songID', 'name', 'artists']])
    recommended_song_ids = np.vectorize(mapper_to_external_ids.get)(top_k_indices)
    songs_recommended = songsDF[songsDF['songID'].isin(recommended_song_ids)]

    for row, i in zip(songs_recommended.itertuples(), range(1, 11)):
        song_id = row.songID
        df_col = str(model_type) +'top' +str(i)
        session['df_new_users'][df_col] = song_id
    
    print("User: " +session['df_new_users']['name'] +"\nModel " +model_type +"\n")
    print(songs_recommended[['songID', 'name', 'artists']])
    
    return songs_recommended[['songID', 'name', 'artists']]

@app.route('/')
def index():
    session.clear()
    
    session['df_new_users'] =  {
            'userID': '',
            'name': '',
            'studentidno': '',
            'email': '',
            'educationallvl': '',
            'coursestrand': '',
            'q1':0,
            'q2':0,
            'q3':0,
            'q4':0,
            'q5':0,
            'q6':0,
            'q7':0,
            'q8':0,
            'q9':0,
            'q10':0,
            'extraversion_high': 0,
            'extraversion_average': 0,
            'extraversion_low': 0,
            'agreeableness_high': 0,
            'agreeableness_average': 0,
            'agreeableness_low': 0,
            'neuroticism_high': 0,
            'neuroticism_average': 0,
            'neuroticism_low': 0,
            'conscientiousness_high': 0,
            'conscientiousness_average': 0,
            'conscientiousness_low': 0,
            'openness_high': 0,
            'openness_average': 0,
            'openness_low': 0,
            'edm': 0,
            'country': 0,
            'indie': 0,
            'metal': 0,
            'pop': 0,
            'pop punk': 0,
            'rap': 0,
            'rock': 0,
            'singer songwriter': 0,
            'soul': 0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'openness': 0.0,
            'conscientiousness': 0.0,
            'neuroticism': 0.0,
            'selfrating_E': 0.0,
            'selfrating_A': 0.0,
            'selfrating_O': 0.0,
            'selfrating_C': 0.0,
            'selfrating_N': 0.0,
            'Atop1': '',
            'Atop2': '',
            'Atop3': '',
            'Atop4': '',
            'Atop5': '',
            'Atop6': '',
            'Atop7': '',
            'Atop8': '',
            'Atop9': '',
            'Atop10': '',
            'Btop1': '',
            'Btop2': '',
            'Btop3': '',
            'Btop4': '',
            'Btop5': '',
            'Btop6': '',
            'Btop7': '',
            'Btop8': '',
            'Btop9': '',
            'Btop10': ''
            } 
        
    return render_template('index.html')

@app.route("/", methods=['POST'])
def get_recos():
    if 'getrecos' in request.form:
        with app.app_context():
            session['userID'] = getLargestNumber() + 1
            session['df_new_users']['userID'] = 'U' +str(session['userID'])

            session['top_genres_user'] = request.form.getlist('genre')
            
            session['df_new_users']['q1'] = int(request.form['q1'])
            session['df_new_users']['q2'] = int(request.form['q2'])
            session['df_new_users']['q3'] = int(request.form['q3'])
            session['df_new_users']['q4'] = int(request.form['q4'])
            session['df_new_users']['q5'] = int(request.form['q5'])
            session['df_new_users']['q6'] = int(request.form['q6'])
            session['df_new_users']['q7'] = int(request.form['q7'])
            session['df_new_users']['q8'] = int(request.form['q8'])
            session['df_new_users']['q9'] = int(request.form['q9'])
            session['df_new_users']['q10'] = int(request.form['q10'])
            session['df_new_users']['name'] = request.form['name']
            session['df_new_users']['studentidno'] = request.form['idno']
            session['df_new_users']['email'] = request.form['email']
            session['df_new_users']['educationallvl'] = request.form['edu']
            session['df_new_users']['coursestrand'] = request.form['coursestrand']

            recommendations = newUserRecommendation(IPPmodel, 'A', IPPdataset, new_user_feature=computePersonalityScore(session['df_new_users']['q1'],session['df_new_users']['q2'],session['df_new_users']['q3'],session['df_new_users']['q4'],session['df_new_users']['q5'],session['df_new_users']['q6'],session['df_new_users']['q7'],session['df_new_users']['q8'],session['df_new_users']['q9'],session['df_new_users']['q10'],session['top_genres_user']))
            
            recommendations2 = newUserRecommendation(noIPPmodel, 'B', noIPPdataset, new_user_feature=computePersonalityScore(session['df_new_users']['q1'],session['df_new_users']['q2'],session['df_new_users']['q3'],session['df_new_users']['q4'],session['df_new_users']['q5'],session['df_new_users']['q6'],session['df_new_users']['q7'],session['df_new_users']['q8'],session['df_new_users']['q9'],session['df_new_users']['q10'],session['top_genres_user']))

            session['htmltable_string'] = """
            <h4 class='text-center'> Model A Results </h4>
            <table class='table table-striped table-dark'> 
                <thead>
                    <tr> 
                        <th scope='col'>Rank</th>
                        <th scope='col'>Name</th> 
                        <th scope='col'>Artist</th> 
                        <th scope='col'>Spotify Preview</th>
                        <th scope='col'>Like/Dislike</th> 
                    </tr>
                </thead>
                <tbody>
            """

            for row, i in zip(recommendations.itertuples(), range(1, 11)):
                session['htmltable_string'] += f"""                                               
                    <tr>
                        <th scope='row'>{i}</th> 
                        <td>{row.name}</td>
                        <td>{row.artists}</td>
                        <td> <a href='https://open.spotify.com/track/{row.songID}' target='_blank'> Link to Spotify </a> </td>
                        <td> <input class='form-check-input' type='checkbox' name='like_atop{i}' id='defaultCheck1'></td>
                    </tr>
                    """
            session['htmltable_string'] += "</tbody></table><br><br>"


            session['htmltable_string'] += """
            <h4 class='text-center'> Model B Results </h4>
            <table class='table table-striped table-dark'> 
                <thead>
                    <tr> 
                        <th scope='col'>Rank</th>
                        <th scope='col'>Name</th> 
                        <th scope='col'>Artist</th> 
                        <th scope='col'>Spotify Preview</th> 
                        <th scope='col'>Like/Dislike</th> 
                    </tr>
                </thead>
                <tbody>
            """
           
            
            for row, i in zip(recommendations2.itertuples(), range(1, 11)):
                session['htmltable_string'] += """
                <tr> 
                    <th scope='row'>""" +str(i) +"""</th>
                    <td>""" +row.name +"""</td> 
                    <td>""" +row.artists +"""</td> 
                    <td> <a href='https://open.spotify.com/track/""" +row.songID +"""' target='_blank'> Link to Spotify </a> </td> 
                    <td> <input class='form-check-input' type='checkbox' name='like_btop""" +str(i) +"""' id='defaultCheck1'></td>
                </tr>"""
            session['htmltable_string'] += "</tbody></table></form>"
            
            
            append_row_to_csv_blob(connect_str, container_name, "new_users.csv", session['df_new_users'])

            return render_template("results.html", htmlstrr=session['htmltable_string'])
    elif 'getsurvey' in request.form:
        session['df_newusers_surveyresults'] =  {
            'userID': session['df_new_users']['userID'],
            'aq1': int(request.form['aq1']),
            'aq2': int(request.form['aq2']),
            'aq3': int(request.form['aq3']),
            'bq1': int(request.form['bq1']),
            'bq2': int(request.form['bq2']),
            'bq3': int(request.form['bq3']),
            'lcq1': request.form['lcq1'],
            'lcq2': request.form['lcq2'],
            'like_atop1': '1' if request.form.get('like_atop1') else '0',
            'like_atop2': '1' if request.form.get('like_atop2') else '0',
            'like_atop3': '1' if request.form.get('like_atop3') else '0',
            'like_atop4': '1' if request.form.get('like_atop4') else '0',
            'like_atop5': '1' if request.form.get('like_atop5') else '0',
            'like_atop6': '1' if request.form.get('like_atop6') else '0',
            'like_atop7': '1' if request.form.get('like_atop7') else '0',
            'like_atop8': '1' if request.form.get('like_atop8') else '0',
            'like_atop9': '1' if request.form.get('like_atop9') else '0',
            'like_atop10': '1' if request.form.get('like_atop10') else '0',
            'like_btop1': '1' if request.form.get('like_btop1') else '0',
            'like_btop2': '1' if request.form.get('like_btop2') else '0',
            'like_btop3': '1' if request.form.get('like_btop3') else '0',
            'like_btop4': '1' if request.form.get('like_btop4') else '0',
            'like_btop5': '1' if request.form.get('like_btop5') else '0',
            'like_btop6': '1' if request.form.get('like_btop6') else '0',
            'like_btop7': '1' if request.form.get('like_btop7') else '0',
            'like_btop8': '1' if request.form.get('like_btop8') else '0',
            'like_btop9': '1' if request.form.get('like_btop9') else '0',
            'like_btop10': '1' if request.form.get('like_btop10') else '0'
            }  
        append_row_to_csv_blob(connect_str, container_name, "newusers_surveyresults.csv", session['df_newusers_surveyresults'])
        return render_template("index.html")

if __name__ == "__main__":
  app.run(debug=True)

