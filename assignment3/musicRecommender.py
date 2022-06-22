# get arguments from command line
import sys
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import requests
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def main(args) -> None:
    """ Main function to be called when the script is run from the command line. 
    This function will recommend songs based on the user's input and save the
    playlist to a csv file.
    
    Parameters
    ----------
    args: list 
        list of arguments from the command line
    Returns
    -------
    None
    """
    arg_list = args[1:]
    if len(arg_list) == 0:
        print("Usage: python3 musicRecommender.py <csv file>")
        sys.exit()
    else:
        file_name = arg_list[0]
        print(file_name)
        if not os.path.isfile(file_name):
            print("File does not exist")
            sys.exit()
        else:
            userPreferences = pd.read_csv(file_name)


    # this code is just to check, delete later.
    print(userPreferences.head())
    def get_playlist_tracks(playlist_id, sample, token):
        """
        Returns a list of tracks from a playlist
        """
        url = "https://api.spotify.com/v1/playlists/"+playlist_id
        headers = {"Accept": "application/json", "Content-Type": "application/json" ,'Authorization': "Bearer {}".format(token)}
        r = requests.get(url, headers=headers)
        r = r.json()

        response_df = pd.json_normalize(r['tracks']["items"])
        response_df.shape
        
        requested_tracks = ""

        for i in response_df.sample(sample)['track.id'].to_list():
            requested_tracks += i + ","

        return requested_tracks

    def get_audio_features(track_id_list, token):
        """
        Returns a list of audio features for a list of tracks
        """

        url = "https://api.spotify.com/v1/audio-features/?ids=" + track_id_list
        headers = {"Accept": "application/json", "Content-Type": "application/json" ,'Authorization': "Bearer {}".format(token)}
        r = requests.get(url, headers=headers)
        r = r.json()
        response_df = pd.json_normalize(r['audio_features'])
        return response_df

    def get_token():
        client_id = '3090a5e2b0ae4b528f5df74a13f1277a'
        client_secret = 'afb00b552775410e889212d4c8831347'
        r = requests.post(
            'https://accounts.spotify.com/api/token',
            data={'grant_type': 'client_credentials'},
                headers={'Authorization':'Basic '+ base64.urlsafe_b64encode((client_id + ':' + client_secret).encode('ascii')).decode('ascii')}
        )
        return  r.json().get('access_token')

    token = get_token()
    sample = 34
    playlist_id = "3QKro1eeWtDypIxIVj3qC0"
    req_tracks = get_playlist_tracks(playlist_id, sample, token)
    mix = get_audio_features(req_tracks, token)

    print(mix.shape)

    print(df_genres.shape)

    df_genres.info()

    mix.info()

    """# **Data Cleaning**"""

    missing_values_count = df_genres.isnull().sum()
    print(missing_values_count)

    df_genres.track_href.value_counts()

    df_genres.id.value_counts()

    df_genres.song_name.value_counts()

    df_genres.analysis_url.value_counts()

    df_genres  = df_genres.drop(columns=['song_name', 'title', 'analysis_url', 'track_href', 'uri', 'id', 'type', 'Unnamed: 0', 'genre'])

    input = mix.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url'])

    df_genres.duplicated().sum()

    df_genres.drop_duplicates(inplace=True)
    df_genres.duplicated().sum()

    print(df_genres.shape)
    print(input.shape)

    dataset = pd.concat([df_genres,input], ignore_index=True)

    print(dataset.shape)

    dataset.info()

    dataset

    """# **Data Visualization**"""

    dataset.hist(layout=(7,2),figsize=(30,25), color='#E433FF');

    musicSkewness = dataset.skew(axis=0)

    musicSkewness = np.round(musicSkewness,decimals=2)

    def FindSkewness(value):
        if value > 0: 
            return 'Positive Skewness'
        elif value < 0:
            return'Negative Skewness'
        return 'No Skewness'

    numeric_cols  = dataset._get_numeric_data().columns.tolist()

    for i, column in enumerate(numeric_cols):
        sns.histplot(dataset[column],bins = 20,kde = True);
        conclusion = FindSkewness(musicSkewness[i])
        plt.title(conclusion)
        plt.figtext(0.45, -0.05 ,f"skewness : {musicSkewness[i]}")
        plt.show()

    corrilation_data = dataset.corr()
    corrilation_data

    plt.figure (figsize=(20, 12))
    sns.heatmap(corrilation_data,annot=True);

    """# **Clustring**"""

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(dataset)
    normalized_data = scaler.transform(dataset)
    normalized_data = pd.DataFrame(normalized_data, index=dataset.index, columns=dataset.columns)

    for col in normalized_data.columns:
        if normalized_data[col].dtype == 'float64':
            normalized_data[col] = normalized_data[col].astype('float32')

    normalized_data.dtypes

    kmeans = KMeans(n_clusters = 10 , random_state = 7).fit(normalized_data)

    temp = pd.DataFrame()
    temp['labels'] = kmeans.labels_

    temp.labels.value_counts()

    input_labels = temp.labels.tail(34)

    normalized_data['label'] = kmeans.labels_
    normalized_df_genres = normalized_data[0:35096]

    from sklearn import decomposition
    pca = decomposition.PCA(random_state=5)
    pca.n_components =2
    pca_data = pca.fit_transform(normalized_data.drop(columns='label'))

    pca_df = pd.DataFrame(pca_data, columns = ['p1','p2'])
    pca_df['label'] = normalized_data.label

    sns.FacetGrid(pca_df, hue='label', size=6).map(plt.scatter, 'p1', 'p2').add_legend()

    def plot_label_horizontal_bar(col, title=None):
        data = normalized_data.groupby('label')[col].mean().sort_values()

        cmap = plt.cm.coolwarm_r
        norm = plt.Normalize(vmin=data.min(), vmax=data.max())
        colors = [cmap(norm(value)) for value in data]

        data.plot.barh(color=colors)
        plt.xlabel(col)
        plt.title(title, fontdict={'size': 10, 'color': '#de5d83'})
        plt.show()

    sns.boxplot(data=normalized_data, x='label', y='danceability')
    plt.xticks(rotation=30)
    plt.show()

    plot_label_horizontal_bar('danceability',title="Average Danceability in each label")

    sns.boxplot(data=normalized_data, x='label', y='energy')
    plt.xticks(rotation=30)
    plt.show()

    plot_label_horizontal_bar('energy',title="Average energy in each label")

    sns.boxplot(data=normalized_data, x='label', y='loudness')
    plt.xticks(rotation=30)
    plt.show()

    plot_label_horizontal_bar('loudness',title="Average loudness in each label")

    sns.boxplot(data=normalized_data, x='label', y='speechiness')
    plt.xticks(rotation=30)
    plt.show()

    plot_label_horizontal_bar('speechiness',title="Average speechiness in each label")

    sns.boxplot(data=normalized_data, x='label', y='acousticness')
    plt.xticks(rotation=30)
    plt.show()

    plot_label_horizontal_bar('acousticness',title="Average acousticness in each label")

    sns.boxplot(data=normalized_data, x='label', y='instrumentalness')
    plt.xticks(rotation=30)
    plt.show()

    plot_label_horizontal_bar('instrumentalness',title="Average instrumentalness in each label")

    sns.boxplot(data=normalized_data, x='label', y='liveness')
    plt.xticks(rotation=30)
    plt.show()

    plot_label_horizontal_bar('liveness',title="Average liveness in each label")

    sns.boxplot(data=normalized_data, x='label', y='valence')
    plt.xticks(rotation=30)
    plt.show()

    plot_label_horizontal_bar('valence',title="Average valence in each label")

    sns.boxplot(data=normalized_data, x='label', y='tempo')
    plt.xticks(rotation=30)
    plt.show()

    plot_label_horizontal_bar('tempo',title="Average tempo in each label")

    sns.boxplot(data=normalized_data, x='label', y='duration_ms')
    plt.xticks(rotation=30)
    plt.show()

    plot_label_horizontal_bar('duration_ms',title="Average duration_ms in each label")

    plt.figure (figsize=(20, 12))
    sns.scatterplot(normalized_data['energy'],normalized_data['loudness'],hue=normalized_data['label'], x_bins=30);

    plt.figure (figsize=(20, 12))
    sns.scatterplot(normalized_data['loudness'],normalized_data['acousticness'],hue=normalized_data['label'],x_bins=30);

    plt.figure (figsize=(20, 12))
    sns.scatterplot(normalized_data['valence'],normalized_data['energy'],hue=normalized_data['label'],x_bins=30);

    plt.figure (figsize=(20, 12))
    sns.scatterplot(normalized_data['instrumentalness'],normalized_data['valence'],hue=normalized_data['label'],x_bins=30);

    plt.figure (figsize=(20, 12))
    sns.scatterplot(normalized_data['valence'],normalized_data['liveness'],hue=normalized_data['label'],x_bins=30);

    plt.figure (figsize=(20, 12))
    sns.scatterplot(normalized_data['duration_ms'],normalized_data['instrumentalness'],hue=normalized_data['label'],x_bins=30);

    input_labels.value_counts()

    """# **Recommender System**"""

    def make_recommendation_with_label(label, count):
        data = []
        data.append(df.iloc[normalized_df_genres[normalized_df_genres['label']==label].sample(count).index])
        return data

    result = {'data_1': list(), 'data_2': list(), 'data_3': list(), 'data_4': list(), 'data_5': list()}
    for index, data in enumerate(result.items()):
        tmp = make_recommendation_with_label(input_labels.value_counts().index[index], 5)
        result[data[0]] = tmp

    result['data_1'][0]

    result['data_2'][0]

    result['data_3'][0]

    result['data_4'][0]

    result['data_5'][0]

    result['data_1'][0].to_csv('recommendation_1.csv')
    result['data_2'][0].to_csv('recommendation_2.csv')
    result['data_3'][0].to_csv('recommendation_3.csv')
    result['data_4'][0].to_csv('recommendation_4.csv')
    result['data_5'][0].to_csv('recommendation_5.csv')
    all_clusters = pd.DataFrame()
    for label in set(input_labels):
        tmp = make_recommendation_with_label(label, 1)
        all_clusters = pd.concat([all_clusters,tmp[0]], ignore_index=True)
    
    all_clusters.to_csv('recommendation_with_all_cluster.csv')
    # TODO:
    # 1. Use your train model to make recommendations for the user.
    # 2. Output the recommendations as 5 different playlists with
    #    the top 5 songs in each playlist. (5 playlists x 5 songs)
    # 2.1. Musics in a single playlist should be from the same cluster.
    # 2.2. Save playlists to a csv file.
    # 3. Output another single playlist recommendation with all top songs from all clusters.



if __name__ == "__main__":
    # get arguments from command line
    args = sys.argv
    main(args)