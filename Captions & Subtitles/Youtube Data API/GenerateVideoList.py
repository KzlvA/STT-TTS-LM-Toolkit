# YouTube scrape script to pull results based on keywords
# download subtitles using commandline tools & youtube.dl
import pandas as pd
import os
import csv

import google.oauth2.credentials
import google_auth_oauthlib.flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow

# 
CLIENT_SECRETS_FILE = "client_secrets.json"
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

def get_authenticated_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_console()
    return build(API_SERVICE_NAME, API_VERSION, credentials= credentials)

# Remove keyword arguments that are not set
def remove_empty_kwargs(**kwargs):
    good_kwargs = {}
    if kwargs is not None:
        for key, value in kwargs.items():
            if value:
                good_kwargs[key] = value
    return good_kwargs

client = get_authenticated_service()

def youtube_keyword(client, **kwargs):
    kwargs = remove_empty_kwargs(**kwargs)
    response = client.search().list(
        **kwargs
        ).execute()
    return response

def youtube_search(criteria, max_res):
    # create lists and empty dataframe
    titles = []
    videoIds = []
    channelIds = []
    description = []
    resp_df = pd.DataFrame()

    while len(titles) < max_res:
        token = None
        response = youtube_keyword(client,
                                   part='id,snippet',
                                   maxResults=3,
                                   q=criteria,
                                   relevanceLanguage='en',
                                   videoCaption='closedCaption',
                                   type='video',
                                   videoDuration='long',
                                   pageToken=token)

        for item in response['items']:
            titles.append(item['snippet']['title'])
            description.append(item['snippet']['description'])
            channelIds.append(item['snippet']['channelTitle'])
            videoIds.append(item['id']['videoId'])

        token = response["nextPageToken"]
    # search:list returns truncated description. Videos method for full description
    resp_df['title'] = titles
    resp_df['channelId'] = channelIds
    resp_df['videoId'] = videoIds
    resp_df['subject'] = criteria
    resp_df['description'] = description

    return resp_df

# generate pandas dataframe of a given search
# in this iteration (and because of quota limits)
# 4500 videos were called in practice. minimum is 50 regardless of value based on 'max results'
Found_Videos = youtube_search('search query', 4500)
Found_Videos.shape

print(Found_Videos.head())
Found_Videos.to_csv('Found_Videos.csv')






