# CSV conversion file
# Takes vtt files from a directory and outputs a clean CSV of a pandas dataframe
# Removes triplicates, by default vtt files contain 3 iterations
import os
import pandas as pd
import webvtt

# filter filenames by .vtt
filenames_vtt = [os.fsdecode(file) for file in os.listdir(os.getcwd()) if os.fsdecode(file).endswith(".vtt")]

#Check file names
filenames_vtt[:2]

def convert_vtt(filenames):
    #create an assets folder if one does not yet exist
    if os.path.isdir('{}/assets'.format(os.getcwd())) == False:
        os.makedirs('/Users/.../pythonProject/assets')
    #extract the text and times from the vtt file
    for file in filenames:
        captions = webvtt.read(file)
        transcript = ""
        lines = []
        for line in captions:
            # Strip the newlines from the end of the text.
            # Split the string if it has a newline in the middle
            # Add the lines to an array
            lines.extend(line.text.strip().splitlines())
        # Remove repeated lines
        previous = None
        for line in lines:
            if line == previous:
                continue
            transcript += " " + line
            previous = line
        text_time = pd.DataFrame()
        text_time['text'] = [transcript]
        #
        # print(transcript)
        # text_time['start'] = [caption.start for caption in captions]
        # text_time['stop'] = [caption.end for caption in captions]
        text_time.to_csv('assets/{}.csv'.format(file[:-4]),index=False)
        #-4 to remove '.vtt'
        #remove files from local drive
        # os.remove(file)

#call the function
convert_vtt(filenames_vtt)

csv_files = [os.fsdecode(file) for file in os.listdir(os.getcwd()+'/assets/') if os.fsdecode(file).endswith('.csv')]
#take a look at a file name
csv_files[0]

path = '/Users/.../pythonProject/assets/'
for filename in csv_files:
    os.rename(os.path.join(path, filename), os.path.join(path, filename.replace(' ', '')))

#to verify the results
clean_csv = [os.fsdecode(file) for file in os.listdir(os.getcwd()+'/assets')]

clean_csv[0]

#extract the text and videoid
vidText = []
csv_vidid = []

for file in clean_csv:
   df = pd.read_csv(path+file)
   text = " ".join(df.text)
   vidText.append(text)
   csv_vidid.append(file[-18:-7])

# create pandas df and write to CSV all other converted vtt files from /assets
converted_df = pd.DataFrame()

# 'subt' indicates subtitle transcript data column
converted_df['video_title'] = clean_csv
converted_df['subt'] = vidText
converted_df['vid_id'] = csv_vidid
converted_df.head()
converted_df.to_csv('result_pd.csv')
print(converted_df)
subtitle = converted_df['subt']
print(subtitle)
