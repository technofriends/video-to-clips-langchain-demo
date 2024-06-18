import os
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

import subprocess
import json

from typing import List


# load the environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

youtube_url = "https://www.youtube.com/watch?v=4WO5kJChg3w"

os.makedirs("downloaded_videos", exist_ok=True)

# pytube downlaod video part

yt = YouTube(youtube_url)
video = yt.streams.filter(file_extension='mp4').first()
safe_title = yt.title.replace(' ', '_')
filename = f"downloaded_videos/{safe_title}.mp4"

video.download(filename=filename)

#get the transcript
video_id = yt.video_id
transcript = YouTubeTranscriptApi.get_transcript(video_id)
print(transcript)
# define the llm

llm = ChatOpenAI(model='gpt-4o',
                 temperature=0.7, 
                 max_tokens=None,
                 timeout=None,
                 max_retries=2
                 )

# build prompt for LLM
prompt = f"""Provided to you is a transcript of a video. 
Please identify all segments that can be extracted as 
subtopics from the video based on the transcript.
Make sure each segment is between 30-500 seconds in duration.
Make sure you provide extremely accruate timestamps
and respond only in the format provided. 
\n Here is the transcription : \n {transcript}"""

messages = [
    {"role": "system", "content": "You are a viral content producer. You are master at reading youtube transcripts and identifying the most intriguing content. You have extraordinary skills to extract subtopic from content. Your subtopics can be repurposed as a separate video."},
    {"role": "user", "content": prompt}
]

class Segment(BaseModel):
    """ Represents a segment of a video"""
    start_time: float = Field(..., description="The start time of the segment in seconds")
    end_time: float = Field(..., description="The end time of the segment in seconds")
    yt_title: str = Field(..., description="The youtube title to make this segment as a viral sub-topic")
    description: str = Field(..., description="The detailed youtube description to make this segment viral ")
    duration : int = Field(..., description="The duration of the segment in seconds")

class VideoTranscript(BaseModel):
    """ Represents the transcript of a video with identified viral segments"""
    segments: List[Segment] = Field(..., description="List of viral segments in the video")

structured_llm = llm.with_structured_output(VideoTranscript)
ai_msg = structured_llm.invoke(messages)
print(ai_msg)
parsed_content = ai_msg.dict()['segments']

# create a folder to store clips
os.makedirs("generated_clips", exist_ok=True)
segment_labels = []
video_title = safe_title

for i, segment in enumerate(parsed_content):
    start_time = segment['start_time']
    end_time = segment['end_time']
    yt_title = segment['yt_title']
    description = segment['description']
    duration = segment['duration']

    output_file = f"generated_clips/{video_title}_{str(i+1)}.mp4"
    command = f"ffmpeg -i {filename} -ss {start_time} -to {end_time} -c:v libx264 -c:a aac -strict experimental -b:a 192k {output_file}"
    subprocess.call(command, shell=True)
    segment_labels.append(f"Sub-Topic {i+1}: {yt_title}, Duration: {duration}s\nDescription: {description}\n")

with open('generated_clips/segment_labels.txt', 'w') as f:
    for label in segment_labels:
        f.write(label +"\n")

# save the segments to a json file
with open('generated_clips/segments.json', 'w') as f:
    json.dump(parsed_content, f, indent=4)
    