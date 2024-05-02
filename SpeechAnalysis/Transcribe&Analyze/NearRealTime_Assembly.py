#!pip install sounddevice scipy
import assemblyai as aai
from transformers import pipeline
import threading
import time
import sounddevice as sd 
from scipy.io.wavfile import write
import multiprocessing
import keyboard
import concurrent.futures

recording = False
recorded_data = []
record_count = 0
idx = 0


def record_audio(duration=10, fs=44100, filename='output.wav'):
    """
    Record audio from the microphone.

    Parameters:
    duration (int): Duration of the recording in seconds. Default is 30.
    fs (int): Sampling frequency. Default is 44100.
    filename (str): Name of the output file. Default is 'output.wav'.
    """
    global record_count, recorded_data
    while True:

      print("\n\n***********************")
      print("Recording...")
      recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
      sd.wait()  # Wait until recording is finished
      print("Recording finished. Saving the file...")
      print("***********************")
      filename = f'output_{record_count}.wav'
      write(filename, fs, recording)
      recorded_data.append(recording)
      record_count += 1
      if keyboard.is_pressed('x'):  # if key 'x' is pressed 
        print("\n\n***********************")
        print('You Pressed X Key to exit!')
        print("***********************")
        return filename
      yield filename
    
    #write(filename, fs, recording)  # Save as WAV file
    #print(f"File saved as {filename}")
    #return 'output.wav'

    

#record_audio()

#import threading
#import time
#import gradio as gr
# Global variables for recording


# Function to start recording
def start_recording():
    global recording, recorded_data,record_count
    recording = True
    recorded_data = []
    clip = record_audio()
    recorded_data.append(clip)
    record_count += 1
    


def process_recording():
    global idx,record_count

    while record_count > idx:
      print(record_count, idx)
      inference(recorded_data[idx], 'Sentiment Only')
      idx += 1

# Function to stop recording
def stop_recording():
    global recording
    recording = False

# Function to handle start button
def start_button():
    #threading.Thread(target=start_recording).start()
    #return "Started recording"
    return start_recording()

# Function to handle stop button
def stop_button():
    stop_recording()
    return "Stopped recording and saved data"

#from transformers import pipeline

sentiment_analysis = pipeline(
  "sentiment-analysis",
  #"text-classification"
  framework="pt",
  model="SamLowe/roberta-base-go_emotions"
)


def get_sentiment_emoji(sentiment):
  # Define the mapping of sentiments to emojis
  emoji_mapping = {
    "disappointment": "😞",
    "sadness": "😢",
    "annoyance": "😠",
    "neutral": "😐",
    "disapproval": "👎",
    "realization": "😮",
    "nervousness": "😬",
    "approval": "👍",
    "joy": "😄",
    "anger": "😡",
    "embarrassment": "😳",
    "caring": "🤗",
    "remorse": "😔",
    "disgust": "🤢",
    "grief": "😥",
    "confusion": "😕",
    "relief": "😌",
    "desire": "😍",
    "admiration": "😌",
    "optimism": "😊",
    "fear": "😨",
    "love": "❤️",
    "excitement": "🎉",
    "curiosity": "🤔",
    "amusement": "😄",
    "surprise": "😲",
    "gratitude": "🙏",
    "pride": "🦁"
  }
  return emoji_mapping.get(sentiment, "")


def analyze_sentiment(text):
  results = sentiment_analysis(text)
  sentiment_results = {
    result['label']: result['score'] for result in results
  }
  return sentiment_results


def display_sentiment_results(sentiment_results, option):
  sentiment_text = ""
  for sentiment, score in sentiment_results.items():
    emoji = get_sentiment_emoji(sentiment)
    if option == "Sentiment Only":
      sentiment_text += f"{sentiment} {emoji}\n"
    elif option == "Sentiment + Score":
      sentiment_text += f"{sentiment} {emoji}: {score}\n"
  return sentiment_text



def inference(audio, sentiment_option="Sentiment Only"):
#def inference(audio):

  aai.settings.api_key = "INSERT_API_KEY_HERE"

  #audio_url = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

  config = aai.TranscriptionConfig(
    speaker_labels=True,
    )

  transcript = aai.Transcriber().transcribe(audio, config)
  lang = transcript.json_response['language_code']


  sentiment_results = analyze_sentiment(transcript.text)
  sentiment_output = display_sentiment_results(sentiment_results, 'Sentiment Only')

  print("\n\n***********************")
  print("***********************")
  print(lang.upper(),'\n\n', transcript.text,'\n\n', sentiment_output)
  print("***********************")
  #return lang.upper(), transcript.text, sentiment_output

  #return transcript.text

  

#start_recording()

if __name__ == "__main__":

  with concurrent.futures.ProcessPoolExecutor() as executor:
        # Get the generator from func1
        records = record_audio()

        # Use the executor to run func2 in parallel for each number
        executor.map(inference, records)


  #record_audio()
# Create a pool of processes
  #pool = multiprocessing.Pool()

  # Use the pool to run func2 in parallel
  #pool.map(inference, recorded_data)
   
 # p1 = multiprocessing.Process(target=record_audio, args=(10, ))
 # p2 = multiprocessing.Process(target=process_r\necording)

 # p1.start()
  # starting process 2
 # p2.start()

  # wait until process 1 is finished
 # p1.join()
  # wait until process 2 is finished
 # p2.join()

  # both processes finished
  print("\n\n***********************")
  print("Exiting program! All done")
