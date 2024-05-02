#!pip install sounddevice scipy
#import assemblyai as aai
from transformers import pipeline
import threading
import time
import sounddevice as sd 
from scipy.io.wavfile import write
#from multiprocessing import Process, Queue,Manager
import keyboard
#import concurrent.futures
#import queue
from queue import Queue
#import whisper
from datetime import datetime
from faster_whisper import WhisperModel


print("\n\nStarted Program: ", datetime.now().strftime("%H:%M:%S"))
#recording = False
#recorded_data = []
#record_count = 0
#idx = 0

full_text = ''

#q = queue.Queue()


def record_audio(q, duration=10, fs=44100, filename='output.wav'):
    record_count = 0
    """
    Record audio from the microphone.

    Parameters:
    duration (int): Duration of the recording in seconds. Default is 30.
    fs (int): Sampling frequency. Default is 44100.
    filename (str): Name of the output file. Default is 'output.wav'.
    """
    #global record_count, recorded_data
    while True:

      print("\n\n***********************")
      print("Recording...:", datetime.now().strftime("%H:%M:%S"))
      recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
      sd.wait()  # Wait until recording is finished
      print("Recording finished. Saving the file...:", datetime.now().strftime("%H:%M:%S"))
      print("***********************")
      filename = f'output_{record_count}.wav'
      write(filename, fs, recording)
      #recorded_data.append(recording)
      #record_count += 1
      q.put(filename)
      if keyboard.is_pressed('x'):  # if key 'x' is pressed 
        print("\n\n***********************")
        print('You Pressed X Key to exit!')
        print("***********************")
        break
      #yield filename
    
    #write(filename, fs, recording)  # Save as WAV file
    #print(f"File saved as {filename}")
    #return 'output.wav'

    

#record_audio()

#import threading
#import time
#import gradio as gr
# Global variables for recording


# Function to start recording
# def start_recording():
#     global recording, recorded_data,record_count
#     recording = True
#     recorded_data = []
#     clip = record_audio()
#     recorded_data.append(clip)
#     record_count += 1
    


def process_recording(q):
    #global idx,record_count

    print("Inside process_recording :", datetime.now().strftime("%H:%M:%S"))

    #while True:
    while not q.empty():
        item = q.get()
        print(item)
        inference(item,"Sentiment Only")
        print("Processing item from queue: ", item, datetime.now().strftime("%H:%M:%S"))
        q.task_done()
        time.sleep(1)
    #while record_count > idx:
    #  print(record_count, idx)
    #  inference(recorded_data[idx], 'Sentiment Only')
    #  idx += 1

# Function to stop recording
# def stop_recording():
#     global recording
#     recording = False

# Function to handle start button
# def start_button():
#     #threading.Thread(target=start_recording).start()
#     #return "Started recording"
#     return start_recording()

# Function to handle stop button
# def stop_button():
#     stop_recording()
#     return "Stopped recording and saved data"

#from transformers import pipeline
#model = whisper.load_model("base")
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



# def inference(audio, sentiment_option):
# #def inference(audio):
#   print("Started inference: ", datetime.now().strftime("%H:%M:%S"))
#   audio = whisper.load_audio(audio)
#   audio = whisper.pad_or_trim(audio)

#   mel = whisper.log_mel_spectrogram(audio).to(model.device)

#   _, probs = model.detect_language(mel)
#   lang = max(probs, key=probs.get)

#   options = whisper.DecodingOptions(fp16=False)
#   transcript = whisper.decode(model, mel, options)

#   sentiment_results = analyze_sentiment(transcript.text)
#   sentiment_output = display_sentiment_results(sentiment_results, 'Sentiment Only')

#   print("Stopped inference: ", datetime.now().strftime("%H:%M:%S"))

#   print("\n\n***********************")
#   print("***********************")
#   print(lang.upper(),'\n\n', transcript.text,'\n\n', sentiment_output)
#   print("***********************")
  #return lang.upper(), transcript.text, sentiment_output

  #return transcript.text




model_size = "small"

# Run on GPU with FP16
#model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# segments, info = model.transcribe("audio.mp3", beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))



def inference(audio, sentiment_option):
#def inference(audio):
    
    global full_text
    print("Started inference: ", datetime.now().strftime("%H:%M:%S"))
  
    

    transcript, info = model.transcribe(audio, beam_size=5)

    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # print(list(segments))
    # print(type(segments))

    # for segment in segments:
    #    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    # transcript = dict(transcript)
    
    for segment in transcript:
        seg_text = segment.text
        full_text += seg_text + ' '
        sentiment_results = analyze_sentiment(seg_text)
        sentiment_output = display_sentiment_results(sentiment_results, 'Sentiment Only')
        print("\n\n***********************")
        print("***********************")
        print('\n\n',seg_text,'\n\n', sentiment_output)
        print("***********************")


    print("Stopped inference: ", datetime.now().strftime("%H:%M:%S"))

    print("\n\n***********************")
    print("***********************")
    #print('\n\n', transcript.text,'\n\n', sentiment_output)
    print("***********************")




#start_recording()


if __name__ == "__main__":
   
   #inference('output_1.wav','Sentiment Only')
  
    # with Manager() as manager:
    #     q = manager.Queue()
    #     print("\n\nStarting p1 :", datetime.now().strftime("%H:%M:%S"))
    #     p1 = Process(target=record_audio, args=(q,))
    #     p1.start()
    #     print("\n\nStarted p1 :", datetime.now().strftime("%H:%M:%S"))
    #     time.sleep(13)
    #     print("\n\nStarting p2 :", datetime.now().strftime("%H:%M:%S"))
    #     p2 = Process(target=process_recording, args=(q,))
        
        q = Queue()
        print("\n\nStarting p1 :", datetime.now().strftime("%H:%M:%S"))
        p1 = threading.Thread(target=record_audio, args=(q,))
        p1.start()
        
        print("\n\nStarted p1 :", datetime.now().strftime("%H:%M:%S"))
        time.sleep(11)
        print("\n\nStarting p2 :", datetime.now().strftime("%H:%M:%S"))
        p2 = threading.Thread(target=process_recording, args=(q,))

        # Start the processes
        
        p2.start()
        print("\n\nStarted p2 :", datetime.now().strftime("%H:%M:%S"))
        # Wait for both processes to finish
        p1.join()
        p2.join()
        print("\n\n***********************")
        print("Exiting program! All done :", datetime.now().strftime("%H:%M:%S")) 
        

#inference('output_0.wav', 'Sentiment Only')
#  with concurrent.futures.ProcessPoolExecutor() as executor:
#        # Get the generator from func1
#        records = record_audio()

        # Use the executor to run func2 in parallel for each number
#        executor.map(inference, records)


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
  