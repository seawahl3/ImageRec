import json
import librosa
import moviepy.editor as mp 

from watson_developer_cloud import SpeechToTextV1

class SpeechRecog ():
    def __init__(self):
        pass

    def get_audio_transcription(self, video_file): 
        stt = SpeechToTextV1(    iam_apikey='t0rpKCce8HfPqMie-2bz3QPi301_0Dj5Nfc-ypuxsb1m', url='https://gateway-wdc.watsonplatform.net/speech-to-text/api')
        clip = mp.VideoFileClip(video_file)
        clip.audio.write_audiofile("theaudio.mp3")
        audio_file = open("theaudio.mp3", "rb")


        with open('transcript_result.json', 'w') as fp:
            result = stt.recognize(audio_file, content_type="audio/mp3",
                                continuous=True, timestamps=True,
                                max_alternatives=1)
            json.dump(result.get_result(), fp, indent=2)
            audioTranscription = []
            for transcription in result.get_result()["results"][0]["alternatives"][0]["timestamps"]:
                audioTranscription.append((transcription[0],transcription[1]))
            return audioTranscription