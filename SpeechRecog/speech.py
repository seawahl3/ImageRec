import json
import librosa    

from watson_developer_cloud import SpeechToTextV1

stt = SpeechToTextV1(    iam_apikey='t0rpKCce8HfPqMie-2bz3QPi301_0Dj5Nfc-ypuxsb1m', url='https://gateway-wdc.watsonplatform.net/speech-to-text/api')
audio_file = open("IMG_5623.flac", "rb")


with open('transcript_result.json', 'w') as fp:
    result = stt.recognize(audio_file, content_type="audio/x-flac",
                           continuous=True, timestamps=True,
                           max_alternatives=1)
    json.dump(result.get_result(), fp, indent=2)
