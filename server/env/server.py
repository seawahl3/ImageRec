from flask import render_template, Flask, request
from werkzeug import secure_filename
from flask_cors import CORS
from crossdomain import crossdomain
from flask import jsonify
import time
import sys
from speech import SpeechRecog
from VideoHandling import HandleVideo
import os 
import shutil


# Create the application instance
app = Flask(__name__)
CORS(app)

# Read the swagger.yml file to configure the endpoints
#app.add_api('swagger.yml')

@app.route('/uploader', methods = ['POST'])
@crossdomain(origin='*')
def upload_file():
    if request.method == 'POST':
        try:
                f = request.files['file']
        except:
                if os.path.exists('/VideoFrameData'):
                        shutil.rmtree('/VideoFrameData')
                if os.path.exists('/out'):
                        shutil.rmtree('/out')
                if os.path.exists('/letters'):
                        shutil.rmtree('/letters')
        f.save(secure_filename(f.filename))
        return jsonify(
                boardTranscription=HandleVideo(f.filename),
                audioTranscription= SpeechRecog().get_audio_transcription(f.filename)
                # boardTranscription=[],
                # audioTranscription= []
                )
		
		
# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
	
	