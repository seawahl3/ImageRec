from flask import render_template, Flask, request
from werkzeug import secure_filename
from flask_cors import CORS
from crossdomain import crossdomain
from flask import jsonify
import time
import sys
from speech import SpeechRecog
from VideoHandling import HandleVideo


# Create the application instance
app = Flask(__name__)
CORS(app)

# Read the swagger.yml file to configure the endpoints
#app.add_api('swagger.yml')

@app.route('/uploader', methods = ['POST'])
@crossdomain(origin='*')
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return jsonify(
                boardTranscription=HandleVideo(f.filename),
                audioTranscription= SpeechRecog().get_audio_transcription(f.filename)
                )
		
		
# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
	
	