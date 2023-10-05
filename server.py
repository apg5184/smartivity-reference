from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from wtforms import FileField
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

class UploadForm(FlaskForm):
    image = FileField('Upload an image')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        filename = secure_filename(form.image.data.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        form.image.data.save(filepath)
        process_image(filepath)
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    return render_template('index.html', form=form)

def process_image(filepath):
    with Image.open(filepath) as img:
        # Simple image processing: converting image to grayscale
        img = img.convert('L')
        img.save(filepath)

if __name__ == '__main__':
    app.run(debug=True)