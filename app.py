import os
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import numpy as np
from datetime import datetime
import cv2
# Set matplotlib backend to Agg before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend, no Tkinter
import matplotlib.pyplot as plt
from flask import send_file, Flask, render_template, request
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import urllib.parse

gc.collect()

import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MODEL_PATH = 'static/model/EfficientNet_unfreeze150.keras'
MODEL_PATH = 'static/model/MobileNet.keras'
model = load_model(MODEL_PATH)

print("TensorFlow version:", tf.__version__)
model.summary()

print("Model type:", type(model))
if isinstance(model, tf.keras.Sequential):
    print("Model is Sequential. Building with input shape.")
    model.build((None, 224, 224, 3))
else:
    print("Model is not Sequential. Using dummy input.")
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy_input)
print("Model initialized.")

CLASS_LABELS = {
    0: "COPD Signs",
    1: "COVID-19",
    2: "Lung Metastasis",
    3: "Normal",
    4: "Pneumonia",
    5: "Tuberculosis"
}


def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img) / 255.0
    array = np.expand_dims(array, axis=0)
    print("Inside get_img_array - shape:", array.shape)
    print("Inside get_img_array - min/max:", array.min(), array.max())
    return array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    print("Starting Grad-CAM computation...")
    try:
        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(img_array)
            tape.watch(inputs)
            outputs = model(inputs)
            if pred_index is None:
                pred_index = tf.argmax(outputs[0])
                print(f"Predicted class index: {pred_index}")
            loss = outputs[:, pred_index]
        
        grads = tape.gradient(loss, inputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        pooled_grads = tf.reshape(pooled_grads, [1, 1, 1, 3])
        heatmap = tf.reduce_mean(inputs * pooled_grads, axis=-1)[0]
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10
        print("Heatmap generated.")
        return heatmap
    except Exception as e:
        print(f"Grad-CAM computation failed: {str(e)}")
        raise

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:            
            file = request.files['file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img = get_img_array(filepath, size=(224, 224))
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            diagnosis = CLASS_LABELS[predicted_class]
            print(f"Prediction: {diagnosis}, Confidence: {confidence:.2f}%")
            last_conv_layer_name = "top_conv"
            heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{file.filename}")
            plt.imshow(heatmap, cmap='jet')
            plt.colorbar()
            plt.savefig(heatmap_path)
            plt.close()
            print(f"Heatmap saved to: {heatmap_path}")

            encoded_image_url = urllib.parse.quote(file.filename)
            gradcam_url = f"gradcam_{file.filename}"
            print(f"Passing to result.html: image_url={file.filename}, encoded_image_url={encoded_image_url}, diagnosis={diagnosis}, confidence={confidence:.2f}%, gradcam_url={gradcam_url}")

            return render_template(
                'result.html',
                diagnosis=diagnosis,
                confidence=f"{confidence:.2f}%",
                image_url=file.filename,
                gradcam_url=gradcam_url,
                encoded_image_url=encoded_image_url
            )
        except Exception as e:
            print(f"Error in home route: {str(e)}")
            raise
    return render_template('home.html')



@app.route('/download_report', methods=['GET'])
def download_report():
    try:
        image_url = request.args.get('image_url')
        diagnosis = request.args.get('diagnosis')
        confidence = request.args.get('confidence')
        gradcam_url = request.args.get('gradcam_url')

        print(f"Download report args: image_url={image_url}, diagnosis={diagnosis}, confidence={confidence}, gradcam_url={gradcam_url}")

        if not all([image_url, diagnosis, confidence, gradcam_url]):
            missing = [k for k, v in {'image_url': image_url, 'diagnosis': diagnosis, 'confidence': confidence, 'gradcam_url': gradcam_url}.items() if v is None]
            raise ValueError(f"Missing parameters: {missing}. Ensure all are passed from result.html.")

        original_img_path = os.path.join(UPLOAD_FOLDER, image_url)
        gradcam_img_path = os.path.join(UPLOAD_FOLDER, gradcam_url)

        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        
        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(150, 770, "XpertScan Diagnostic Report")

        # Image name
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, f"Image File: {image_url}")

        # Original image
        original_img = ImageReader(original_img_path)
        c.drawImage(original_img, 100, 580, width=200, height=150)

        # Grad-CAM image
        gradcam_img = ImageReader(gradcam_img_path)
        c.drawImage(gradcam_img, 320, 580, width=200, height=150)

        # Diagnosis and confidence
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, 540, f"Diagnosis: {diagnosis}")
        c.drawString(100, 520, f"Confidence: {confidence}")

        # Footer
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(100, 100, "Generated by XpertScan")
        c.drawString(100, 85, f"Date: {datetime.now().strftime('%B %d, %Y')}")

        c.save()
        pdf_buffer.seek(0)

        return send_file(pdf_buffer, as_attachment=True, download_name="report.pdf", mimetype="application/pdf")
    except Exception as e:
        print(f"Error in download_report: {str(e)}")
        raise


@app.route('/model_summary')
def model_summary_route():
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return f"<pre>{summary_string}</pre>"

if __name__ == '__main__':
    app.run(debug=True)