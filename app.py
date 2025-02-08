from flask import Flask, request, render_template, send_file, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import random
import os
import uuid
import zipfile
from io import BytesIO

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure upload folders
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


#Yolov3 Implementation
#--------------------------------------------------------------------
# # Load the YOLO model configuration and weights
# model = cv2.dnn.readNet('models/yolov3.weights', 'models/yolov3.cfg')
# # Get all the layer names from the YOLO model
# layer_names = model.getLayerNames()
# # Identify the output layers (layers with no connections going forward)
# unconnected_layers = model.getUnconnectedOutLayers()
# output_layers = [layer_names[i[0] - 1] if isinstance(i, np.ndarray) else layer_names[i - 1] 
#                  for i in unconnected_layers]

# # Load COCO class names from file
# with open('models/coco.names', 'r') as f:
#     classes = [line.strip() for line in f.readlines()]

# def detect_objects(img):
#     height, width = img.shape[:2]
#     blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
#     model.setInput(blob)
#     outputs = model.forward(output_layers)

#     boxes, confidences, class_ids = [], [], []
#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 box = detection[0:4] * np.array([width, height, width, height])
#                 (center_x, center_y, w, h) = box.astype("int")
#                 x = int(center_x - (w / 2))
#                 y = int(center_y - (h / 2))
#                 boxes.append([x, y, int(w), int(h)])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     return boxes, confidences, class_ids, indexes


#Yolo11 Implementation
#--------------------------------------------------------------------
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('models/yolo11n.pt')  # Ensure you have yolo11n.pt in models/

def detect_objects(img):
    results = model.predict(img)
    return results[0].plot()  # Returns the annotated image directly


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def serve_processed_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process', methods=['POST'])
def process_files():
    if 'files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files')
    if len(files) == 0 or files[0].filename == '':
        flash('No files selected', 'error')
        return redirect(url_for('index'))

    process_id = uuid.uuid4().hex  # Unique ID for this processing session
    processed_files = []

    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Process image
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                
                # ======== REPLACE THIS SECTION ========
                # YOLOv3 Implementation
                #---------------------------------------------------
                # Old YOLOv3 detection and drawing code:
                # boxes, confidences, class_ids, indexes = detect_objects(img)
                # if len(indexes) > 0:
                #     for i in indexes.flatten():
                #         x, y, w, h = boxes[i]
                #         label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                #         color = [random.randint(0, 255) for _ in range(3)]
                #         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                #         cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # New YOLO11 Implementation
                #---------------------------------------------------
                # YOLO11 detection and auto-annotation
                results = model.predict(img)
                annotated_img = detect_objects(img)
            
                # ======== END OF REPLACEMENT ========

                # Save processed image with unique name
                filename = f"{process_id}_{secure_filename(file.filename)}"
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                #cv2.imwrite(save_path, img) #YOLOv3 Implementation
                cv2.imwrite(save_path, annotated_img) #YOLO11 Implementation
                processed_files.append(filename)

            except Exception as e:
                app.logger.error(f"Error processing {file.filename}: {str(e)}")
                flash(f'Error processing {file.filename}', 'error')

    if len(processed_files) == 0:
        flash('No files processed successfully', 'error')
        return redirect(url_for('index'))

    # Handle single file response
    if len(processed_files) == 1:
        return render_template('result.html', 
                             image_url=url_for('serve_processed_image', 
                             filename=processed_files[0]),
                             process_id=process_id)

    # Handle multiple files as zip
    zip_filename = f"processed_images_{process_id}.zip"
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename in processed_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            zip_file.write(file_path, filename)
    
    zip_buffer.seek(0)
    
    response = send_file(zip_buffer,
                        mimetype='application/zip',
                        as_attachment=True,
                        download_name=zip_filename)
    
    # Add custom header to identify zip response
    response.headers['X-Content-Type'] = 'application/zip'
    return response

if __name__ == "__main__":
    app.run(debug=True)