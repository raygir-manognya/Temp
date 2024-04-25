from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
from traceback import print_exc
import json
from torchvision.transforms.functional import to_pil_image

UPLOAD_FOLDER = 'Uploaded_Files'

app = Flask("_main_", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model architecture
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


# For prediction of output
def predict(model, img, path='./'):
    sm = nn.Softmax(dim=1)
    fmap, logits = model(img.to())
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('confidence of prediction: ', logits[:, int(prediction.item())].item() * 100)
    return [int(prediction.item()), confidence]


# Validation dataset
class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        vidObj = cv2.VideoCapture(video_path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = [(0, image.shape[1], image.shape[0], 0)]  # Default face location if not found
                try:
                    faces = face_recognition.face_locations(image)
                except Exception as e:
                    print(f"Error: {e}")
                top, right, bottom, left = faces[0]
                frame = image[top:bottom, left:right, :]
                if self.transform:
                    frame = self.transform(to_pil_image(frame))
                frames.append(frame)
                if len(frames) == self.count:
                    break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)


def detect_fake_video(video_path):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    path_to_videos = [video_path]

    video_dataset = ValidationDataset(path_to_videos, sequence_length=20, transform=train_transforms)
    model = Model(2)
    path_to_model = r'C:\Users\Adi Narayana Thota\PycharmProjects\deepfake\DeepFake_Detection\df_model.pt'

    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    predictions = []
    for i in range(len(path_to_videos)):
        prediction = predict(model, video_dataset[i])
        predictions.append(prediction)
        if prediction[0] == 1:
            print("REAL")
        else:
            print("FAKE")
    return predictions


@app.route('/', methods=['POST', 'GET'])
def homepage():
    if request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/Detect', methods=['POST', 'GET'])
def detect_page():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        video = request.files['video']
        print("Original Filename:", video.filename)
        video_filename = secure_filename(video.filename)

        # Ensure 'Uploaded_Files' directory exists
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        video_path = os.path.join(upload_folder, video_filename)
        video.save(video_path)

        try:
            predictions = detect_fake_video(video_path)
            print("Predictions:", predictions)

            output = "REAL" if predictions[0][0] == 1 else "FAKE"
            confidence = predictions[0][1]
            data = {'output': output, 'confidence': confidence}
            data = json.dumps(data)

            # Remove the uploaded video after processing
            os.remove(video_path)

            return render_template('index.html', data=data)

        except Exception as e:
            print("An error occurred:", str(e))
            print_exc()  # Print the traceback
            return f"An error occurred while processing the video: {str(e)}", 500  # Return 500 Internal Server Error status


app.run(port=5000, debug=True)
