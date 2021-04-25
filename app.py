 
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch

w = [128, 128, 256, 256]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, w[0], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(w[0])
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(w[0], w[1], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(w[1])
        self.conv3 = nn.Conv2d(w[1], w[2], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(w[2])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(14 * 14 * w[2], w[3])
        self.linear2 = nn.Linear(w[3], 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool2(self.relu(self.bn3(self.conv3(x))))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                               ])

our_model = Net() 
our_model.load_state_dict(torch.load('best5.pth'))  


classes = ["COVID19", "NORMAL", "PNEUMONIA"]

def predict(image_path):  
    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)
    our_model.eval()
    out = our_model(batch_t)
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:3]]

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Predict Covid-19, Normal or Pneumonia")
st.write("Example Using a 3 layer Convolutional Neural Network for DLH Project")
st.write("Team: Gunther Bacellar, Soumava Dey, Rajlakshman Kulkarni and Mallikarjuna Chandrappa")
file_up = st.file_uploader("Upload a JPG x-ray image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
  
    for i, label in enumerate(labels):        
        if i == 0:
            st.write(f"Prediction #1 as {label[0]} (score: {label[1]:.4f})")
            st.write("Other smaller possibilities:")
        else:
            st.write(f" {i+1}. {label[0]} (score: {label[1]:.4f})")