import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import json
import random
from torch import nn
from torchvision import transforms


class_names = ['Empty', 'Occupied']


def display_capture_frame(model:torch.nn.Module, frame, data):
    """
    Display the captured frame with bounding boxes and slot information.

    Parameters:
    - model (torch.nn.Module): The trained model for occupancy prediction.
    - frame: The captured frame from the video.
    - data: JSON data containing slot information.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([ transforms.Resize(size=(32,32), antialias=True) ])

    plt.imshow(frame_rgb)

    for slot in data:
        slot_id = slot['Slot Id']
        vertices_old = slot['Vertices']

        vertices = np.array(vertices_old)
        # to create a bounding box with rectangle
        x_min = np.min(vertices[:, 0])
        x_max = np.max(vertices[:, 0])
        y_min = np.min(vertices[:, 1])
        y_max = np.max(vertices[:, 1])
        extracted_image = frame[y_min:y_max, x_min:x_max]

        # print( extracted_image.shape, extracted_image.dtype)

        extracted_image = extracted_image.transpose(2, 0, 1)
        extracted_image = torch.tensor(extracted_image, dtype=torch.float32)
        occupied = pred_image(model, extracted_image, class_names=class_names, transform=transform)
        color = 'red' if occupied == 1 else 'green'
        bbox = Polygon(vertices_old, closed=True, edgecolor='b', facecolor=color, alpha=0.4)
        if len(data) < 15:
            ax.text(x_min, y_max, f'Slot {slot_id}', color='white', fontsize=8, weight='bold')
        
        ax.add_patch(bbox)
    plt.tight_layout()
    plt.title('Captured Frame')
    plt.axis('off')
    plt.show()


def load_model(model_path):
    """
    Load the trained model.

    Parameters:
    - model_path (str): Path to the model file.

    Returns:
    - model_state_dict: The state dictionary of the loaded model.
    """    
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu')) 
    return(model_state_dict)


def pred_image(model:torch.nn.Module,
               image: torch.Tensor,
               class_names = class_names,
               transform=None,
               device=None):
  """
    Make a prediction on a target image.

    Parameters:
    - model (torch.nn.Module): The trained model for occupancy prediction.
    - image (torch.Tensor): The input image tensor.
    - class_names (list): List of class names.
    - transform: Image transformation.
    - device: Target device for the model.

    Returns:
    - target_image_pred_label: Predicted label for the input image.
   """
  # 1. Load in image, convert the tensor values to float32 Divide the image pixel values by 255 to get them between [0, 1]
  target_image = image / 255.

  # 2. Transform if necessary
  if transform:
      target_image = transform(target_image)

  # 3. Make sure the model is on the target device
  model

  # 4. Turn on model evaluation mode and inference mode
  model.eval()

  with torch.inference_mode():
      # Add an extra dimension to the image
      target_image = target_image.unsqueeze(dim=0)

      # Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(target_image)

  # 5. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
  target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

  # 6. Convert prediction probabilities -> prediction labels
  target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

  # 7 . return the target_image_pred_label
  return target_image_pred_label

def load_data(file_path):
    """
    Load JSON data from a file.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - data: Loaded JSON data.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main():

    # Load video and it's json file
    video_path = './data/parking_crop_loop.mp4'
    file_path = './data/mask_crop.json'

    # Load model path
    car_detector_path = './models/CarDetectorModel_0_state.pth'

    cap = cv2.VideoCapture(video_path)

    # Read the JSON file
    data = load_data(file_path)
    
    car_detector_state = load_model(car_detector_path)

    class CarDetectorModel0(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
                super(CarDetectorModel0, self).__init__()
                self.conv_block_1 = nn.Sequential(
                    nn.Conv2d(in_channels=input_shape,
                            out_channels=hidden_units,
                            kernel_size=3, # how big is the square that's going over the image?
                            stride=1, # default
                            padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2,
                                stride=2) # default stride value is same as kernel_size
                )
                self.conv_block_2 = nn.Sequential(
                    nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    # Where did this in_features shape come from?
                    # It's because each layer of our network compresses and changes the shape of our inputs data.
                    nn.Linear(in_features=hidden_units*8*8,
                            out_features=output_shape)
                )

        def forward(self, x: torch.Tensor):
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = self.classifier(x)
            return x

    
    car_detector = CarDetectorModel0(input_shape=3, # number of color channels (3 for RGB)
                    hidden_units=10,
                    output_shape=2)

    car_detector.load_state_dict(car_detector_state)

    # print(car_detector.state_dict())

    while True:
        ret, frame = cap.read()

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)

        if key == ord('k'):
            capture = frame.copy()
            display_capture_frame(car_detector, capture, data)

        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()

if __name__ == "__main__":
    main()





