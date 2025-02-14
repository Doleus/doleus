import os
import zipfile
import urllib.request
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from moonwatcher.utils.data import Task
from moonwatcher.check import Check, CheckSuite
from moonwatcher.dataset.dataset import MoonwatcherClassification

# Step 1) Download the Dataset
data_dir = "./tiny-imagenet-200"
zip_file = "tiny-imagenet-200.zip"
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
model_checkpoint_url = "https://huggingface.co/zeyuanyin/tiny-imagenet/resolve/main/rn18_50ep/checkpoint.pth"
checkpoint_path = "./rn18_50ep.pth"


def download_tiny_imagenet(data_dir, zip_file, url):
    if not os.path.exists(data_dir):
        print("Tiny ImageNet not found. Downloading...")
        urllib.request.urlretrieve(url, zip_file)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete.")
        os.remove(zip_file)
    else:
        print("Tiny ImageNet is already set up.")


download_tiny_imagenet(data_dir, zip_file, url)

# Step 2) Download the Model Checkpoint
if not os.path.exists(checkpoint_path):
    print("Downloading pretrained model checkpoint...")
    urllib.request.urlretrieve(model_checkpoint_url, checkpoint_path)
    print("Checkpoint downloaded.")

# Step 3) Define the Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 4) Load the Dataset
train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "train"), transform=transform)
subset_size = 100
indices = torch.randperm(len(train_dataset))[:subset_size]
subset = Subset(train_dataset, indices)
train_loader = DataLoader(subset, batch_size=8, shuffle=False)

# Step 5) Load the Model
model = torchvision.models.get_model('resnet18', num_classes=200)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(
    3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = torch.nn.Identity()
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model_weights = checkpoint['model']
model.load_state_dict(model_weights)
model.eval()

# Step 6) Generate Predictions
predictions = []
with torch.no_grad():
    for images, lbls in train_loader:
        outputs = model(images)
        predictions.append(outputs)
predictions = torch.cat(predictions, dim=0)

# Step 7) Create Moonwatcher Dataset
moonwatcher_dataset = MoonwatcherClassification(
    name="tiny_imagenet_subset",
    dataset=subset,
    task=Task.MULTICLASS.value,
    num_classes=200
)

# Step 8) Add Metadata
moonwatcher_dataset.add_predefined_metadata("brightness")

# Step 9) Create Slices
slice_bright = moonwatcher_dataset.slice_by_percentile("brightness", ">=", 50)
slice_dim = moonwatcher_dataset.slice_by_percentile("brightness", "<", 50)

# Step 10) Create Checks
check_bright = Check(
    name="accuracy_bright",
    dataset=slice_bright,
    predictions=predictions,
    metric="Accuracy",
    operator=">",
    value=0.5,
)
check_dim = Check(
    name="accuracy_dim",
    dataset=slice_dim,
    predictions=predictions,
    metric="Accuracy",
    operator=">",
    value=0.5,
)

# Step 11) Create Check Suite
check_suite = CheckSuite(
    name="test_brightness",
    checks=[check_bright, check_dim]
)

# Step 12) Run Checks
test_results = check_suite.run_all(show=True)
