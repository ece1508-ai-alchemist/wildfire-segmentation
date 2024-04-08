from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, send_file
from PIL import Image
from torch import optim

from data.load_data import POST_FIRE_DIR
from src.data_loader.dataloader import IMAGE_MASK_TRANSFORM, IMAGE_TRANSFORM
from src.data_loader.dataset import WildfireDataset
from src.model.unet import UNet
from src.train.train import load_checkpoint
from visualize_sample import adjust_image

app = Flask(__name__)

config = dict(n_channels=12, n_classes=1, bilinear=False)
model = UNet(**config)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
scaler = torch.cuda.amp.GradScaler()

CUR_EPOCH = load_checkpoint("model_weights\pre_trained.chkpt", model, scaler, optimizer)

model.eval()

ds = WildfireDataset(POST_FIRE_DIR, None)
ds.test.__set_transforms__(IMAGE_MASK_TRANSFORM, IMAGE_TRANSFORM)

cache = Path(__file__).parent / ".cache"
cache.mkdir(exist_ok=True)


@app.route("/test/<id>/<type>")
def get(id, type):
    id = int(id)

    if (cache / str(id) / f"{type}.png").is_file():
        return send_file(cache / str(id) / f"{type}.png", mimetype="image/png")
    else:
        (cache / str(id)).mkdir(exist_ok=True)

    d = ds.test[id]
    img = d["image"].unsqueeze(0)
    mask = d["mask"]
    out = model(img)

    preds_binary = (torch.sigmoid(out) > 0.5).float()

    flip_img = img[0].permute(1, 2, 0)
    pre_fire = np.array(flip_img)

    flip_pred = torch.squeeze(preds_binary)
    flip_mask = torch.squeeze(mask)

    pre_fire_rgb = adjust_image(pre_fire) * 255

    flip_pred = flip_pred.detach().numpy()

    flip_mask = flip_mask.detach().numpy() * 255

    masked_img = np.expand_dims(flip_pred, axis=-1)

    masked_img = np.where(masked_img == 0, 1, np.where(masked_img == 1, 100, masked_img))
    masked_img = masked_img * pre_fire_rgb

    flip_pred = flip_pred * 255

    if type == "pred":
        image = Image.fromarray(flip_pred.astype(np.uint8))
    elif type == "mask":
        image = Image.fromarray(flip_mask.astype(np.uint8))
    elif type == "img":
        image = Image.fromarray(pre_fire_rgb.astype(np.uint8))
    else:
        image = Image.fromarray(masked_img.astype(np.uint8))

    image.save(cache / str(id) / f"{type}.png", format="PNG")

    return send_file(cache / str(id) / f"{type}.png", mimetype="image/png")


@app.route("/")
def home():
    html = """
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet"></head>
        <title>Flask HTML with Image</title>
    </head>
    <script>
        function updateImage() {
            // Get the input value
            var inputValue = document.getElementById('inputNumber').value;

            // Construct the image URL based on the input value
            var imageUrl = '/test/' + inputValue;

            // Update the src attribute of the image element
            document.getElementById('imageElement').src = imageUrl + '/img';
            document.getElementById('maskElement').src = imageUrl + '/mask';
            document.getElementById('predElement').src = imageUrl + '/pred';
            document.getElementById('maskedElement').src = imageUrl + '/masked';
        }
    </script>
    <body class="bg-gray-100 h-screen flex flex-col items-center justify-center">
    <div class="w-full bg-white p-8 rounded-lg shadow-md">
        <h1 class="text-2xl mb-4">Enter test data index:</h1>
        <div class="flex items-center space-x-2">
            <input type="number" id="inputNumber" min="1" value="1" class="border border-gray-300 rounded-md px-4 py-2 flex-1">
            <button onclick="updateImage()" class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md">Update Images</button>
        </div>
        <br>
        <div class="grid grid-cols-3 gap-4">
        <!-- First image -->
            <div>
                <p>Image</p>
                <img id="imageElement" src="" alt="Image 1" class="mt-4 mx-auto">
            </div>
            <div>
            <p>Mask</p>
                <img id="maskElement" src="" alt="Image 3" class="mt-4 mx-auto">
            </div>
            <div>
            <p>Predicted Mask</p>
                <img id="predElement" src="" alt="Image 2" class="mt-4 mx-auto">
            </div>
            <div>
            <p>Image with Predicted Mask</p>
                <img id="maskedElement" src="" alt="Image 4" class="mt-12 mx-auto">
            </div>
        </div>
    </div>
    </body>
    </html>
        """
    return render_template_string(html)
