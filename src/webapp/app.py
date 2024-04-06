from flask import Flask, jsonify,send_file
from src.data_loader.dataset import WildfireDataset
from data.load_data import POST_FIRE_DIR
import torch
from src.train.train import load_checkpoint
from src.model.unet import UNet
from torch import optim
import numpy as np
from io import BytesIO
from src.data_loader.dataloader import get_loader, IMAGE_MASK_TRANSFORM, IMAGE_TRANSFORM
from visualize_sample import adjust_image
from PIL import Image

app = Flask(__name__)


model = UNet(n_channels=12, n_classes=1, bilinear=False)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
scaler = torch.cuda.amp.GradScaler()

CUR_EPOCH = load_checkpoint("/Users/claude/Dev/wildfire-segmentation/model_weights/pre_trained.chkpt", model, scaler, optimizer)

model.eval()


@app.route('/')
def hello():
    ds = WildfireDataset(POST_FIRE_DIR, None)
    ds.test.__set_transforms__(IMAGE_MASK_TRANSFORM, IMAGE_TRANSFORM)
    d = ds.test[0]
    img = d['image'].unsqueeze(0)
    mask = d['mask']
    out = model(img)

    preds_binary = (torch.sigmoid(out) > 0.5).float()

    flip_img = img[0].permute(1, 2, 0)
    pre_fire = np.array(flip_img)
    print(preds_binary.size())
    flip_pred = torch.squeeze(preds_binary)
    print(flip_pred.size())

    pre_fire_rgb = adjust_image(pre_fire) * 255

    flip_pred = flip_pred.detach().numpy() * 255

    image = Image.fromarray(flip_pred.astype(np.uint8))
    image_buffer = BytesIO()
    image.save(image_buffer, format='PNG')
    image_buffer.seek(0)

    return send_file(image_buffer, mimetype='image/png')
