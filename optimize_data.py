import os
from PIL import Image
from lightning_cloud.utils import add_s3_connection
from lightning_sdk import Machine
from lightning.data import map
import clip

# ----------------------------
# 1. Define a preprocess function

# preprocess the image to save time during training...
# do it on a GPU to make it even faster
# ----------------------------
_, preprocess = clip.load("ViT-B/32")
def preprocess_fn(output_dir, input_filepath):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = preprocess(Image.open(input_filepath)).unsqueeze(0).to(device)

    output_filepath = os.path.join(output_dir, os.path.basename(input_filepath))
    img = Image.fromarray(image.squeeze().cpu().numpy().astype('uint8'))
    img.save(output_filepath)


# ----------------------------
# 2. Add the S3 connection (you can also use the UI via Lightning Drive)

# Attach the bucket and the path to the bucket in the Lightning drive.
# Each file will be passed to the preprocess_fn
# ----------------------------
bucket_name = "coco"
add_s3_connection(bucket_name)
input_dir = f"/teamspace/s3_connections/{bucket_name}"
inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

# ----------------------------
# 2. Define where outputs will be saved

# Attach the bucket and the path to the bucket in the Lightning drive.
# Each file will be passed to the preprocess_fn
# ----------------------------
output_dir = os.getenv("OUTPUT_DIR", f"/teamspace/datasets/{bucket_name}/{%Y-%m-%d-%H-%M-%S}")
outputs = map(
  resize_fn,
  inputs,
  output_dir=output_dir,
  fast_dev_run=int(os.getenv("FAST_DEV_RUN", "0")),
  num_downloaders=6,
  num_nodes=2,
  machine=Machine.A10G
)