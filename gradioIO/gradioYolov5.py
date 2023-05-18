import gradio as gr
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", device="cpu")  # force_reload=True to update


def yolo(im, size=640):
    g = size / max(im.size)  # gain
    im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize

    results = model(im)  # inference
    detections = results.xyxy[0]
    im = transforms.ToTensor()(im)
    im = im.unsqueeze(0)
    labels = []
    obj_imgs = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf > 0.3:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            obj_img = im[:, :, y1:y2, x1:x2].numpy()
            obj_img = obj_img.transpose(0, 2, 3, 1)
            obj_img = obj_img[0]
            obj_imgs.append(obj_img)
        idx = int(det[-1])
        label = model.names[idx]
        if label not in labels:
            labels.append(label)
    results.show()  # updates results.imgs with boxes and labels
    result_np = np.array(results.render())
    return labels, np.squeeze(result_np), obj_imgs


def apply_model(x):
    return x + ": Highest Cosine Similarity"


def updateCheckbox(img):
    new_options, output, obj_imgs = yolo(img)
    return [
        gr.CheckboxGroup.update(choices=new_options),
        gr.Image().update(value=output),
        gr.Image().update(value=concat_images(obj_imgs)),
    ]


def concat_images(images):
    max_height = max(image.shape[0] for image in images)
    resized_images = []

    for image in images:
        _, width, _ = image.shape
        resized_images.append(cv2.resize(image, (width, max_height)))
    return np.concatenate(resized_images, axis=1)


demo = gr.Blocks()

with demo:
    img_cuts = []
    with gr.Row():
        img_input = gr.inputs.Image(type="pil", label="input image")
        img_output = gr.Image(type="numpy", label="detection image")
    # b1 = gr.Button("Match Closest Title")
    with gr.Row():
        updateConst = gr.Button("update constraints")
        clearData = gr.Button("clear")

    img_cut = gr.Image(type="numpy", label="partial detection image")

    text_options = gr.CheckboxGroup([], label="select options")
    text_input = gr.Textbox(label="Input")
    b3 = gr.Button("generate scriptions")

    new_title = gr.Textbox(label="Here you go!")

    # b1.click(closest_match, inputs=img_input, outputs=text_options)
    updateConst.click(updateCheckbox, inputs=img_input, outputs=[text_options, img_output, img_cut])
    b3.click(apply_model, inputs=text_options, outputs=new_title)

demo.launch(debug=True)
