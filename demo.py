import gradio as gr
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from generate_blip2 import generate_blip2
from lexical_constraints import init_batch

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


def apply_model(img, constraints):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto"
    )

    inputs = processor(images=img, return_tensors="pt").to(device, torch.float16)

    # special token id 모음
    period_id = [processor.tokenizer.convert_tokens_to_ids('.')]
    period_id.append(processor.tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [processor.tokenizer.eos_token_id] + period_id

    PAD_ID = processor.tokenizer.convert_tokens_to_ids('<pad>')

    def tokenize_constraints_chk(tokenizer, raw_cts):
        def tokenize2(phrase):
            tokens = tokenizer.tokenize(phrase)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            return token_ids, True
        return [[list(map(tokenize2, clause)) for clause in ct] for ct in raw_cts]
    
    constraints_list = [[[" " + x] for x in constraints]]
    print(constraints_list)
    constraints_list = tokenize_constraints_chk(processor.tokenizer, constraints_list)

    constraints = init_batch(raw_constraints=constraints_list,
                                beam_size=20,
                                eos_id=eos_ids)

    new_generated_ids = generate_blip2(model, **inputs, new_constraints=constraints)
    new_generated_text = processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0].strip()

    return new_generated_text


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
    b3 = gr.Button("generate scriptions")

    new_title = gr.Textbox(label="Here you go!")

    # b1.click(closest_match, inputs=img_input, outputs=text_options)
    updateConst.click(updateCheckbox, inputs=img_input, outputs=[text_options, img_output, img_cut])
    b3.click(apply_model, inputs=[img_input, text_options], outputs=new_title)

demo.launch(debug=True)