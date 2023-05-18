from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

from generate_blip2 import generate_blip2
from lexical_constraints import init_batch

# import logging

    # logger = logging.getLogger()
    # file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # # console_formatter = logging.Formatter('%(message)s')

    # # file log
    # file_handler = logging.FileHandler("./tracking.log", mode='w')
    # file_handler.setFormatter(file_formatter)

    # console log
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(console_formatter)

    # logger.addHandler(file_handler)
    # # logger.addHandler(console_handler)

    # logger.setLevel(logging.INFO)

class Runner:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto"
        )
        self.device = device
        self.processor = processor
        self.model = model
    #model.to(device)

    #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)
    def __call__(self, image_path: str = './image/mir.png'):
        image = Image.open(image_path)

        #prompt = "Question: What is the man looking at? Answer:"
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)

    # logger.info('inputs')
    # logger.info(inputs)

        generated_ids_2 = self.model.generate(**inputs, num_beams=20)
        generated_text_2 = self.processor.batch_decode(generated_ids_2, skip_special_tokens=True)[0].strip()
        print('original generate function without constraint:', generated_text_2)
        res1 = generated_text_2

        const_list = [" remote control"]
        force_words_ids = []
        for word in const_list:
            force_words_ids.append(self.processor.tokenizer.convert_tokens_to_ids(self.processor.tokenizer.tokenize(word))[0])
        #print(force_words_ids)
        inputs['force_words_ids'] = [force_words_ids]
        #print(processor.tokenizer.convert_ids_to_tokens([560, 34046, 17283]))

        generated_ids = self.model.generate(**inputs, num_beams=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print('original generate function with constraint:', generated_text)
        res2 = generated_text

        #########################################################

        # special token id 모음
        period_id = [self.processor.tokenizer.convert_tokens_to_ids('.')]
        period_id.append(self.processor.tokenizer.convert_tokens_to_ids('Ġ.'))
        eos_ids = [self.processor.tokenizer.eos_token_id] + period_id
        #print(eos_ids)
        PAD_ID = self.processor.tokenizer.convert_tokens_to_ids('<pad>')

        def tokenize_constraints_chk(tokenizer, raw_cts):
            def tokenize2(phrase):
                tokens = tokenizer.tokenize(phrase)
                #print(phrase)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                return token_ids, True
            return [[list(map(tokenize2, clause)) for clause in ct] for ct in raw_cts]
        
        constraints_list = [[[" pink"], [" cat"]]] # 앞에 띄어쓰기!
        #constraints_list = [[[" game", " games"], [" league"], [" exciting", " exicted"]]]
        constraints_list = tokenize_constraints_chk(self.processor.tokenizer, constraints_list)
        #print(constraints_list)

        constraints = init_batch(raw_constraints=constraints_list,
                                    beam_size=20,
                                    eos_id=eos_ids)

        new_generated_ids = generate_blip2(self.model, **inputs, new_constraints=constraints)
        new_generated_text = self.processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0].strip()
        print('new generate function with constraint:', new_generated_text)

        res3 = new_generated_text
        return res1, res2, res3

if __name__ == "__main__":
    fx = Runner()
    #import ipdb; ipdb.set_trace()
    res1, res2, res3 = fx(image_path='./image/cat.jpg')