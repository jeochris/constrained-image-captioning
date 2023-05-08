from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

from generate_blip2 import generate_blip2
from lexical_constraints import init_batch

# import logging

def main():

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto"
    )
    #model.to(device)

    #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open('./image/dog.jpg')

    #prompt = "Which dog is winning the fight?"
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    # logger.info('inputs')
    # logger.info(inputs)

    generated_ids_2 = model.generate(**inputs)
    generated_text_2 = processor.batch_decode(generated_ids_2, skip_special_tokens=True)[0].strip()
    print('original generate function without constraint:', generated_text_2)




    force_words_ids = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize(" bark barking"))
    #print(force_words_ids)
    inputs['force_words_ids'] = [force_words_ids]
    #print(processor.tokenizer.convert_ids_to_tokens([560, 34046, 17283]))

    generated_ids = model.generate(**inputs, num_beams=5)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print('original generate function with constraint:', generated_text)




    # special token id 모음
    period_id = [processor.tokenizer.convert_tokens_to_ids('.')]
    period_id.append(processor.tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [processor.tokenizer.eos_token_id] + period_id
    #print(eos_ids)
    PAD_ID = processor.tokenizer.convert_tokens_to_ids('<pad>')

    def tokenize_constraints_chk(tokenizer, raw_cts):
        def tokenize2(phrase):
            tokens = tokenizer.tokenize(phrase)
            #print(phrase)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            return token_ids, True
        return [[list(map(tokenize2, clause)) for clause in ct] for ct in raw_cts]
    
    constraints_list = [[[" bark", " barking", " barks"]]] # 앞에 띄어쓰기!
    constraints_list = tokenize_constraints_chk(processor.tokenizer, constraints_list)
    #print(constraints_list)

    constraints = init_batch(raw_constraints=constraints_list,
                                beam_size=5,
                                eos_id=eos_ids)

    new_generated_ids = generate_blip2(model, **inputs, new_constraints=constraints)
    new_generated_text = processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0].strip()
    print('new generate function with constraint:', new_generated_text)

if __name__ == "__main__":
    main()