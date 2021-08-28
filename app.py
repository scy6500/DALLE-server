from flask import Flask, request, jsonify
from tqdm import tqdm
import numpy as np
from queue import Queue, Empty
from threading import Thread
import time

# torch

import torch

from einops import repeat

# vision imports

from PIL import Image

# dalle related classes and utils

from dalle_pytorch import VQGanVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer

import base64
from io import BytesIO

app = Flask(__name__)

requests_queue = Queue()  # request queue.
REQUEST_BATCH_SIZE = 4  # max request size.
CHECK_INTERVAL = 0.1

# load model

vae = VQGanVAE(None, None)

load_obj = torch.load("./16L_64HD_8H_512I_128T_cc12m_cc3m_3E.pt")  # model checkpoint : https://github.com/robvanvolt/DALLE-models/tree/main/models/taming_transformer
dalle_params, _, weights = load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights')
dalle_params.pop('vae', None)  # cleanup later

dalle = DALLE(vae=vae, **dalle_params).cuda()

dalle.load_state_dict(weights)

batch_size = 4

top_k = 0.9

# generate images

image_size = vae.image_size


def handle_requests_by_batch():
    while True:
        request_batch = []

        while not (len(request_batch) >= REQUEST_BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in request_batch:
                try:
                    requests["output"] = make_images(requests['input'][0], requests['input'][1])

                except Exception as e:
                    requests["output"] = e


handler = Thread(target=handle_requests_by_batch).start()


def make_images(text_input, num_images):
    try:
        text = tokenizer.tokenize([text_input], dalle.text_seq_len).cuda()

        text = repeat(text, '() n -> b n', b=num_images)

        outputs = []
        for text_chunk in tqdm(text.split(batch_size), desc=f'generating images for - {text}'):
            output = dalle.generate_images(text_chunk, filter_thres=top_k)
            outputs.append(output)

        outputs = torch.cat(outputs)

        response = []

        for i, image in tqdm(enumerate(outputs), desc='saving images'):
            np_image = np.moveaxis(image.cpu().numpy(), 0, -1)
            formatted = (np_image * 255).astype('uint8')

            img = Image.fromarray(formatted)

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            response.append(img_str)

        return response

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'Error': e}), 500


@app.route('/generate', methods=['POST'])
def generate():
    if requests_queue.qsize() > REQUEST_BATCH_SIZE:
        return jsonify({'Error': 'Too Many Requests. Please try again later'}), 429

    try:
        args = []
        json_data = request.get_json()
        text_input = json_data["text"]
        num_images = json_data["num_images"]

        if num_images > 10:
            return jsonify({'Error': 'Too many images requested. Request no more than 10'}), 500

        args.append(text_input)
        args.append(num_images)

    except Exception as e:
        return jsonify({'Error': 'Invalid request'}), 500

    req = {'input': args}
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return jsonify(req['output'])


@app.route('/healthz', methods=["GET"])
def health_check():
    return "Health", 200


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
