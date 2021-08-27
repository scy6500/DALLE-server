from flask import Flask, request, jsonify
from tqdm import tqdm
import numpy as np

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

# load model

vae = VQGanVAE(None, None)

load_obj = torch.load("./dalle.pt") # model checkpoint : https://github.com/robvanvolt/DALLE-models/tree/main/models/taming_transformer
dalle_params, _, weights = load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights')
dalle_params.pop('vae', None)  # cleanup later

dalle = DALLE(vae=vae, **dalle_params).cuda()

dalle.load_state_dict(weights)

batch_size = 4

top_k = 0.9

# generate images

image_size = vae.image_size


@app.route('/generate', methods=['POST'])
def make_images():
    try:
        json_data = request.get_json()
        text_input = json_data["text"]
        num_images = json_data["num_images"]

    except Exception as e:
        return jsonify({'Error': 'Invalid request'}), 500

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

        return jsonify(response)

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'Error': e}), 500


@app.route('/healthz', methods=["GET"])
def health_check():
    return "Health", 200


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
