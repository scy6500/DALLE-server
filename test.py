from dalle_pytorch import VQGanVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer

# load model

vae = VQGanVAE(None, None)

load_obj = torch.load("./16L_64HD_8H_512I_128T_cc12m_cc3m_3E.pt")  # model checkpoint : https://github.com/robvanvolt/DALLE-models/tree/main/models/taming_transformer
dalle_params, _, weights = load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights')
dalle_params.pop('vae', None)  # cleanup later

dalle = DALLE(vae=vae, **dalle_params).cuda()

dalle.load_state_dict(weights)

batch_size = 4

top_k = 0.9

text_input = "apple"

# generate images

image_size = vae.image_size

text = tokenizer.tokenize([text_input], dalle.text_seq_len).cuda()

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

print(response)
