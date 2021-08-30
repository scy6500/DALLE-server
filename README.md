# DALL-E Pytorch

Creating Images from Text

Github: [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch)


### Post parameter

    text: Text for the image you want to create
    num_images: Number of images you want


### Output format

    [Base64]


## * With CLI *

### Input example


    curl https://main-dalle-server-scy6500.endpoint.ainize.ai/generate \ 
    --header "Content-Type: application/json" --request POST \ 
    --data '{"text":"apple", "num_images":1}'
    

### Output example


    ["/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."]


## * With swagger *

API page: [Ainize](https://ainize.ai/scy6500/DALLE-server?branch=main)

## * With a Demo *

Demo page: [End-point](https://main-dalle-client-scy6500.endpoint.ainize.ai/)
