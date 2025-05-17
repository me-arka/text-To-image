# text-To-image
!pip install diffusers transformers torch 
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
).to("cpu")
prompt = "A majestic lion standing on a cliff during sunset, ultra detailed, digital art"
guidance_scale = 7.5
num_inference_steps = 50

    result = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
image = result.images[0]
plt.imshow(image)
plt.axis("off")
plt.title("Generated Image")
plt.show()
image.save("generated_image.png")
print("Image saved as 'generated_image.png'")
