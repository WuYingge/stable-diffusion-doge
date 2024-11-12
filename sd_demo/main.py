import datetime
import os
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel


def main():
    print(f"token is {os.getenv("HUGGINGFACE_TOKEN", "")}")
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium",
                                                    torch_dtype=torch.bfloat16,
                                                    token=os.getenv("HUGGINGFACE_TOKEN", ""),
                                                    )
    print(f"success load {pipe}")

    pipe.to("cuda")
    print(f"success too cuda")

    image = pipe("A cute cow-patterned cat is basking in the sunshine on the windowsill.").images[0]
    print(f"sucess get image {image}")

    image.save(f"tmp/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.jpg")


if __name__ == "__main__":
    main()
