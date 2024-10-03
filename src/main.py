from io import BytesIO
from multiprocessing.connection import Listener
from os import chmod, remove
from os.path import abspath, exists
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from PIL.JpegImagePlugin import JpegImageFile
from pipelines.models import TextToImageRequest
from PIL.Image import Image
import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from pipelines.models import TextToImageRequest
from torch import Generator
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)

import packaging.version
import torch
from pipeline import load_pipeline, infer

SOCKET = abspath(Path(__file__).parent.parent / "inferences.sock")

def infer2(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator("cuda").manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=9
    ).images[0]

def main():
    print(f"Loading pipeline")
    pipeline = load_pipeline()
    print(f"Pipeline loaded, creating socket at '{SOCKET}'")

    if exists(SOCKET):
        remove(SOCKET)

    with Listener(SOCKET) as listener:
        chmod(SOCKET, 0o777)

        print(f"Awaiting connections")
        with listener.accept() as connection:
            print(f"Connected")

            while True:
                try:
                    request = TextToImageRequest.model_validate_json(connection.recv_bytes().decode("utf-8"))
                except EOFError:
                    print(f"Inference socket exiting")
                    return
                image = infer2(request, pipeline)

                data = BytesIO()
                image.save(data, format=JpegImageFile.format)

                packet = data.getvalue()

                connection.send_bytes(packet)


if __name__ == '__main__':
    main()
