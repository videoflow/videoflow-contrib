'''
Vision-language frame captioner built on Hugging Face transformers (default
model: Qwen/Qwen2.5-VL-7B-Instruct) — the reference multi-GPU component
(videoflow RFC 0003).

A 7B-parameter VLM in bf16 does not fit on one small GPU, so ``open()`` loads
the model with ``device_map = 'auto'`` and Hugging Face shards it across every
visible device. That is the entire multi-GPU story, and it works because of the
visibility contract the framework guarantees: inside a worker the visible GPUs
are exactly the granted GPUs, numbered ``cuda:0..N-1`` with
``N == self.gpu_count`` — on Kubernetes via the device plugin (the pod requests
``nvidia.com/gpu: N`` and sees only those devices), locally via the engine's
``CUDA_VISIBLE_DEVICES`` partitioning. The node body therefore contains zero
device arithmetic: it declares a grant (``gpu_count``, defaulted to 2 by the
descriptor's ``spec.resources.gpu.count``) and lets HF place the shards on
whatever was granted. The descriptor count is a default, not a floor — a
quantized or smaller model runs fine with ``gpu_count = 1``, or on CPU.

The torch/transformers imports are lazy (inside ``open()``) so this module
imports, the descriptor validates, and the wheel builds without the ML stack
installed.
'''
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image
from videoflow.core.constants import GPU
from videoflow.core.node import ProcessorNode
from videoflow.utils.system import granted_gpus

logger = logging.getLogger(__package__)

DEFAULT_MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
DEFAULT_PROMPT = 'Describe this image in one sentence.'


def build_chat(prompt: str) -> list[dict[str, Any]]:
    '''
    Builds the one-turn chat-template input the HF processor expects: a single
    user message carrying an image slot followed by the instruction text.
    Pure so it is unit-testable without the model.

    - Arguments:
        - prompt: the instruction sent along with the frame.
    '''
    return [{
        'role': 'user',
        'content': [
            {'type': 'image'},
            {'type': 'text', 'text': prompt},
        ],
    }]


def clean_caption(text: str) -> str:
    '''
    Normalizes a decoded generation into a single-line caption: collapses every
    whitespace run (including newlines) to one space and strips one pair of
    matching wrapping quotes, which chatty VLMs like to add around short
    answers. Pure so it is unit-testable without the model.
    '''
    caption = ' '.join(text.split())
    if len(caption) >= 2 and caption[0] == caption[-1] and caption[0] in ('"', "'"):
        caption = caption[1:-1].strip()
    return caption


def to_pil_rgb(frame: np.ndarray) -> Image.Image:
    '''
    Converts an ``(h, w, 3)`` BGR uint8 frame (the repo's frame convention) into
    the RGB PIL image the HF processor expects. The channel flip is a contiguous
    copy, not a view — tensor conversion downstream rejects negative strides.
    '''
    return Image.fromarray(np.ascontiguousarray(frame[..., ::-1]))


class VlmCaptioner(ProcessorNode):
    '''
    Captions video frames with a Hugging Face vision-language model.

    Takes an ``(h, w, 3)`` BGR frame and returns a one-line caption string.
    It never returns ``None`` — a ``None`` from ``process()`` would still be
    published downstream, not dropped.

    - Arguments:
        - model_id: Hugging Face model id of an image-text-to-text model.
        - prompt: instruction sent with every frame.
        - max_new_tokens: generation budget per caption.
        - nb_tasks: replicas; the captioner is stateless, so replication and \
            ``partition_by`` are both safe.
        - device_type: GPU by default. The descriptor defaults ``gpu_count`` \
            to 2 (``spec.resources.gpu.count``); pass ``gpu_count = 1`` for a \
            model that fits one device, or ``device_type = CPU`` to run \
            (slowly) without any.
    '''

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, prompt: str = DEFAULT_PROMPT,
                 max_new_tokens: int = 64, nb_tasks: int = 1, device_type: str = GPU,
                 **kwargs: Any) -> None:
        self._model_id = model_id
        self._prompt = prompt
        self._max_new_tokens = int(max_new_tokens)
        self._model: Any = None        # set in open(), released in close()
        self._processor: Any = None    # set in open(), released in close()
        super().__init__(nb_tasks=nb_tasks, device_type=device_type, **kwargs)

    def open(self) -> None:
        # The one sanctioned function-level import: torch/transformers stay out
        # of module scope so descriptor validation and wheel builds work
        # without the ML stack installed.
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if self._device_type == GPU:
            logger.info('loading %s across %d granted GPU(s) %s',
                        self._model_id, self.gpu_count, granted_gpus())
            # device_map='auto' shards across exactly the visible devices, and
            # the visible devices are exactly the granted ones — cuda:0..N-1,
            # N == self.gpu_count (RFC 0003). No device arithmetic anywhere.
            device_map = 'auto'
            dtype = torch.bfloat16
        else:
            device_map = 'cpu'
            dtype = torch.float32
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_id, torch_dtype=dtype, device_map=device_map).eval()

    def process(self, frame: np.ndarray) -> str:
        '''
        - Arguments:
            - frame: ``(h, w, 3)`` BGR frame.
        - Returns: the caption string.
        '''
        import torch  # already loaded by open(); function-level to keep the module framework-free
        image = to_pil_rgb(frame)
        text = self._processor.apply_chat_template(
            build_chat(self._prompt), tokenize=False, add_generation_prompt=True)
        # .to(self._model.device) targets the model's entry device; with a
        # sharded model that is where HF placed the embeddings — still not our
        # arithmetic to do.
        inputs = self._processor(text=[text], images=[image],
                                 return_tensors='pt').to(self._model.device)
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        generated = output_ids[:, inputs['input_ids'].shape[1]:]
        decoded = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return clean_caption(decoded)

    def close(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch  # lazy, mirrors open(); absent on a build-only host
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
