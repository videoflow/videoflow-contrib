'''
Weights-free tests for VlmCaptioner: the pure pre/post-processing helpers and
the worker reconstruction contract (get_params round trip). No torch or
transformers required — the heavy imports live in open(), which these tests
never call.
'''
import json

import numpy as np
from videoflow.core.constants import CPU, GPU
from videoflow_contrib.vlm_caption.captioner import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    VlmCaptioner,
    build_chat,
    clean_caption,
    to_pil_rgb,
)


def test_build_chat_single_user_turn_image_before_text():
    chat = build_chat('What is happening?')
    assert len(chat) == 1 and chat[0]['role'] == 'user'
    content = chat[0]['content']
    assert content[0] == {'type': 'image'}
    assert content[1] == {'type': 'text', 'text': 'What is happening?'}


def test_clean_caption_collapses_whitespace_and_strips_quotes():
    assert clean_caption('  "A dog\n  chasing a ball."  ') == 'A dog chasing a ball.'
    assert clean_caption("'single quoted'") == 'single quoted'
    assert clean_caption('no quotes\nhere') == 'no quotes here'
    assert clean_caption('   ') == ''
    # mismatched quotes are content, not wrapping — left alone
    assert clean_caption('"mismatched\'') == '"mismatched\''


def test_to_pil_rgb_flips_bgr_channels():
    frame = np.zeros((4, 6, 3), np.uint8)
    frame[..., 0] = 255                          # blue channel in BGR
    image = to_pil_rgb(frame)
    assert image.mode == 'RGB' and image.size == (6, 4)
    assert np.asarray(image)[0, 0].tolist() == [0, 0, 255]   # ...is B in RGB


def test_defaults_and_get_params_round_trip():
    node = VlmCaptioner(name='captioner')
    assert node.device_type == GPU               # GPU by default, not hard-pinned
    params = node.get_params()
    assert params['model_id'] == DEFAULT_MODEL_ID
    assert params['prompt'] == DEFAULT_PROMPT
    assert params['max_new_tokens'] == 64
    json.dumps(params)                           # every param JSON-serializable
    rebuilt = VlmCaptioner(**params)             # what the worker does
    assert rebuilt.get_params() == params


def test_scaling_and_device_overrides_flow_through():
    node = VlmCaptioner(model_id='Qwen/Qwen2.5-VL-3B-Instruct', gpu_count=1,
                        device_type=CPU, nb_tasks=2, name='captioner')
    assert node.gpu_count == 1 and node.device_type == CPU and node.nb_tasks == 2
    rebuilt = VlmCaptioner(**node.get_params())
    assert rebuilt.gpu_count == 1
    assert rebuilt.get_params()['model_id'] == 'Qwen/Qwen2.5-VL-3B-Instruct'
