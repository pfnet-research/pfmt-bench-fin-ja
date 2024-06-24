import json
import warnings

import fastchat.conversation
from fastchat.conversation import Conversation
from fastchat.conversation import SeparatorStyle
from fastchat.conversation import register_conv_template

fastchat.conversation.conv_templates = {}

# you should translate the roles to Japanese from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
register_conv_template(
    Conversation(
        name="one_shot",
        system_message="あなたは優秀なアシスタントです。",
        roles=("ユーザー", "アシスタント"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n### ",
    )
)

register_conv_template(
    Conversation(
        name="claude",
        system_message="あなたは優秀なアシスタントです。",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
        max_image_size_mb=5 / 1.35,
    )
)

register_conv_template(
    Conversation(
        name="claude-3-haiku-20240307",
        system_message="あなたは優秀なアシスタントです。",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.35,
    )
)

register_conv_template(
    Conversation(
        name="claude-3-sonnet-20240229",
        system_message="あなたは優秀なアシスタントです。",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.35,
    )
)

register_conv_template(
    Conversation(
        name="claude-3-opus-20240229",
        system_message="あなたは優秀なアシスタントです。",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.35,
    )
)

register_conv_template(
    Conversation(
        name="chatgpt",
        system_message="あなたは優秀なアシスタントです。",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=None,  # OpenAI does auto-resizing
    )
)

register_conv_template(
    Conversation(
        name="llama-3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
)

register_conv_template(
    Conversation(
        name="llama-3-70b-pfn-qfin",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
)

register_conv_template(
    Conversation(
        name="llama-3-70b-pfn-qfin-inst-merge",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
)


json._dumps = json.dumps


def json_dumps_japanese(*args, **kwargs):
    kwargs["ensure_ascii"] = False
    return json._dumps(*args, **kwargs)


json.dumps = json_dumps_japanese
