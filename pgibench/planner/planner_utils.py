import os
import re
import base64
from mimetypes import guess_type
import typing_extensions as typing
from pydantic import BaseModel, Field

template_lang = '''\
The output json format should be {'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, \
2. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, \
3. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!
'''

template = '''
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, 
3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, 
4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''


def fix_json(json_str):
    """
    Locates the substring between the keys "reasoning_and_reflection" and "language_plan"
    and escapes any inner double quotes that are not already escaped.

    Also handles quotes in other JSON string fields: answer, visual_state_description, language_plan.
    """
    # first fix common errors
    json_str = json_str.replace("'", '"')
    json_str = json_str.replace('\"s ', "\'s ")
    json_str = json_str.replace('\"re ', "\'re ")
    json_str = json_str.replace('\"ll ', "\'ll ")
    json_str = json_str.replace('\"t ', "\'t ")
    json_str = json_str.replace('\"d ', "\'d ")
    json_str = json_str.replace('\"m ', "\'m ")
    json_str = json_str.replace('\"ve ', "\'ve ")
    json_str = json_str.replace('```json', '').replace('```', '')

    fixed_json = json_str

    # Known JSON keys that may contain unescaped quotes in their string values
    for key in [
            'answer', 'visual_state_description', 'reasoning',
            'reasoning_and_reflection', 'language_plan'
    ]:
        # Pattern to match: "key": "anything until closing quote"
        # We use a state machine approach to find and fix each occurrence
        key_pattern = f'"{key}"\\s*:\\s*"'
        pos = 0

        while True:
            match = re.search(key_pattern, fixed_json[pos:], re.DOTALL)
            if not match:
                break

            # Find the opening quote position of the value
            value_start = pos + match.end() - 1  # Position of the opening "

            # Find the closing quote by scanning forward
            # We need to find a " that's followed by either , or } or ]
            i = value_start + 1
            while i < len(fixed_json):
                if fixed_json[i] == '"' and fixed_json[i - 1] != '\\':
                    # Check if this is followed by , or } or ]
                    next_chars = fixed_json[i + 1:i + 3].lstrip()
                    if next_chars.startswith(',') or next_chars.startswith(
                            '}') or next_chars.startswith(']'):
                        # This is the closing quote
                        value_end = i
                        # Extract and fix the value
                        value = fixed_json[value_start + 1:value_end]
                        fixed_value = value.replace('"', '\\"')
                        # Replace in the string
                        fixed_json = fixed_json[:value_start +
                                                1] + fixed_value + fixed_json[
                                                    value_end:]
                        break
                i += 1

            # Move past this key for next iteration
            pos = pos + match.end()

    return fixed_json


def convert_format_2claude(messages):
    new_messages = []

    for message in messages:
        if message["role"] == "user":
            new_content = []

            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)

    return new_messages


class ExecutableAction(typing.TypedDict):
    action_id: int
    action_name: str


class ActionPlan(BaseModel):
    visual_state_description: str
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction]


class ExecutableAction_lang(typing.TypedDict):
    action_id: int
    action_name: str


class ActionPlan_lang(BaseModel):
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction_lang]


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(
            image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def truncate_image_history(messages: list, keep_last_k: int):
    """Image sliding-window: keep all text, keep images only in the most recent K user messages.

    system/assistant messages are never modified. For user messages older than the most
    recent keep_last_k, image_url content blocks are stripped while text blocks are kept,
    so the VLM retains the full textual history (action descriptions, env feedback, agent
    state) but only sees the recent K observation frames. Returns a new list — the caller's
    episode_messages stays intact for persistence and EQA context reuse.

    keep_last_k <= 0 (incl. negative) disables truncation; all images are kept.
    This lets callers opt out of the sliding window by passing a negative K.
    """
    if not messages or keep_last_k is None or keep_last_k <= 0:
        return messages

    user_indices = [i for i, m in enumerate(messages) if m.get('role') == 'user']
    keep = set(user_indices[-keep_last_k:])

    truncated = []
    for i, m in enumerate(messages):
        if m.get('role') == 'user' and i not in keep:
            truncated.append({
                "role": "user",
                "content": [c for c in m.get("content", []) if c.get("type") != "image_url"],
            })
        else:
            truncated.append(m)
    return truncated
