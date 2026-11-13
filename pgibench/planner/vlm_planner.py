import re
import os
import time
import numpy as np
import cv2
import json
from pgibench.planner.planner_utils import (
    fix_json, local_image_to_data_url, template, template_lang, truncate_image_history)
from pgibench.evaluator.config.system_prompts import eqa_system_prompt
from pgibench.evaluator.config.action_modes import get_profile
from pgibench.planner.remote_model import RemoteModel
from pgibench.planner.custom_model import CustomModel
from pgibench.main import logger


def _format_waypoint(w):
    """Render one waypoint entry for the '## Available Waypoints' section.

    Accepts the dict form produced by PGIAlfEnv.get_available_waypoints()
    ({name, x, z, rotation, direction, horizon}) and formats it with the same
    position/facing convention used by '## Agent Current State' (ground-plane
    (x, z) only). Falls back to a bare '- <name>' for the legacy string form.
    """
    if isinstance(w, dict):
        return (f"- {w['name']} (position: ({w['x']:.2f}, {w['z']:.2f}); "
                f"facing: {w['direction']} "
                f"(rotation {float(w['rotation']):.1f}°, "
                f"horizon {float(w['horizon']):.1f}°))")
    return f"- {w}"


class VLMPlanner():

    def __init__(self,
                 model_name,
                 model_type,
                 actions,
                 system_prompt,
                 obs_key='head_rgb',
                 chat_history=False,
                 language_only=False,
                 use_feedback=True,
                 multistep=0,
                 action_display_mode='visible',
                 object_reference_mode=None,
                 nav_mode=None,
                 image_history_keep=5,
                 eqa_image_history_keep=5,
                 kwargs={}):
        self.model_name = model_name
        self.obs_key = obs_key
        self.chat_history = chat_history  # whether to includ all chat history for prompting
        self.profile = get_profile(action_display_mode, object_reference_mode)
        # command mode uses the profile's own system prompt; index modes keep the
        # caller-supplied system_prompt (alfred_system_prompt).
        self.system_prompt = (self.profile.system_prompt
                              if self.profile.output_kind == 'command' else system_prompt)
        # command mode: render the {nav_section}/{nav_strategy_section} placeholders
        # (both depend on the env's nav_mode list, which is not part of the profile key).
        if self.profile.output_kind == 'command':
            from pgibench.evaluator.config.system_prompts import (
                get_command_nav_section, get_command_nav_strategy)
            self.nav_mode = list(nav_mode) if nav_mode else ['default_waypoint']
            self.system_prompt = self.system_prompt.replace(
                '{nav_section}', get_command_nav_section(self.nav_mode))
            self.system_prompt = self.system_prompt.replace(
                '{nav_strategy_section}', get_command_nav_strategy(self.nav_mode))
        else:
            self.nav_mode = []
        self.object_catalog = ''
        self.nav_waypoints = []
        self.set_actions(actions)
        self.model_type = model_type
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, language_only=language_only)

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.image_history_keep = image_history_keep
        self.eqa_image_history_keep = eqa_image_history_keep
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')
        self.episode_messages = []

    def set_actions(self, actions, object_catalog=None, nav_waypoints=None):
        self.actions = actions
        self.nav_waypoints = list(nav_waypoints) if nav_waypoints else []
        if self.profile.output_kind == 'command':
            # command mode: no enumerated action list; the object catalog (desc/type)
            # is shown instead. Other reference schemes rely on the annotated image.
            self.object_catalog = object_catalog or ''
            self.available_action_str = self.object_catalog
        else:
            self.object_catalog = ''
            self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(
                available_actions[i])
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str

    def process_prompt(self,
                       user_instruction,
                       prev_act_feedback=[],
                       agent_position=None):
        user_instruction = user_instruction.rstrip('.')

        # Format agent position information
        agent_info = ""
        if agent_position is not None:
            pos = agent_position.get('position', {})
            agent_info = f"Agent Position: ({pos.get('x', 'N/A'):.2f}, {pos.get('z', 'N/A'):.2f}), "

            # Add waypoint information
            closest_path_point = agent_position.get('closest_path_point')
            is_on_path_point = agent_position.get('is_on_path_point', False)
            relative_direction = agent_position.get(
                'relative_direction_to_path_point')

            if closest_path_point is not None:
                if is_on_path_point:
                    agent_info += f"Location: at {closest_path_point}, "
                else:
                    agent_info += f", Location: {relative_direction} of '{closest_path_point}'"
            # Facing shows both the cardinal direction and the numeric heading
            # (rotation_y: 0°=North, increases clockwise) plus the vertical
            # camera horizon so the agent can reason about exact orientation.
            direction = agent_position.get('direction', 'N/A')
            rotation_y = agent_position.get('rotation_y', None)
            horizon = agent_position.get('horizon', None)
            heading = [f"Facing: {direction}"]
            if rotation_y is not None:
                heading.append(f"rotation {float(rotation_y):.1f}°")
            if horizon is not None:
                heading.append(f"horizon {float(horizon):.1f}°")
            agent_info += heading[0]
            if len(heading) > 1:
                agent_info += f" ({', '.join(heading[1:])})"

        # Determine whether the previous action succeeded, for the explicit
        # "Last Action Result" section. The feedback tuple layout is
        # [action_id, action_description, env_feedback, agent_position,
        #  last_action_success]; last_action_success is 1.0/0.0/None.
        last_action_result = None
        if len(prev_act_feedback) > 0:
            last_fb = prev_act_feedback[-1]
            last_success = last_fb[-1] if len(last_fb) > 4 else None
            if last_success is not None:
                last_action_result = (
                    f"Last action SUCCEEDED. ({last_fb[2]})"
                    if float(last_success) > 0
                    else f"Last action FAILED. ({last_fb[2]})")

        # When chat_history carries the conversation, skip the textual action-history
        # recap to avoid duplicating tokens with the message list.
        include_text_history = (len(prev_act_feedback) > 0) and (not self.chat_history)

        if not include_text_history:
            # Use system prompt as-is (no placeholders)

            # Add current task
            prompt = f'\n\n## Current Task: {user_instruction}.\n\n'

            # Explicit last-action success/failure result. Shown on every step
            # after the first so the agent can immediately tell whether its
            # previous action succeeded without parsing the history text.
            if last_action_result is not None:
                prompt += f'## Last Action Result\n{last_action_result}\n\n'

            # Agent Current State is always present (command mode relies on it;
            # other modes also benefit). Empty agent_info still renders the section
            # so the section is guaranteed in every mode.
            prompt += f'## Agent Current State\n{agent_info or "unknown"}\n\n'

            # Available Actions (enumerated list) is only for index modes; command
            # mode uses the action grammar + object reference convention in the
            # system prompt instead, so it is deliberately omitted here.
            if self.profile.output_kind != 'command':
                prompt += '## Available Actions\n\n'
                prompt += 'aciotn id -1: Done, '
                prompt += self.available_action_str
                prompt += '\n\n'

            # id reference mode lists each visible object's ID + normalised bbox
            # [x1,y1,x2,y2] in [0,1] (provided by env.get_object_catalog(mode='id')).
            if (self.profile.output_kind == 'command'
                    and self.profile.object_reference_mode == 'id'
                    and self.object_catalog):
                prompt += '## Visible Objects (id, normalized bbox [x1,y1,x2,y2])\n'
                prompt += self.object_catalog
                prompt += '\n\n'

            if self.profile.output_kind == 'command' and self.nav_waypoints:
                prompt += '\n\n## Available Waypoints\n'
                prompt += '\n'.join(_format_waypoint(w) for w in self.nav_waypoints)
                prompt += '\n'
            prompt += f"\n\nYour response must be in JSON format as described in the system prompt."

        else:
            # chat_history=False and not first turn: inline textual action history
            # (when chat_history=True the conversation is carried by the message list)

            # Add current task
            prompt = f'\n\n## Current Task: {user_instruction}.\n\n'

            # Explicit last-action success/failure result (mirrors the no-history
            # branch); the full textual history follows below.
            if last_action_result is not None:
                prompt += f'## Last Action Result\n{last_action_result}\n\n'

            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                # Feedback tuple: [action_id, action_description, env_feedback,
                #                  agent_position, last_action_success]
                # agent_position sits at index 3 (None when not recorded).
                fb_pos = action_feedback[3] if len(action_feedback) > 3 else None
                agent_pos_str = ""
                if fb_pos is not None:
                    pos = fb_pos.get('position', {})
                    agent_pos_str = f"({pos.get('x', 'N/A'):.2f}, {pos.get('z', 'N/A'):.2f})"

                    # Add waypoint information
                    closest_path_point = fb_pos.get('closest_path_point')
                    is_on_path_point = fb_pos.get('is_on_path_point', False)
                    relative_direction = fb_pos.get(
                        'relative_direction_to_path_point')

                    if closest_path_point is not None:
                        if is_on_path_point:
                            agent_pos_str += f", at {closest_path_point}"
                        else:
                            agent_pos_str += f", {relative_direction} of {closest_path_point}"
                    # Include the numeric heading alongside the cardinal direction.
                    _dir = fb_pos.get('direction', 'N/A')
                    _rot = fb_pos.get('rotation_y', None)
                    _hor = fb_pos.get('horizon', None)
                    _heading = [f"facing {_dir}"]
                    if _rot is not None:
                        _heading.append(f"rotation {float(_rot):.1f}°")
                    if _hor is not None:
                        _heading.append(f"horizon {float(_hor):.1f}°")
                    agent_pos_str += ", " + _heading[0]
                    if len(_heading) > 1:
                        agent_pos_str += f" ({', '.join(_heading[1:])})"

                if self.use_feedback:
                    prompt += '\nStep {}. Agent Position: {}. Action: ({}, {}). Env feedback: {}'.format(
                        i, agent_pos_str, action_feedback[0],
                        action_feedback[1], action_feedback[2])
                else:
                    a0 = action_feedback[0]
                    if isinstance(a0, int) and 0 <= a0 < len(self.actions):
                        act_desc = self.actions[a0]
                    else:
                        act_desc = action_feedback[1] if len(
                            action_feedback) > 1 else str(a0)
                    prompt += '\nStep {}. Agent Position: {}. Action: ({}).'.format(
                        i, agent_pos_str, act_desc)

            # Add agent current position
            prompt += f'\n\n## Agent Current State\n{agent_info or "unknown"}\n\n'

            # Available Actions (enumerated list) is only for index modes.
            if self.profile.output_kind != 'command':
                prompt += '## Available Actions\n\n'
                prompt += 'aciotn id -1: Done, '
                prompt += self.available_action_str
                prompt += '\n\n'

            # id reference mode lists each visible object's ID + normalised bbox
            # [x1,y1,x2,y2] in [0,1] (provided by env.get_object_catalog(mode='id')).
            if (self.profile.output_kind == 'command'
                    and self.profile.object_reference_mode == 'id'
                    and self.object_catalog):
                prompt += '## Visible Objects (id, normalized bbox [x1,y1,x2,y2])\n'
                prompt += self.object_catalog
                prompt += '\n\n'

            if self.profile.output_kind == 'command' and self.nav_waypoints:
                prompt += '\n\n## Available Waypoints\n'
                prompt += '\n'.join(_format_waypoint(w) for w in self.nav_waypoints)
                prompt += '\n'
            prompt += f'''\n\nConsidering the above action history and current task, your response must be in JSON format as described in the system prompt.'''

        return prompt

    def get_message(self, image, prompt, messages=[], syetem_prompt=None):
        if syetem_prompt is None:
            syetem_prompt = self.system_prompt
            syetem_prompt = [{
                "role":
                "system",
                "content": [{
                    "type": "text",
                    "text": syetem_prompt
                }],
            }]

        # remove original system message to avoid duplication
        if len(messages) > 0:
            messages = messages[1:]

        if self.language_only:
            return syetem_prompt + messages + [{
                "role":
                "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }],
            }]

        # handle image input
        if type(image) == str:
            image_path = image
        else:
            image_path = './evaluation/tmp_{}.png'.format(len(messages) // 2)
            cv2.imwrite(image_path, image)

        if self.multistep:  # handle multiple images
            ind = int(image_path.split('step_')[-1].strip('.png'))
            content = [{"type": "text", "text": prompt}]
            for i in range(max(ind - self.multistep + 1, 0), ind + 1):
                temp_path = ''.join(
                    image_path.split('step_')[:-1]) + f'step_{str(i)}.png'
                temp_data_url = local_image_to_data_url(image_path=temp_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": temp_data_url,
                    }
                })
        else:
            data_url = local_image_to_data_url(image_path=image_path)
            content = [{
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                }
            }, {
                "type": "text",
                "text": prompt
            }]

        return syetem_prompt + messages + [{
            "role": "user",
            "content": content,
        }]

    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def language_to_action(self, output_text):
        pattern = r'\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action

    def json_to_action(self, output_text, json_key='executable_plan'):
        try:
            try:
                output_text = json.loads(output_text)
            except json.JSONDecodeError:
                fixed_output = fix_json(output_text)
                output_text = json.loads(fixed_output)
            action_list = output_text.get(json_key, [])

            # Validate: must contain exactly 1 action
            if len(action_list) == 0:
                print('empty plan, stopping here')
                action = -1
            elif len(action_list) > 1:
                print(
                    f'Warning: Got {len(action_list)} actions, using only the first one'
                )
                action = int(action_list[0][self.action_key])
            else:
                action = int(action_list[0][self.action_key])

            # Validate action ID
            if action >= len(self.actions) or action < 0:
                print('found invalid action')
                action = -1

        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            action = -1
        except Exception as e:
            print("An unexpected error occurred:", e)
            self.output_json_error += 1
            action = -1

        return action, output_text

    def parse_output(self, out):
        """Dispatch parsing by output_kind.

        Returns an int action_id (index modes), or a command dict
        {'command':..., 'target_bbox':..., 'target_point':...} (command mode),
        or -1 on parse failure / empty plan.
        """
        if self.profile.output_kind == 'command':
            action, _ = self.command_to_action(out)
            return action
        action, _ = self.json_to_action(out)
        return action

    def command_to_action(self, output_text, json_key='executable_plan'):
        """Parse a command-mode JSON response.

        Returns (action_dict | -1, parsed_data). action_dict has shape
        {'command': str, 'target_bbox': list|None, 'target_point': list|None}.
        """
        try:
            try:
                data = json.loads(output_text)
            except json.JSONDecodeError:
                data = json.loads(fix_json(output_text))
            plan = data.get(json_key, [])
            if not plan:
                return -1, data
            item = plan[0]
            cmd = item.get('action')
            if not cmd or not isinstance(cmd, str):
                return -1, data
            # Only the field relevant to the active reference mode is consumed, so
            # a stray field in another mode is ignored rather than confusing the
            # downstream resolver.
            ref_mode = self.profile.object_reference_mode
            target_bbox = item.get('target_bbox') if ref_mode == 'bbox' else None
            target_point = item.get('target_point') if ref_mode == 'point' else None
            return ({
                'command': cmd.strip(),
                'target_bbox': target_bbox,
                'target_point': target_point,
            }, data)
        except Exception as e:
            print("command_to_action failed:", e)
            self.output_json_error += 1
            return -1, None

    def act_custom(self, prompt, obs):
        assert type(obs) == str  # input image path
        out = self.model.respond(prompt, obs)
        # fix common generated json errors
        out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        action = self.parse_output(out)
        self.planner_steps += 1
        return action, out

    def act(self, observation, user_instruction, agent_position=None):
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation  # input image path

        prompt = self.process_prompt(
            user_instruction,
            prev_act_feedback=self.episode_act_feedback,
            agent_position=agent_position)
        # print(f"Prompt for Planner:\n{prompt}\n")

        # # Claude does not support JSON Schema, add format into prompt
        # if 'claude' in self.model_name or self.model_type == 'custom':
        #     prompt = prompt + template_lang if self.language_only else prompt + template

        if self.model_type == 'custom':
            return self.act_custom(prompt, obs)

        # if len(self.episode_messages) == 0:
        #     self.episode_messages = self.get_message(obs, prompt)
        # else:
        if self.chat_history:
            self.episode_messages = self.get_message(obs, prompt,
                                                     self.episode_messages)
        else:
            self.episode_messages = self.get_message(obs, prompt)

        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        send_messages = truncate_image_history(self.episode_messages,
                                               self.image_history_keep)
        try:
            out = self.model.respond(send_messages,
                                     response_format=self.profile.response_format_key)
        except Exception as e:
            print("An unexpected error occurred:", e)
            time.sleep(60)
            out = self.model.respond(send_messages,
                                     response_format=self.profile.response_format_key)
        logger.debug(f"Model Output:\n{out}\n")

        if self.chat_history:
            self.episode_messages.append({
                "role":
                "assistant",
                "content": [{
                    "type": "text",
                    "text": out
                }],
            })
        action = self.parse_output(out)
        self.planner_steps += 1
        return action, out

    def update_info(self, info):
        """Update episode feedback history."""
        agent_position = info.get('agent_position', None)
        last_success = info.get('last_action_success', None)
        # Feedback tuple layout:
        #   [action_id, action_description, env_feedback, agent_position,
        #    last_action_success]
        # The trailing last_action_success (1.0/0.0/None) lets process_prompt
        # render an explicit "last action succeeded/failed" section without
        # having to parse the free-text env_feedback.
        if agent_position is not None:
            self.episode_act_feedback.append([
                info['action_id'], info['action_description'],
                info['env_feedback'], agent_position, last_success
            ])
        else:
            self.episode_act_feedback.append([
                info['action_id'], info['action_description'],
                info['env_feedback'], None, last_success
            ])

    def ask_eqa_question(self, question, observation_path):
        """
        Ask an EQA question using current context but with EQA-specific system prompt.
        Creates a temporary "fork" of the message context.

        Args:
            question: EQA question dict with 'question_text' and 'options'
            observation_path: Path to current observation image

        Returns:
            (answer, reasoning) tuple
        """

        # 1. Save current context
        original_messages = self.episode_messages.copy()

        # 2. Build EQA-specific prompt
        options_text = "\n".join(f"{k}: {v}"
                                 for k, v in question['options'].items())
        eqa_prompt = f'## Question: {question["question_text"]}\n\nOptions:{options_text}\n\nProvide your answer as a JSON object with "answer" (option letter) and "reasoning" fields.'

        # 3. Build message with EQA system prompt
        # Create system message with EQA prompt
        eqa_system_msg = [{
            "role":
            "system",
            "content": [{
                "type": "text",
                "text": eqa_system_prompt
            }]
        }]

        # Include conversation history for context (skip original system message)
        history_messages = self.episode_messages[1:] if len(
            self.episode_messages) > 1 else []

        # Build final EQA messages - system message must come first
        eqa_messages = eqa_system_msg + history_messages + [{
            "role":
            "user",
            "content": [{
                "type": "text",
                "text": eqa_prompt
            }]
        }]

        # 4. Get response with EQA-specific format
        reasoning = ''
        try:
            # Get EQA response format
            eqa_send = truncate_image_history(eqa_messages, self.eqa_image_history_keep)
            response = self.model.respond(eqa_send, response_format='eqa')

            # Try to parse as JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to fix JSON if parsing fails
                fixed_output = fix_json(response)
                result = json.loads(fixed_output)
            answer = result.get('answer', '')
            reasoning = result.get('reasoning', '')
        except Exception as e:
            logger.warning(f"EQA question failed: {e}")
            answer = ''
            reasoning = reasoning

        # 5. Restore original context
        self.episode_messages = original_messages

        return answer, reasoning
