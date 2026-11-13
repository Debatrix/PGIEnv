"""Human evaluation web backend (frontend/backend split).

Each session owns one ``PGIAlfEnv`` instance bound to a dedicated AI2-THOR /
Xvfb display. The human acts through the web UI (typed command + click point /
drag bbox); the backend forwards the action plus normalized [0, 1] coordinates
to ``env.step``.

Results are written per evaluator to
``running/pgi_alfred/{exp_name}/{eval_set}/{evaluator_id}/results/`` using the
same schema as ``pgi_alfred_evaluator.py`` (``evaluator_type='human'``), so the
same analysis scripts work for both.

Launch:
    python -m pgibench.human_eval.server --config pgibench/configs/human_eval.yaml

Prerequisite: one Xvfb display per concurrent session, e.g.
    python -m pgibench.envs.pgi_alfred.scripts.start_xvfb 1
"""

import os
import re
import sys
import json
import time
import base64
import argparse
import threading
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import io

# Allow running as a standalone module (python -m pgibench.human_eval.server):
# server.py -> human_eval -> pgibench -> repo_root (3 levels up).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pgibench.envs.pgi_alfred.PGIAlfEnv import PGIAlfEnv
from pgibench.evaluator.evaluator_utils import make_json_serializable

# ===================== X display allocation =====================
_LOCK = threading.Lock()
_USED_DISPLAYS = set()


def _alloc_display(base=1, max_display=32):
    """Allocate an unused X display number (:N)."""
    with _LOCK:
        for n in range(base, base + max_display):
            if n not in _USED_DISPLAYS:
                _USED_DISPLAYS.add(n)
                return str(n)
    raise RuntimeError(
        "No available X display numbers; make sure Xvfb is running or raise "
        "the limit")


def _free_display(d):
    with _LOCK:
        _USED_DISPLAYS.discard(int(str(d).lstrip(':')))


# ===================== Config loading =====================
def load_config(path):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('down_sample_ratio', 1.0)
    cfg.setdefault('exp_name', 'human_eval')
    cfg.setdefault('max_steps', 100)
    cfg.setdefault('resolution', 500)
    cfg.setdefault('action_display_mode', 'command')
    cfg.setdefault('object_reference_mode', 'point')
    cfg.setdefault('nav_mode', ['created_waypoint', 'object'])
    cfg.setdefault('detection_box', False)
    cfg.setdefault('visual_prompt', False)
    cfg.setdefault('enable_eqa', True)
    cfg.setdefault('instruction_mode', 'micro')
    cfg.setdefault('bbox_iou_threshold', 0.3)
    cfg.setdefault('point_dist_ratio', 0.3)
    cfg.setdefault('eqa_end_episode_questions', -15)
    # Idle sessions whose last request is older than this many seconds are
    # reaped (env closed, display freed) by a background thread.
    cfg.setdefault('session_idle_timeout', 1800)
    return cfg


# ===================== Persistence (aligned with the VLM evaluator) =====================
def _compare_eqa_answer(model_answer, correct_answer, question_type):
    if not model_answer or not correct_answer:
        return False
    model_norm = model_answer.replace(' ', '').upper()
    correct_norm = correct_answer.replace(' ', '').upper()
    if question_type == 'multi_choice':
        return set(model_norm.split(',')) == set(correct_norm.split(','))
    return model_norm == correct_norm


def _episode_idx(env):
    # Identical to pgi_alfred_evaluator.save_episode_metric
    return env._current_episode_num if not len(env.selected_indexes) \
        else env.selected_indexes[env._current_episode_num - 1] + 1


def _frame_to_b64(env):
    frame = env.env.last_event.frame
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _save_episode(env, episode_info):
    episode_idx = _episode_idx(env)
    res_path = os.path.join(env.log_path, 'results')
    os.makedirs(res_path, exist_ok=True)
    episode_info = make_json_serializable(episode_info)
    episode_messages = episode_info.pop('episode_messages', [])
    with open(os.path.join(res_path, f'episode_{episode_idx}_final_res.json'),
              'w', encoding='utf-8') as f:
        json.dump(episode_info, f, ensure_ascii=False, indent=2)
    with open(os.path.join(res_path, f'episode_{episode_idx}_messages.json'),
              'w', encoding='utf-8') as f:
        json.dump(episode_messages, f, ensure_ascii=False, indent=2)
    env.save_thor_tracking(res_path)
    return res_path


# ===================== Session management =====================
class Session:
    def __init__(self, config, eval_set, selected_indexes, evaluator_id,
                 display):
        self.config = dict(config)
        self.eval_set = eval_set
        self.selected_indexes = list(selected_indexes) if selected_indexes else []
        self.evaluator_id = evaluator_id
        self.display = display
        self.env = None
        self.step_history = []
        self.episode_info = {}
        self.eqa_state = {'questions': [], 'results': [], 'current_idx': 0}
        self._action_select_time = 0.0
        self.last_active = time.time()

    def touch(self):
        self.last_active = time.time()

    def start(self):
        # Per-evaluator results directory:
        # running/pgi_alfred/{exp_name}/{eval_set}/{evaluator_id}/
        exp_name = self.config.get('exp_name') or 'human_eval'
        exp_name = f"{exp_name}/{self.eval_set}/{self.evaluator_id}"
        self.env = PGIAlfEnv(
            eval_set=self.eval_set,
            down_sample_ratio=self.config['down_sample_ratio'],
            exp_name=exp_name,
            selected_indexes=self.selected_indexes,
            detection_box=self.config['detection_box'],
            action_display_mode=self.config['action_display_mode'],
            visual_prompt=self.config['visual_prompt'],
            resolution=self.config['resolution'],
            enable_eqa=self.config['enable_eqa'],
            nav_mode=self.config['nav_mode'],
            object_reference_mode=self.config['object_reference_mode'],
            bbox_iou_threshold=self.config['bbox_iou_threshold'],
            point_dist_ratio=self.config['point_dist_ratio'],
            instruction_mode=self.config['instruction_mode'],
            x_display=self.display,
        )

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
        _free_display(self.display)


SESSIONS: Dict[str, Session] = {}
_SESS_LOCK = threading.Lock()


def _get_session(session_id: str) -> Session:
    with _SESS_LOCK:
        s = SESSIONS.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found or closed")
    s.touch()
    return s


def _reaper_loop():
    """Close sessions that have been idle beyond the configured timeout."""
    while True:
        time.sleep(60)
        timeout = GLOBAL_CONFIG.get('session_idle_timeout', 1800)
        now = time.time()
        with _SESS_LOCK:
            stale = [sid for sid, s in SESSIONS.items()
                     if now - s.last_active > timeout]
            for sid in stale:
                s = SESSIONS.pop(sid)
                try:
                    s.close()
                except Exception as e:
                    # Reaper must keep running; log and continue reaping.
                    print(f"[reaper] failed to close session {sid}: {e}")


# ===================== FastAPI =====================
@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_reaper_loop, daemon=True).start()
    yield
    # Shutdown: close every live session so THOR controllers and displays
    # are released with the server process.
    with _SESS_LOCK:
        for s in SESSIONS.values():
            s.close()
        SESSIONS.clear()


app = FastAPI(title="PGI-ALFRED Human Evaluation", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global config, loaded at startup
GLOBAL_CONFIG: Dict = {}


# ---------- Request models ----------
class StartReq(BaseModel):
    session_id: str
    evaluator_id: str
    eval_set: Optional[str] = None
    selected_indexes: Optional[List[int]] = None


class SessionReq(BaseModel):
    session_id: str


class StepReq(BaseModel):
    session_id: str
    command: str
    target_point: Optional[List[float]] = None   # normalized [0, 1] [cx, cy]
    target_bbox: Optional[List[float]] = None    # normalized [0, 1] [x1, y1, x2, y2]


class EqaAnswerReq(BaseModel):
    session_id: str
    selected: List[str]   # selected option letters


@app.get("/config")
def get_config():
    cfg = {k: GLOBAL_CONFIG.get(k) for k in (
        'eval_sets', 'max_steps', 'resolution', 'action_display_mode',
        'object_reference_mode', 'nav_mode', 'enable_eqa',
        'eqa_end_episode_questions', 'exp_name')}
    return cfg


@app.post("/start")
def start_session(req: StartReq):
    # evaluator_id becomes part of the filesystem path; keep it safe.
    evaluator_id = re.sub(r'[^A-Za-z0-9._-]', '_', req.evaluator_id.strip())
    if not evaluator_id:
        raise HTTPException(status_code=400, detail="evaluator_id is required")
    eval_set = req.eval_set or (GLOBAL_CONFIG.get('eval_sets') or ['base'])[0]
    # Replace any live session bound to the same id (otherwise the old THOR
    # controller and X display would leak).
    with _SESS_LOCK:
        old = SESSIONS.pop(req.session_id, None)
    if old is not None:
        old.close()
    display = _alloc_display()
    try:
        s = Session(
            config=GLOBAL_CONFIG,
            eval_set=eval_set,
            selected_indexes=req.selected_indexes,
            evaluator_id=evaluator_id,
            display=display,
        )
        s.start()
        with _SESS_LOCK:
            SESSIONS[req.session_id] = s
    except Exception as e:
        _free_display(display)
        raise HTTPException(status_code=500, detail=str(e))
    return {"session_id": req.session_id, "eval_set": eval_set,
            "display": display, "num_episodes": s.env.number_of_episodes}


@app.post("/reset")
def reset(req: SessionReq):
    s = _get_session(req.session_id)
    s.env.reset()
    s.step_history = []
    s.episode_info = {
        'reward': [],
        'num_invalid_actions': 0,
        'empty_plan': 0,
        'instruction': s.env.episode_language_instruction,
        'task_success': 0.0,
        'task_progress': 0.0,
        'num_steps': 0,
        'num_subtasks': 0,
        'total_goal_conditions': 0,
        'subtask_completion': 0.0,
        'subtask_details': [],
        'action_selection_times': [],
    }
    s.episode_info['episode_idx'] = _episode_idx(s.env)
    s._action_select_time = time.time()
    # Record the initial state
    s.step_history.append({
        'step': 0,
        'action': '(initial state)',
        'frame': _frame_to_b64(s.env),
        'feedback': 'Environment reset',
        'success': True,
    })
    return {
        'instruction': s.env.episode_language_instruction,
        'frame': _frame_to_b64(s.env),
        'state': s.env.get_state_info(),
        'episode_idx': s.episode_info['episode_idx'],
        'num_episodes': s.env.number_of_episodes,
    }


@app.get("/state")
def state(session_id: str, since: int = 0):
    s = _get_session(session_id)
    return {
        'frame': _frame_to_b64(s.env),
        'state': s.env.get_state_info(),
        'task_success': s.episode_info.get('task_success', 0.0),
        'task_progress': s.episode_info.get('task_progress', 0.0),
        'num_steps': s.episode_info.get('num_steps', 0),
        'max_steps': GLOBAL_CONFIG.get('max_steps', 100),
        # Incremental history: only entries after `since`, so clients no
        # longer re-download the full base64 history on every poll.
        'history_total': len(s.step_history),
        'history': [{'index': i, **h} for i, h in enumerate(s.step_history)
                    if i >= since],
    }


@app.post("/step")
def step(req: StepReq):
    s = _get_session(req.session_id)
    if s.env is None or not s.env._reset:
        raise HTTPException(status_code=400, detail="Call /reset first")
    decision_time = time.time() - s._action_select_time
    try:
        obs, reward, done, info = s.env.step(
            req.command,
            reasoning='',
            target_bbox=req.target_bbox,
            target_point=req.target_point,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Action failed: {e}")

    s.episode_info['reward'].append(reward)
    s.episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
    s.episode_info['task_success'] = info['task_success']
    s.episode_info['task_progress'] = info['task_progress']
    s.episode_info['num_steps'] = info['env_step']
    # Same subtask fields the VLM evaluator records from step info
    s.episode_info['num_subtasks'] = info.get('num_subtasks', 0)
    s.episode_info['total_goal_conditions'] = info.get('total_goal_conditions', 0)
    s.episode_info['subtask_completion'] = info.get('subtask_completion', 0.0)
    s.episode_info['subtask_details'] = info.get('subtask_details', [])
    s.episode_info['action_selection_times'].append(decision_time)

    # Record history (with frame) for later review
    frame_b64 = _frame_to_b64(s.env)
    s.step_history.append({
        'step': info['env_step'],
        'action': req.command,
        'frame': frame_b64,
        'feedback': info['env_feedback'],
        'success': bool(info['last_action_success']),
    })

    s._action_select_time = time.time()
    max_steps = GLOBAL_CONFIG.get('max_steps', 100)
    step_limit_reached = info['env_step'] >= max_steps
    return {
        'frame': frame_b64,
        'reward': reward,
        # Treat the configured step budget as episode end so the client moves
        # on to the EQA phase; the env itself would keep going until its own
        # internal cap otherwise.
        'done': done or step_limit_reached,
        'step_limit_reached': step_limit_reached,
        'info': {
            'task_success': info['task_success'],
            'task_progress': info['task_progress'],
            'last_action_success': info['last_action_success'],
            'env_feedback': info['env_feedback'],
            'env_step': info['env_step'],
        },
        'state': s.env.get_state_info(),
    }


@app.post("/end_task")
def end_task(req: SessionReq):
    """Manually end the task and enter the EQA phase (if enabled)."""
    s = _get_session(req.session_id)
    return _enter_eqa(s)


def _enter_eqa(s: Session):
    """Generate end-of-episode EQA questions."""
    if not GLOBAL_CONFIG.get('enable_eqa', True):
        return {"eqa_enabled": False, "questions": []}
    eqa_count = GLOBAL_CONFIG.get('eqa_end_episode_questions', -15)
    if eqa_count > 0:
        questions = s.env.eqa_generator.generate_batch_questions(
            eqa_count, specific=True)
    elif eqa_count < 0:
        questions = s.env.eqa_generator.generate_batch_questions(
            -eqa_count, specific=False)
    else:
        questions = []
    s.eqa_state = {'questions': questions, 'results': [], 'current_idx': 0}
    return {
        "eqa_enabled": True,
        "num_questions": len(questions),
        "questions": [
            {
                'id': q.get('id', ''),
                'question_text': q['question_text'],
                'options': q['options'],
                'question_type': q.get('metadata', {}).get(
                    'question_type', 'single_choice'),
                'evidence': q.get('evidence', {}),
            } for q in questions
        ],
    }


@app.post("/eqa_answer")
def eqa_answer(req: EqaAnswerReq):
    s = _get_session(req.session_id)
    idx = s.eqa_state['current_idx']
    if idx >= len(s.eqa_state['questions']):
        raise HTTPException(status_code=400, detail="All EQA questions answered")
    q = s.eqa_state['questions'][idx]
    q_type = q.get('metadata', {}).get('question_type', 'single_choice')
    human_answer = ','.join(sorted(req.selected)) if q_type == 'multi_choice' \
        else (req.selected[0] if req.selected else '')
    correct_answer = q.get('correct_answer', '')
    is_correct = _compare_eqa_answer(human_answer, correct_answer, q_type)
    result = {
        'question_id': q.get('id', ''),
        'question_text': q['question_text'],
        'options': q['options'],
        'correct_answer': correct_answer,
        'model_answer': human_answer,
        'reasoning': '',
        'is_correct': is_correct,
        'evidence': q.get('evidence', {}),
        'metadata': q.get('metadata', {}),
    }
    s.eqa_state['results'].append(result)
    s.eqa_state['current_idx'] += 1
    return {
        'is_correct': is_correct,
        'correct_answer': correct_answer,
        'done': s.eqa_state['current_idx'] >= len(s.eqa_state['questions']),
    }


@app.post("/save")
def save(req: SessionReq):
    """Compile and save the episode result (call after EQA is finished)."""
    s = _get_session(req.session_id)
    info = s.episode_info
    info['reward'] = float(np.mean(info['reward'])) if info['reward'] else 0.0
    info['planner_steps'] = info['num_steps']
    info['planner_output_error'] = 0
    info['empty_plan'] = 0
    num_steps = info['num_steps']
    info['num_invalid_action_ratio'] = (
        info['num_invalid_actions'] / num_steps if num_steps > 0 else 0)
    info['episode_elapsed_seconds'] = time.time() - s.env._episode_start_time
    info['episode_log'] = s.env.episode_log
    info['episode_messages'] = []

    # EQA summary
    eqa_results = s.eqa_state.get('results', [])
    if eqa_results:
        total = len(eqa_results)
        correct = sum(1 for r in eqa_results if r['is_correct'])
        info['eqa_results'] = eqa_results
        info['eqa_accuracy'] = correct / total if total > 0 else 0
        info['eqa_total'] = total
        info['eqa_correct'] = correct
    else:
        info['eqa_accuracy'] = -1
        info['eqa_total'] = -1
        info['eqa_correct'] = -1

    info['evaluator_type'] = 'human'
    info['evaluator_id'] = s.evaluator_id
    res_path = _save_episode(s.env, info)
    return {"saved": True, "path": res_path,
            "episode_idx": info.get('episode_idx')}


@app.post("/close")
def close(req: SessionReq):
    with _SESS_LOCK:
        s = SESSIONS.pop(req.session_id, None)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found or closed")
    s.close()
    return {"closed": True}


# ===================== Static frontend =====================
_HERE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")),
          name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))


def main():
    parser = argparse.ArgumentParser(description="PGI-ALFRED human evaluation server")
    parser.add_argument('--config', type=str,
                        default=os.path.join(ROOT, 'pgibench/configs/human_eval.yaml'))
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = load_config(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
