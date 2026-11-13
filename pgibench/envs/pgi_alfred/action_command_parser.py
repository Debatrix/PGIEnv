"""
Free-text action command parser + multi-scheme object reference resolver
for the PGI-ALFRED 'command' action-display mode.

The parsing grammar mirrors ThorConnector.llm_skill_interact() prefixes so that a
resolved `interaction_str` can be fed straight back into the (unchanged) execution
layer. Object disambiguation is handled by exactly one of five reference schemes
selected by `object_reference_mode` (id|bbox|point|ordinal|type).

This module is intentionally pure (no PGIAlfEnv import) to avoid circular imports
and to allow logic testing against synthetic ResolverContext data. PGIAlfEnv builds
a ResolverContext from live simulator state and calls resolve_command().
"""
import math
import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Set

from pgibench.envs.pgi_alfred.utils import (alfred_objs,
                                            alfred_slice_sliced_obj,
                                            natural_word_to_ithor_name,
                                            ithor_name_to_natural_word)

# --- action_type tags (internal to this module) ---
ACTION_GLOBAL = 'global'
ACTION_PICKUP = 'pickup'
ACTION_DROP = 'drop'
ACTION_PUTDOWN = 'putdown'
ACTION_OPEN = 'open'
ACTION_CLOSE = 'close'
ACTION_TOGGLE_ON = 'toggle_on'
ACTION_TOGGLE_OFF = 'toggle_off'
ACTION_SLICE = 'slice'
ACTION_GOTO = 'goto'
ACTION_NAV_OBJ = 'nav_obj'
ACTION_GOTO_COORD = 'goto_coord'
ACTION_DONE = 'done'

# canonical (PascalCase) global movement names accepted by the execution layer
# (ThorConnector.llm_skill_interact). The DISPLAY layer shows these as lower-case +
# space ('look up', 'move ahead', ...); parse_command() normalises any casing/spacing
# variant back to these canonical names so the execution layer is unchanged.
_GLOBAL_CANONICAL = {
    'lookup': 'LookUp',
    'lookdown': 'LookDown',
    'rotateleft': 'RotateLeft',
    'rotateright': 'RotateRight',
    'moveahead': 'MoveAhead',
    'moveback': 'MoveBack',
    'moveleft': 'MoveLeft',
    'moveright': 'MoveRight',
    'savewaypoint': 'SaveWaypoint',
}
GLOBAL_ACTIONS = set(_GLOBAL_CANONICAL.keys())

# ordinal word -> 1-based index
ORDINAL_WORDS = {
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
    'eighth': 8,
    'ninth': 9,
    'tenth': 10,
    '1st': 1,
    '2nd': 2,
    '3rd': 3,
    '4th': 4,
    '5th': 5,
    '6th': 6,
}

# natural-word -> objectType alias table, longest first so 'pot' does not steal 'potato'.
# Includes the Sliced child types (e.g. AppleSliced) so the type/bbox/point/ordinal
# resolvers can also match sliced instances. Each Sliced type contributes two
# aliases: the spaced natural form ('apple sliced') and the unspaced form
# ('applesliced') that matches the id_2_name label used in id mode.
_TYPE_ALIASES = sorted([(ithor_name_to_natural_word(t).lower(), t)
                        for t in (alfred_objs + alfred_slice_sliced_obj)] +
                       [(t.lower(), t) for t in alfred_slice_sliced_obj],
                       key=lambda x: len(x[0]),
                       reverse=True)


@dataclass
class ParsedCommand:
    action_type: str
    raw_target: Optional[str] = None  # object-reference substring (target)
    raw_receptacle: Optional[str] = None  # putdown receptacle reference
    coords: Optional[Tuple[float, float, float]] = None
    waypoint: Optional[str] = None
    target_bbox: Optional[List[float]] = None  # normalized [0,1] [x1,y1,x2,y2]
    target_point: Optional[List[float]] = None  # normalized [0,1] [cx,cy]
    global_name: Optional[
        str] = None  # canonical PascalCase global action passed to the execution layer, e.g. 'LookUp'


@dataclass
class ResolverContext:
    """Snapshot of simulator state needed by the resolvers. Built by PGIAlfEnv."""
    allowed_objs: Set[str]  # obj_ids in the mode's display range
    instance_detections2D: Dict[
        str, List[float]]  # obj_id -> [x1, y1, x2, y2] in PIXELS
    id_2_name: Dict[str, str]  # obj_id -> 'mug_#0'
    name_2_id: Dict[str, str]  # 'mug_#0' -> obj_id (reverse map)
    object_types: Dict[str, str]  # obj_id -> objectType
    object_distances: Dict[str, float]  # obj_id -> agent distance
    image_size: Tuple[int, int] = (500, 500)  # (width, height) in pixels
    iou_threshold: float = 0.3  # bbox match: accept best IoU >= threshold
    point_dist_threshold: float = 212.0  # point match: max center-to-box-center dist (px)
    locdesc_map: Dict[str, str] = field(
        default_factory=dict)  # retained; unused after desc removal


# ============================ parsing ============================


def parse_command(cmd: str) -> Optional[ParsedCommand]:
    """Parse a free-text command string into a ParsedCommand, or None if unrecognised."""
    if cmd is None:
        return None
    s = cmd.strip()
    if not s:
        return None
    low = s.lower()

    # task termination
    if low in ('done', '-1', 'finish', 'finished', 'stop', 'end',
               'task complete'):
        return ParsedCommand(action_type=ACTION_DONE)

    # global movement (allow 'look up' / 'lookup' / 'LookUp')
    glob = low.replace(' ', '')
    if glob in GLOBAL_ACTIONS:
        return ParsedCommand(action_type=ACTION_GLOBAL,
                             global_name=_GLOBAL_CANONICAL[glob])

    # GoTo (x,y,z) coordinate navigation
    m = re.match(r'^goto\s*\(([-0-9eE.,\s]+)\)$', low)
    if m:
        nums = [
            float(x.strip()) for x in m.group(1).split(',') if x.strip() != ''
        ]
        if len(nums) >= 3:
            return ParsedCommand(action_type=ACTION_GOTO_COORD,
                                 coords=(nums[0], nums[1], nums[2]))

    # goto <waypoint>  (no-space prefix => waypoint navigation; object nav uses 'go to')
    if low.startswith('goto '):
        return ParsedCommand(action_type=ACTION_GOTO, waypoint=s[5:].strip())

    # navigate to / go to <ref>  (object navigation)
    for pref in ('navigate to ', 'go to '):
        if low.startswith(pref):
            return ParsedCommand(action_type=ACTION_NAV_OBJ,
                                 raw_target=s[len(pref):].strip())

    # pick up / pickup <ref>
    for pref in ('pick up ', 'pickup ', 'grab ', 'take '):
        if low.startswith(pref):
            return ParsedCommand(action_type=ACTION_PICKUP,
                                 raw_target=s[len(pref):].strip())

    # put down/putdown  [obj] to/in/on <recep>  |  put down <recep_id>
    # The canonical command-mode form is 'put down {object} to {receptacle}'
    # (preposition 'to'); 'in/on/into/onto' are still accepted for backward compat.
    body = None
    if low.startswith('put down '):
        body = s[9:].strip()
    elif low.startswith('putdown '):
        body = s[8:].strip()
    if body is not None:
        m = re.match(r'^(.+?)\s+(?:to|in|on|into|onto)\s+(.+)$', body,
                     re.IGNORECASE)
        if m:
            return ParsedCommand(action_type=ACTION_PUTDOWN,
                                 raw_target=m.group(1).strip(),
                                 raw_receptacle=m.group(2).strip())
        return ParsedCommand(action_type=ACTION_PUTDOWN, raw_receptacle=body)

    if low.startswith('drop'):
        return ParsedCommand(action_type=ACTION_DROP)

    if low.startswith('open '):
        return ParsedCommand(action_type=ACTION_OPEN, raw_target=s[5:].strip())
    if low.startswith('close '):
        return ParsedCommand(action_type=ACTION_CLOSE,
                             raw_target=s[6:].strip())

    for pref in ('turn on ', 'toggle on ', 'switch on '):
        if low.startswith(pref):
            return ParsedCommand(action_type=ACTION_TOGGLE_ON,
                                 raw_target=s[len(pref):].strip())
    for pref in ('turn off ', 'toggle off ', 'switch off '):
        if low.startswith(pref):
            return ParsedCommand(action_type=ACTION_TOGGLE_OFF,
                                 raw_target=s[len(pref):].strip())

    if low.startswith('slice ') or low.startswith('cut '):
        pref = 'slice ' if low.startswith('slice ') else 'cut '
        return ParsedCommand(action_type=ACTION_SLICE,
                             raw_target=s[len(pref):].strip())

    return None


# ============================ helpers ============================


def extract_object_type(ref: Optional[str]) -> Optional[str]:
    """Find the *first-mentioned* alfred object type in ref (the subject).

    Among all matched type words, pick the one with the earliest start position
    (so 'mug on the dining table' -> Mug, not DiningTable). Ties at the same start
    position resolve to the longest name (so 'pot' never steals 'potato').
    """
    if not ref:
        return None
    # Strip the id-mode '<type>_#<n>' suffix so a sliced reference such as
    # 'applesliced_#0' (or 'apple sliced_#0') still resolves to AppleSliced via
    # the type aliases, instead of failing on the trailing _#<n>.
    low = re.sub(r'_#\d+\b', '', ref.lower())
    best = None  # (start_pos, -length, objectType)
    for nat, ot in _TYPE_ALIASES:
        m = re.search(r'\b' + re.escape(nat) + r'\b', low)
        if m:
            key = (m.start(), -len(nat))
            if best is None or key < best[0]:
                best = (key, ot)
    return best[1] if best else None


def _extract_ordinal(text: Optional[str]) -> Optional[Tuple[str, int]]:
    """Return ('num', n) | ('left', 1) | ('right', 1) | None."""
    if not text:
        return None
    low = text.lower()
    if 'leftmost' in low or 'left-most' in low:
        return ('left', 1)
    if 'rightmost' in low or 'right-most' in low:
        return ('right', 1)
    m = re.search(
        r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|'
        r'1st|2nd|3rd|4th|5th|6th)\b', low)
    if m:
        return ('num', ORDINAL_WORDS[m.group(1)])
    return None


def _bbox_center(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _center_dist(p: Tuple[float, float], box: List[float]) -> float:
    cx, cy = _bbox_center(box)
    return (p[0] - cx)**2 + (p[1] - cy)**2


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nearest_center(ctx: ResolverContext, target_point: Tuple[float, float],
                    candidates: List[str]) -> Optional[str]:
    """Return the candidate objectId whose bbox center is closest to target_point."""
    if target_point is None:
        return None
    tx, ty = target_point
    best, best_d = None, float('inf')
    for oid in candidates:
        box = ctx.instance_detections2D.get(oid)
        if not box or len(box) < 4:
            continue
        d = (tx - (box[0] + box[2]) / 2.0)**2 + (ty -
                                                 (box[1] + box[3]) / 2.0)**2
        if d < best_d:
            best_d, best = d, oid
    return best


# ============================ per-scheme resolvers ============================


def _resolve_id(ctx: ResolverContext,
                ref: Optional[str]) -> Tuple[Optional[str], str]:
    if not ref:
        return None, 'empty reference'
    key = ref.lower().strip()
    obj_id = ctx.name_2_id.get(key) or ctx.name_2_id.get(ref)
    if obj_id is None:
        for name, oid in ctx.name_2_id.items():
            if name.lower() == key:
                obj_id = oid
                break
    if obj_id is None:
        return None, f'ID {ref!r} not in object catalog'
    if obj_id not in ctx.allowed_objs:
        return None, f'{obj_id} not in visible range'
    return obj_id, 'id match'


def _candidates_by_type(ctx: ResolverContext,
                        ref: Optional[str]) -> Tuple[Optional[str], List[str]]:
    otype = extract_object_type(ref)
    if otype is None:
        return None, list(
            ctx.allowed_objs)  # bbox/point may work without a type hint
    cands = [o for o in ctx.allowed_objs if ctx.object_types.get(o) == otype]
    return otype, cands


def _resolve_bbox(ctx: ResolverContext, target_bbox: Optional[List[float]],
                  candidates: List[str]) -> Tuple[Optional[str], str]:
    if not target_bbox or len(target_bbox) < 4:
        return None, 'missing target_bbox'
    # target_bbox arrives normalized to [0,1]; rescale to pixel space so it is
    # directly comparable with instance_detections2D (which is in pixels).
    W, H = ctx.image_size
    nx1, ny1, nx2, ny2 = (float(v) for v in target_bbox[:4])
    tb = [nx1 * W, ny1 * H, nx2 * W, ny2 * H]
    best, best_iou = None, 0.0
    for oid in candidates:
        box = ctx.instance_detections2D.get(oid)
        if not box or len(box) < 4:
            continue
        iou = _iou(tb, box[:4])
        if iou > best_iou:
            best_iou, best = iou, oid
    if best is None:
        return None, 'cannot determine which object you referring to'
    if best_iou < ctx.iou_threshold:
        # Threshold gated: a poor IoU is a misreference and is rejected, rather than
        # silently falling back to the nearest instance (which often points at the
        # wrong object).
        return None, 'cannot determine which object you referring to'
    return best, f'iou={best_iou:.2f}'


def _resolve_point(ctx: ResolverContext, target_point: Optional[List[float]],
                   candidates: List[str]) -> Tuple[Optional[str], str]:
    if not target_point or len(target_point) < 2:
        return None, 'missing target_point'
    # target_point arrives normalized to [0,1]; rescale to pixel space so it is
    # directly comparable with instance_detections2D (which is in pixels).
    W, H = ctx.image_size
    px, py = float(target_point[0]) * W, float(target_point[1]) * H
    contained = []
    for oid in candidates:
        box = ctx.instance_detections2D.get(oid)
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        if x1 <= px <= x2 and y1 <= py <= y2:
            contained.append((oid, _center_dist((px, py), box[:4])))
    if contained:
        contained.sort(key=lambda x: x[1])
        return contained[0][0], 'point inside bbox'
    # Otherwise take the nearest-centre bbox, but only if it is close enough; below
    # the threshold the point is treated as missing the target and rejected.
    best, best_d = None, float('inf')
    for oid in candidates:
        box = ctx.instance_detections2D.get(oid)
        if not box or len(box) < 4:
            continue
        d = math.sqrt(_center_dist((px, py), box[:4]))
        if d < best_d:
            best_d, best = d, oid
    if best is None:
        return None, 'cannot determine which object you referring to'
    if best_d >= ctx.point_dist_threshold:
        return None, 'cannot determine which object you referring to'
    return best, f'nearest-centre (dist={best_d:.1f}px)'


def _resolve_ordinal(ctx: ResolverContext, ref: Optional[str],
                     candidates: List[str]) -> Tuple[Optional[str], str]:
    """Ordinal resolver with 2D row x column support.

    Syntax accepted:
      - "row R column C" / "row R col C"   -> pick the instance at (R, C) of the grid
      - "first/second/.../leftmost/rightmost" -> 1D flatten in reading order
      - out-of-range / unparseable           -> fall back to image-centre instance

    Grid construction: cluster candidates into rows by bbox centre y (rows separated
    by more than 0.6 x mean-row-height are distinct); within a row sort by centre x.
    Rows are numbered top-to-bottom, columns left-to-right, both 1-based.
    """
    box_of = {o: ctx.instance_detections2D.get(o) for o in candidates}
    cands = [(o, _bbox_center(b[:4])) for o, b in box_of.items()
             if b and len(b) >= 4]
    if not cands:
        return None, 'no visible bbox to order'

    W, H = ctx.image_size
    image_center = (W / 2.0, H / 2.0)
    mean_h = sum(
        (box_of[o][3] - box_of[o][1]) for o, _ in cands) / float(len(cands))
    row_gap = max(1.0, mean_h * 0.6)

    # cluster into rows by centre y
    by_y = sorted(cands, key=lambda oc: oc[1][1])
    rows: List[List[Tuple[str, Tuple[float, float]]]] = []
    for oid, c in by_y:
        if rows and abs(c[1] - rows[-1][-1][1][1]) <= row_gap:
            rows[-1].append((oid, c))
        else:
            rows.append([(oid, c)])
    rows = [[(o, c) for o, c in sorted(r, key=lambda oc: oc[1][0])]
            for r in rows]

    # 2D addressing
    m_row = re.search(r"\brow\s+(\d+)", ref or "", re.IGNORECASE)
    m_col = re.search(r"\bcol(?:umn)?\s+(\d+)", ref or "", re.IGNORECASE)
    if m_row and m_col:
        r, c = int(m_row.group(1)), int(m_col.group(1))
        if 1 <= r <= len(rows) and 1 <= c <= len(rows[r - 1]):
            return rows[r - 1][c - 1][0], f"row {r} col {c}"
        fb = _nearest_center(ctx, image_center, [o for o, _ in cands])
        return (fb if fb else
                None), f"row/col ({r},{c}) out of range -> centre fallback"

    # 1D ordinal over reading-order flatten
    flat = [o for r in rows for o, _ in r]
    n = len(flat)
    ord_info = _extract_ordinal(ref)
    if ord_info is None:
        fb = _nearest_center(ctx, image_center, [o for o, _ in cands])
        return (fb if fb else None), "no ordinal -> centre fallback"
    kind, val = ord_info
    if kind == "left":
        return flat[0], "leftmost"
    if kind == "right":
        return flat[-1], "rightmost"
    if 1 <= val <= n:
        return flat[val - 1], f"#{val} in reading order"
    fb = _nearest_center(ctx, image_center, [o for o, _ in cands])
    return (fb if fb else
            None), f"ordinal {val} out of range ({n}) -> centre fallback"


def _resolve_type(ctx: ResolverContext,
                  candidates: List[str]) -> Tuple[Optional[str], str]:
    if not candidates:
        return None, 'no visible candidate of that type'
    W, H = ctx.image_size
    image_center = (W / 2.0, H / 2.0)
    best, best_d = None, float('inf')
    for oid in candidates:
        box = ctx.instance_detections2D.get(oid)
        if not box or len(box) < 4:
            continue
        d = _center_dist(image_center, box[:4])
        if d < best_d:
            best_d, best = d, oid
    if best is None:
        return None, 'no bbox for type candidates'
    return best, 'closest to image centre'


def resolve_reference(
        ctx: ResolverContext,
        parsed: ParsedCommand,
        mode: str,
        is_receptacle: bool = False) -> Tuple[Optional[str], str]:
    """Resolve the target (or receptacle) reference to a concrete objectId."""
    ref = parsed.raw_receptacle if is_receptacle else parsed.raw_target
    if mode == 'id':
        return _resolve_id(ctx, ref)

    otype, candidates = _candidates_by_type(ctx, ref)
    if not candidates:
        return None, f'cannot determine which object you referring to'
    if mode == 'bbox':
        return _resolve_bbox(ctx, parsed.target_bbox, candidates)
    if mode == 'point':
        return _resolve_point(ctx, parsed.target_point, candidates)
    if mode == 'ordinal':
        return _resolve_ordinal(ctx, ref, candidates)
    if mode == 'type':
        return _resolve_type(ctx, candidates)
    return None, f'unknown reference mode {mode!r}'


# ============================ top-level resolve ============================


def resolve_command(
        parsed: ParsedCommand, ctx: ResolverContext,
        held_obj_id: Optional[str], object_reference_mode: str
) -> Tuple[Optional[str], Optional[str], str]:
    """Resolve a ParsedCommand to an executable interaction string.

    Returns (interaction_str | None, resolved_target_objid | None, reason).
    interaction_str is None on failure (caller emits invalid feedback);
    resolved_target is the objectId the command referred to (for logging), even on
    some precondition failures.
    """
    at = parsed.action_type

    if at == ACTION_DONE:
        return None, None, 'done'  # handled by planner/evaluator, not env.step
    if at == ACTION_GLOBAL:
        return parsed.global_name, None, 'global'
    if at == ACTION_GOTO_COORD:
        x, y, z = parsed.coords
        return f"GoTo ({x},{y},{z})", None, 'coordinate'
    if at == ACTION_GOTO:
        return f"goto {parsed.waypoint}", None, 'waypoint'
    if at == ACTION_DROP:
        if held_obj_id is None:
            return None, None, 'drop: not holding any object'
        return 'drop', held_obj_id, 'drop'

    if at == ACTION_PUTDOWN:
        recep_id, reason = resolve_reference(ctx,
                                             parsed,
                                             object_reference_mode,
                                             is_receptacle=True)
        if recep_id is None:
            return None, None, f'putdown receptacle: {reason}'
        if held_obj_id is None:
            return None, recep_id, 'putdown: not holding any object'
        if recep_id not in ctx.allowed_objs:
            return None, recep_id, 'putdown: receptacle not in visible range'
        return f"put down {recep_id}", recep_id, 'putdown'

    # target-object actions
    obj_id, reason = resolve_reference(ctx, parsed, object_reference_mode)
    if obj_id is None:
        return None, None, f'{at}: {reason}'

    if at == ACTION_PICKUP:
        if held_obj_id is not None:
            return None, obj_id, 'pickup: already holding an object'
        if obj_id not in ctx.allowed_objs:
            return None, obj_id, 'pickup: object not in visible range'
        return f"pick up {obj_id}", obj_id, 'pickup'
    if at == ACTION_OPEN:
        if obj_id not in ctx.allowed_objs:
            return None, obj_id, 'open: not in visible range'
        return f"open {obj_id}", obj_id, 'open'
    if at == ACTION_CLOSE:
        return f"close {obj_id}", obj_id, 'close'
    if at == ACTION_TOGGLE_ON:
        return f"turn on {obj_id}", obj_id, 'toggle_on'
    if at == ACTION_TOGGLE_OFF:
        return f"turn off {obj_id}", obj_id, 'toggle_off'
    if at == ACTION_SLICE:
        if held_obj_id is None or 'knife' not in held_obj_id.lower():
            return None, obj_id, 'slice: must hold a knife'
        return f"slice {obj_id}", obj_id, 'slice'
    if at == ACTION_NAV_OBJ:
        if obj_id not in ctx.allowed_objs:
            return None, obj_id, 'navigate: not in visible range'
        return f"navigate to {obj_id}", obj_id, 'nav_obj'

    return None, None, f'unsupported action type {at!r}'
