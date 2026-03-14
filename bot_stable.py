"""
NM i AI 2026 — Grocery Bot (stable, scored 95 on easy)
Usage:
    python bot_stable.py --token YOUR_TOKEN

Get a token by clicking "Play" on a map at https://app.ainm.no/challenge
"""

import asyncio
import json
import argparse
from collections import deque, Counter

WS_URL = "wss://game.ainm.no/ws?token={token}"


# ---------------------------------------------------------------------------
# Pathfinding
# ---------------------------------------------------------------------------

def bfs(start, goal, walls, width, height, blocked=None):
    """
    BFS from start to goal. Returns the first step to take as an action string,
    or 'wait' if no path exists.
    """
    if start == goal:
        return "wait"

    wall_set = set(map(tuple, walls))

    sx, sy = start
    gx, gy = goal

    queue = deque()
    queue.append((sx, sy, []))
    visited = {(sx, sy)}

    directions = [
        ("move_up",    0, -1),
        ("move_down",  0,  1),
        ("move_left", -1,  0),
        ("move_right", 1,  0),
    ]

    while queue:
        x, y, path = queue.popleft()
        for action, dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if (nx, ny) in wall_set:
                continue
            new_path = path + [action]
            if (nx, ny) == (gx, gy):
                return new_path[0]
            visited.add((nx, ny))
            queue.append((nx, ny, new_path))

    return "wait"


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_adjacent(pos, target):
    return manhattan(pos, target) == 1


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def get_needed_items(state):
    """Return list of item types still needed for the active order."""
    active = next((o for o in state["orders"] if o["status"] == "active"), None)
    if not active:
        return []
    needed = list(active["items_required"])
    for delivered in active["items_delivered"]:
        if delivered in needed:
            needed.remove(delivered)
    return needed


def get_preview_items(state):
    """Return item types needed for the preview order (pre-fetch opportunity)."""
    preview = next((o for o in state["orders"] if o["status"] == "preview"), None)
    if not preview:
        return []
    return list(preview["items_required"])


def nearest_drop_off(pos, state):
    """Return the closest drop-off zone position."""
    zones = state.get("drop_off_zones", [state["drop_off"]])
    return min(zones, key=lambda z: manhattan(pos, z))


def best_adjacent_cell(item_pos, walls, width, height):
    """
    Items sit on shelf tiles. Return all walkable cells adjacent to item_pos.
    """
    wall_set = set(map(tuple, walls))
    ix, iy = item_pos
    candidates = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = ix + dx, iy + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in wall_set:
            candidates.append((nx, ny))
    return candidates


def decide_bot(bot, state, assigned_items):
    """
    Decide one action for a single bot.

    assigned_items: set of item IDs already claimed by other bots this round.
    Returns (action_dict, claimed_item_id_or_None)
    """
    x, y = bot["position"]
    pos = (x, y)
    inventory = bot["inventory"]
    bot_id = bot["id"]

    grid = state["grid"]
    width, height = grid["width"], grid["height"]
    walls = grid["walls"]

    # Item positions are shelf tiles — not in grid["walls"] but impassable to bots.
    # Items respawn at the same positions, so treat them as permanent obstacles.
    item_walls = [item["position"] for item in state["items"]]
    eff_walls = walls + item_walls

    # Other bots' positions (for soft collision avoidance)
    other_positions = {
        tuple(b["position"]) for b in state["bots"] if b["id"] != bot_id
    }

    drop_off = nearest_drop_off(pos, state)
    on_drop_off = list(pos) == drop_off

    needed = get_needed_items(state)
    preview = get_preview_items(state)

    # --- If on drop-off and carrying items, drop them ---
    if on_drop_off and inventory:
        return {"bot": bot_id, "action": "drop_off"}, None

    # --- If inventory is full, head to drop-off ---
    if len(inventory) >= 3:
        action = bfs(pos, tuple(drop_off), eff_walls, width, height, other_positions)
        return {"bot": bot_id, "action": action}, None

    # --- If carrying all needed items, head to drop-off ---
    remaining_needed = list((Counter(needed) - Counter(inventory)).elements())
    if inventory and not remaining_needed:
        action = bfs(pos, tuple(drop_off), eff_walls, width, height, other_positions)
        return {"bot": bot_id, "action": action}, None

    # --- Try to pick up an adjacent needed item ---
    for item in state["items"]:
        if item["id"] in assigned_items:
            continue
        if item["type"] not in remaining_needed:
            continue
        ix, iy = item["position"]
        if is_adjacent(pos, (ix, iy)):
            return {"bot": bot_id, "action": "pick_up", "item_id": item["id"]}, item["id"]

    # --- Move toward nearest needed item ---
    candidates = [
        item for item in state["items"]
        if item["type"] in remaining_needed and item["id"] not in assigned_items
    ]
    if candidates:
        target = min(candidates, key=lambda i: manhattan(pos, tuple(i["position"])))
        neighbours = best_adjacent_cell(tuple(target["position"]), eff_walls, width, height)
        if neighbours:
            goal = min(neighbours, key=lambda n: manhattan(pos, n))
            action = bfs(pos, goal, eff_walls, width, height, other_positions)
        else:
            action = "wait"
        return {"bot": bot_id, "action": action}, target["id"]

    # --- Pre-fetch for preview order ---
    preview_candidates = [
        item for item in state["items"]
        if item["type"] in preview and item["id"] not in assigned_items
    ]
    if preview_candidates and len(inventory) < 3:
        target = min(preview_candidates, key=lambda i: manhattan(pos, tuple(i["position"])))
        if is_adjacent(pos, tuple(target["position"])):
            return {"bot": bot_id, "action": "pick_up", "item_id": target["id"]}, target["id"]
        neighbours = best_adjacent_cell(tuple(target["position"]), eff_walls, width, height)
        if neighbours:
            goal = min(neighbours, key=lambda n: manhattan(pos, n))
            action = bfs(pos, goal, eff_walls, width, height, other_positions)
        else:
            action = "wait"
        return {"bot": bot_id, "action": action}, target["id"]

    # --- Nothing to do: head to drop-off if holding items, else wait ---
    if inventory:
        action = bfs(pos, tuple(drop_off), eff_walls, width, height, other_positions)
        return {"bot": bot_id, "action": action}, None

    return {"bot": bot_id, "action": "wait"}, None


def decide_all(state):
    """Decide actions for all bots, avoiding duplicate item assignments."""
    actions = []
    assigned_items = set()

    # Sort bots: prioritise those already holding items (closer to completion)
    bots_sorted = sorted(state["bots"], key=lambda b: -len(b["inventory"]))

    for bot in bots_sorted:
        action, claimed = decide_bot(bot, state, assigned_items)
        actions.append(action)
        if claimed:
            assigned_items.add(claimed)

    return actions


# ---------------------------------------------------------------------------
# WebSocket runner
# ---------------------------------------------------------------------------

async def play(token):
    if token.startswith("wss://") or token.startswith("ws://"):
        url = token
    else:
        url = WS_URL.format(token=token)
    print(f"Connecting to {url}")

    import websockets
    async with websockets.connect(url) as ws:
        round_num = 0
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)

            if msg["type"] == "game_over":
                score = msg.get("score", 0)
                items = msg.get("items_delivered", "?")
                orders = msg.get("orders_completed", "?")
                rounds = msg.get("rounds_used", "?")
                print(f"\nGame over! Score: {score} | Items: {items} | Orders: {orders} | Rounds: {rounds}")
                break

            if msg["type"] != "game_state":
                continue

            state = msg
            round_num = state.get("round", round_num)
            score = state.get("score", 0)

            actions = decide_all(state)

            if round_num < 10 or round_num % 20 == 0:
                for bot in state["bots"]:
                    a = next((a for a in actions if a["bot"] == bot["id"]), "?")
                    print(f"  Round {round_num} | Bot {bot['id']} @ {bot['position']} | inv={bot['inventory']} | action={a}")
                needed = get_needed_items(state)
                items_on_map = [(i["type"], i["position"]) for i in state["items"] if i["type"] in needed]
                print(f"  Needed: {needed} | Items on map: {items_on_map} | Drop-off: {state['drop_off']} | Score: {score}")

            await ws.send(json.dumps({"actions": actions}))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NM i AI 2026 Grocery Bot (stable)")
    parser.add_argument("--token", required=True, help="JWT token from app.ainm.no/challenge")
    args = parser.parse_args()

    asyncio.run(play(args.token))
