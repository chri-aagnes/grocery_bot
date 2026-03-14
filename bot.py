"""
NM i AI 2026 — Grocery Bot
Usage:
    python bot.py --token YOUR_TOKEN

Get a token by clicking "Play" on a map at https://app.ainm.no/challenge
"""

import asyncio
import json
import argparse
from collections import deque, Counter
from itertools import permutations as iperms

WS_URL = "wss://game.ainm.no/ws?token={token}"


# ---------------------------------------------------------------------------
# Pathfinding
# ---------------------------------------------------------------------------

def bfs(start, goal, walls, width, height, blocked=None):
    """Return the first action to take toward goal, or 'wait' if unreachable."""
    if start == goal:
        return "wait"
    wall_set = set(map(tuple, walls))
    queue = deque([(start[0], start[1], [])])
    visited = {start}
    dirs = [("move_up",0,-1), ("move_down",0,1), ("move_left",-1,0), ("move_right",1,0)]
    while queue:
        x, y, path = queue.popleft()
        for action, dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if (nx,ny) in visited or nx<0 or ny<0 or nx>=width or ny>=height:
                continue
            if (nx,ny) in wall_set:
                continue
            new_path = path + [action]
            if (nx,ny) == goal:
                return new_path[0]
            visited.add((nx,ny))
            queue.append((nx, ny, new_path))
    return "wait"


def bfs_all_dists(start, wall_set, width, height):
    """BFS from start; returns {(x,y): distance} for all reachable cells."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = x+dx, y+dy
            if (nx,ny) in dist or nx<0 or ny<0 or nx>=width or ny>=height:
                continue
            if (nx,ny) in wall_set:
                continue
            dist[(nx,ny)] = dist[(x,y)] + 1
            queue.append((nx,ny))
    return dist


def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def is_adjacent(pos, target):
    return manhattan(pos, target) == 1


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def get_needed_items(state):
    """Item types still required by the active order (minus already delivered)."""
    active = next((o for o in state["orders"] if o["status"] == "active"), None)
    if not active:
        return []
    needed = list(active["items_required"])
    for d in active["items_delivered"]:
        if d in needed:
            needed.remove(d)
    return needed


def get_preview_items(state):
    """Item types in the upcoming preview order."""
    preview = next((o for o in state["orders"] if o["status"] == "preview"), None)
    return list(preview["items_required"]) if preview else []


def nearest_drop_off(pos, state):
    zones = state.get("drop_off_zones", [state["drop_off"]])
    return min(zones, key=lambda z: manhattan(pos, z))


def pickup_cells(item_pos, wall_set, width, height):
    """Walkable floor cells adjacent to a shelf item."""
    ix, iy = item_pos
    return [
        (ix+dx, iy+dy)
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]
        if 0 <= ix+dx < width and 0 <= iy+dy < height
        and (ix+dx, iy+dy) not in wall_set
    ]


# ---------------------------------------------------------------------------
# Trip planner
# ---------------------------------------------------------------------------

def plan_trip(pos, inventory, needed, preview_types,
              items_on_map, assigned_items,
              eff_wall_set, width, height, drop_off):
    """
    Plan the optimal trip for one bot:
      - Select up to (3 - len(inventory)) items to collect
      - Find the permutation that minimises total travel distance
      - Opportunistically insert preview items en route if detour <= 4 moves

    Returns list of (item_id, pickup_cell) in execution order.
    """
    INV_CAP = 3
    capacity_left = INV_CAP - len(inventory)
    remaining_needed = list((Counter(needed) - Counter(inventory)).elements())
    remaining_counter = Counter(remaining_needed)

    if not remaining_counter or capacity_left == 0:
        return []

    # Collect candidates with valid pickup cells
    item_cells = {}
    candidates = []
    for item in items_on_map:
        if item["type"] not in remaining_counter or item["id"] in assigned_items:
            continue
        cells = pickup_cells(tuple(item["position"]), eff_wall_set, width, height)
        if cells:
            item_cells[item["id"]] = cells
            candidates.append(item)

    if not candidates:
        return []

    # Prune to 3 closest items per type to keep search tractable
    by_type = {}
    for item in candidates:
        by_type.setdefault(item["type"], []).append(item)
    pruned = []
    for items in by_type.values():
        items.sort(key=lambda i: manhattan(pos, tuple(i["position"])))
        pruned.extend(items[:3])

    k = min(capacity_left, len(remaining_needed))

    # Pre-compute BFS distance maps from pos and all candidate pickup cells
    drop_t = tuple(drop_off)
    all_source_cells = {c for item in pruned for c in item_cells[item["id"]]}
    all_source_cells.add(drop_t)
    dist_from = {pos: bfs_all_dists(pos, eff_wall_set, width, height)}
    for cell in all_source_cells:
        if cell not in dist_from:
            dist_from[cell] = bfs_all_dists(cell, eff_wall_set, width, height)

    def route_cost_and_cells(ordered_items):
        """Total distance: pos → item0.cell → item1.cell → … → drop_off."""
        cost = 0
        cur = pos
        used_cells = []
        for item in ordered_items:
            best_d, best_c = float('inf'), None
            for c in item_cells[item["id"]]:
                d = dist_from[cur].get(c, float('inf'))
                if d < best_d:
                    best_d, best_c = d, c
            if best_c is None:
                return float('inf'), []
            cost += best_d
            cur = best_c
            used_cells.append(best_c)
        cost += dist_from[cur].get(drop_t, float('inf'))
        return cost, used_cells

    # Permutation search: find the ordering of k items with minimum route cost
    best_cost, best_items, best_cells = float('inf'), None, None
    for perm in iperms(pruned, k):
        tc = Counter(i["type"] for i in perm)
        if any(tc[t] > remaining_counter[t] for t in tc):
            continue
        cost, cells = route_cost_and_cells(perm)
        if cost < best_cost:
            best_cost, best_items, best_cells = cost, list(perm), cells

    if best_items is None:
        return []

    plan = list(zip([i["id"] for i in best_items], best_cells))

    # En-route pre-fetch: insert preview items that cost ≤ 4 extra moves
    if len(plan) < capacity_left and preview_types:
        already_planned = {pid for pid, _ in plan}
        preview_counter = Counter(preview_types)
        preview_cands = []
        for item in items_on_map:
            if (item["type"] in preview_counter
                    and item["id"] not in assigned_items
                    and item["id"] not in already_planned):
                cells = pickup_cells(tuple(item["position"]), eff_wall_set, width, height)
                if cells:
                    item_cells[item["id"]] = cells
                    preview_cands.append(item)
        preview_cands.sort(key=lambda i: manhattan(pos, tuple(i["position"])))

        for pitem in preview_cands:
            if len(plan) >= capacity_left:
                break
            for c in item_cells[pitem["id"]]:
                if c not in dist_from:
                    dist_from[c] = bfs_all_dists(c, eff_wall_set, width, height)

            route_nodes = [pos] + [c for _, c in plan] + [drop_t]
            best_detour, best_idx, best_cell = float('inf'), None, None
            for idx in range(len(route_nodes) - 1):
                fn, tn = route_nodes[idx], route_nodes[idx+1]
                orig = dist_from[fn].get(tn, float('inf'))
                for c in item_cells[pitem["id"]]:
                    detour = (dist_from[fn].get(c, float('inf'))
                              + dist_from[c].get(tn, float('inf'))
                              - orig)
                    if detour < best_detour:
                        best_detour, best_idx, best_cell = detour, idx, c
            if best_detour <= 4 and best_idx is not None:
                plan.insert(best_idx, (pitem["id"], best_cell))

    return plan


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide_bot(bot, state, assigned_items):
    x, y = bot["position"]
    pos = (x, y)
    inventory = bot["inventory"]
    bot_id = bot["id"]

    grid = state["grid"]
    width, height = grid["width"], grid["height"]
    walls = grid["walls"]

    # Shelf/item positions are impassable (not in grid["walls"] but server blocks them)
    eff_walls = walls + [item["position"] for item in state["items"]]
    eff_wall_set = set(map(tuple, eff_walls))

    other_positions = {tuple(b["position"]) for b in state["bots"] if b["id"] != bot_id}
    drop_off = nearest_drop_off(pos, state)
    drop_off_t = tuple(drop_off)
    on_drop_off = (list(pos) == drop_off)

    needed = get_needed_items(state)
    preview = get_preview_items(state)
    remaining_needed = list((Counter(needed) - Counter(inventory)).elements())

    # Drop off only if we're carrying items the active order actually needs
    deliverable = [item for item in inventory if item in needed]
    if on_drop_off and deliverable:
        return {"bot": bot_id, "action": "drop_off"}, None

    # Head to drop-off if we have all needed items, or inventory is full and has deliverables
    if deliverable and (not remaining_needed or len(inventory) >= 3):
        action = bfs(pos, drop_off_t, eff_walls, width, height, other_positions)
        return {"bot": bot_id, "action": action}, None

    # Plan the optimal trip
    plan = plan_trip(
        pos, inventory, needed, preview,
        state["items"], assigned_items,
        eff_wall_set, width, height, drop_off,
    )

    if not plan:
        if inventory:
            action = bfs(pos, drop_off_t, eff_walls, width, height, other_positions)
            return {"bot": bot_id, "action": action}, None
        return {"bot": bot_id, "action": "wait"}, None

    target_id, goal_cell = plan[0]
    target_item = next((i for i in state["items"] if i["id"] == target_id), None)
    if target_item is None:
        return {"bot": bot_id, "action": "wait"}, None

    if is_adjacent(pos, tuple(target_item["position"])):
        return {"bot": bot_id, "action": "pick_up", "item_id": target_id}, target_id

    action = bfs(pos, goal_cell, eff_walls, width, height, other_positions)
    return {"bot": bot_id, "action": action}, target_id


def decide_all(state):
    actions = []
    assigned_items = set()
    for bot in sorted(state["bots"], key=lambda b: -len(b["inventory"])):
        action, claimed = decide_bot(bot, state, assigned_items)
        actions.append(action)
        if claimed:
            assigned_items.add(claimed)
    return actions


# ---------------------------------------------------------------------------
# WebSocket runner
# ---------------------------------------------------------------------------

async def play(token):
    url = token if token.startswith("wss://") or token.startswith("ws://") else WS_URL.format(token=token)
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
                print(f"  Needed: {needed} | On map: {items_on_map} | Drop-off: {state['drop_off']} | Score: {score}")

            await ws.send(json.dumps({"actions": actions}))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NM i AI 2026 Grocery Bot")
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    asyncio.run(play(args.token))
