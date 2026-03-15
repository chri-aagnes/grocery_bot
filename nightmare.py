"""
NM i AI 2026 — Grocery Bot (experimental v2: cross-order planning)
Usage:
    python bot_experiment.py --token YOUR_TOKEN

Changes from v1:
  - Unified cross-order planning: preview items fill spare capacity in the
    same permutation search as active items (no more detour-≤-4 heuristic)
  - Two-trip optimization: for orders needing >3 items, picks the subset
    that minimises total cost across BOTH trips
  - Grab-on-the-way: when all active items are collected, grabs preview
    items if the detour is ≤ ~10 rounds (saves a full trip later)
  - Smarter drop-off zone selection across all plans
"""

import asyncio
import json
import argparse
from collections import deque, Counter
from itertools import permutations as iperms

WS_URL = "wss://game.ainm.no/ws?token={token}"

# Max detour (rounds) to grab preview items when active order is already fulfilled
MAX_PREVIEW_DETOUR = 10


# ---------------------------------------------------------------------------
# Pathfinding
# ---------------------------------------------------------------------------

def bfs_first_action(start, goal, wall_set, width, height, soft_blocked=None):
    """Return the first move action toward goal, or 'wait'.
    If soft_blocked is given, tries to avoid those cells first; falls back
    to ignoring them if no path exists (e.g. unavoidable narrow corridor)."""
    if start == goal:
        return "wait"

    def _bfs(wset):
        queue = deque([(start[0], start[1], None)])
        visited = {start}
        dirs = [("move_up",0,-1),("move_down",0,1),("move_left",-1,0),("move_right",1,0)]
        while queue:
            x, y, first = queue.popleft()
            for action, dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited or nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if (nx, ny) in wset:
                    continue
                fa = first or action
                if (nx, ny) == goal:
                    return fa
                visited.add((nx, ny))
                queue.append((nx, ny, fa))
        return "wait"

    if soft_blocked:
        result = _bfs(wall_set | soft_blocked)
        if result != "wait":
            return result
        # Fallback: allow path through soft_blocked (unavoidable narrow corridor)
        return _bfs(wall_set)
    return _bfs(wall_set)


def bfs_dists(start, wall_set, width, height):
    """BFS flood-fill from start. Returns {(x,y): distance}."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        d = dist[(x, y)]
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in dist or nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if (nx, ny) in wall_set:
                continue
            dist[(nx, ny)] = d + 1
            queue.append((nx, ny))
    return dist


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def get_order_info(state):
    """Returns (active_remaining, preview_needed) lists of item types."""
    active = next((o for o in state["orders"] if o["status"] == "active"), None)
    preview = next((o for o in state["orders"] if o["status"] == "preview"), None)

    active_remaining = []
    if active:
        active_remaining = list(active["items_required"])
        for d in active["items_delivered"]:
            if d in active_remaining:
                active_remaining.remove(d)

    preview_needed = list(preview["items_required"]) if preview else []
    return active_remaining, preview_needed


def pickup_cells(item_pos, wall_set, width, height):
    """Walkable cells adjacent to a shelf item."""
    ix, iy = item_pos
    return [
        (ix + dx, iy + dy)
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]
        if 0 <= ix + dx < width and 0 <= iy + dy < height
        and (ix + dx, iy + dy) not in wall_set
    ]


def preview_still_needed(inventory, active_needed, preview_needed):
    """
    After the bot delivers at drop-off, active-matching items leave inventory.
    Remaining items might already satisfy parts of the preview order.
    Returns a Counter of preview types we still need to fetch.
    """
    # Simulate delivery: remove inventory items that match active order
    post_delivery = list(inventory)
    temp_active = list(active_needed)
    for item in list(post_delivery):
        if item in temp_active:
            post_delivery.remove(item)
            temp_active.remove(item)

    # What preview items do we already have after delivery?
    have = Counter(post_delivery)
    need = Counter(preview_needed)
    for t in list(need):
        need[t] = max(0, need[t] - have.get(t, 0))
    return +need  # drop zero counts


# ---------------------------------------------------------------------------
# Trip planner v2
# ---------------------------------------------------------------------------

def plan_trip(pos, inventory, active_needed, preview_needed,
              items_on_map, assigned_items,
              wall_set, width, height, drop_zones,
              assigned_type_counts=None):
    """
    Plan optimal item collection trip.

    Returns (plan, drop_zone, cost) where:
      plan = [(item_id, pickup_cell), ...] in execution order
      drop_zone = (x, y) best drop-off for this plan
      cost = total BFS-distance rounds for the full trip
    """
    INV_CAP = 3
    capacity = INV_CAP - len(inventory)
    if capacity <= 0:
        return [], drop_zones[0], 0

    # Remaining active items (accounting for inventory)
    remaining = list((Counter(active_needed) - Counter(inventory)).elements())
    remaining_ctr = Counter(remaining)

    # Subtract items already assigned to other bots to avoid over-collection
    if assigned_type_counts:
        remaining_ctr = +(remaining_ctr - assigned_type_counts)  # drop zeros
        remaining = list(remaining_ctr.elements())

    # Preview items still needed (accounting for what we'll have after delivery)
    preview_ctr = preview_still_needed(inventory, active_needed, preview_needed)

    # Build candidates
    item_cells = {}
    active_cands, preview_cands = [], []
    active_ids, preview_ids = set(), set()

    for item in items_on_map:
        if item["id"] in assigned_items:
            continue
        cells = pickup_cells(tuple(item["position"]), wall_set, width, height)
        if not cells:
            continue
        item_cells[item["id"]] = cells
        if item["type"] in remaining_ctr:
            active_cands.append(item)
            active_ids.add(item["id"])
        elif item["type"] in preview_ctr:
            preview_cands.append(item)
            preview_ids.add(item["id"])

    def prune(cands, n=3):
        by_type = {}
        for it in cands:
            by_type.setdefault(it["type"], []).append(it)
        out = []
        for group in by_type.values():
            group.sort(key=lambda i: manhattan(pos, tuple(i["position"])))
            out.extend(group[:n])
        return out

    active_pruned = prune(active_cands)
    preview_pruned = prune(preview_cands)

    active_k = min(capacity, len(remaining))
    preview_slots = capacity - active_k

    # --- Pre-compute BFS distances ---
    sources = {pos}
    for it in active_pruned + preview_pruned:
        sources.update(item_cells[it["id"]])
    for dz in drop_zones:
        sources.add(dz)

    dist = {}
    for s in sources:
        if s not in dist:
            dist[s] = bfs_dists(s, wall_set, width, height)

    def route_cost(seq, start, end):
        """Cost: start → items (best pickup cell each) → end."""
        cost = 0
        cur = start
        cells = []
        for it in seq:
            best_d, best_c = float('inf'), None
            for c in item_cells[it["id"]]:
                d = dist.get(cur, {}).get(c, float('inf'))
                if d < best_d:
                    best_d, best_c = d, c
            if best_c is None:
                return float('inf'), []
            cost += best_d
            cur = best_c
            cells.append(best_c)
        cost += dist.get(cur, {}).get(end, float('inf'))
        return cost, cells

    # ================================================================
    # CASE 1: Multi-trip  (active order needs > capacity items)
    # Pick the subset that minimises trip1 + estimated trip2.
    # ================================================================
    if len(remaining) > capacity:
        best_total, best_plan, best_dz = float('inf'), None, drop_zones[0]

        for perm in iperms(active_pruned, active_k):
            tc = Counter(i["type"] for i in perm)
            if any(tc[t] > remaining_ctr[t] for t in tc):
                continue

            for dz in drop_zones:
                t1_cost, t1_cells = route_cost(perm, pos, dz)
                if t1_cost >= float('inf'):
                    continue

                # Estimate trip 2: remaining types, round-trip from drop-off
                picked = Counter(i["type"] for i in perm)
                leftover = remaining_ctr - picked

                t2_cost = 0
                for t, cnt in leftover.items():
                    avail = sorted(
                        [i for i in active_pruned if i["type"] == t and i not in perm],
                        key=lambda i: min(
                            (dist.get(dz, {}).get(c, 999) for c in item_cells[i["id"]]),
                            default=999,
                        ),
                    )
                    for it in avail[:cnt]:
                        d = min(
                            (dist.get(dz, {}).get(c, 999) for c in item_cells[it["id"]]),
                            default=999,
                        )
                        t2_cost += d * 2  # round-trip estimate

                total = t1_cost + t2_cost
                if total < best_total:
                    best_total = total
                    best_plan = list(zip([i["id"] for i in perm], t1_cells))
                    best_dz = dz

        return best_plan or [], best_dz, best_total

    # ================================================================
    # CASE 2: Single trip — unified active + preview search
    # ================================================================
    combined = active_pruned + (preview_pruned if preview_slots > 0 else [])
    k = active_k + min(preview_slots, len(preview_pruned))
    k = min(k, len(combined))
    if k == 0:
        return [], drop_zones[0], float('inf')

    best_cost, best_items, best_cells, best_dz = float('inf'), None, None, drop_zones[0]

    for perm in iperms(combined, k):
        # Must have exactly active_k active items
        a_count = sum(1 for i in perm if i["id"] in active_ids)
        if a_count != active_k:
            continue
        p_count = len(perm) - a_count
        if p_count > preview_slots:
            continue

        # Validate type constraints
        atc = Counter(i["type"] for i in perm if i["id"] in active_ids)
        if any(atc[t] > remaining_ctr[t] for t in atc):
            continue
        ptc = Counter(i["type"] for i in perm if i["id"] in preview_ids)
        if any(ptc[t] > preview_ctr[t] for t in ptc):
            continue

        for dz in drop_zones:
            cost, cells = route_cost(perm, pos, dz)
            if cost < best_cost:
                best_cost, best_items, best_cells, best_dz = cost, list(perm), cells, dz

    if best_items:
        return list(zip([i["id"] for i in best_items], best_cells)), best_dz, best_cost
    return [], drop_zones[0], float('inf')


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def decide_bot(bot, state, assigned_items, assigned_type_counts, wall_set, width, height, soft_blocked=None, allow_preview=True, preview_override=None):
    pos = tuple(bot["position"])
    inventory = bot["inventory"]
    bot_id = bot["id"]

    active_needed, preview_needed = get_order_info(state)
    # Use caller-supplied remaining preview demand if provided (avoids over-fetching
    # by accounting for what other bots already carry and what's been assigned).
    if preview_override is not None:
        preview_needed = preview_override
    remaining = list((Counter(active_needed) - Counter(inventory)).elements())
    deliverable = [i for i in inventory if i in active_needed]

    drop_zones = [tuple(z) for z in state.get("drop_off_zones", [state["drop_off"]])]
    nearest_dz = min(drop_zones, key=lambda z: manhattan(pos, z))
    on_drop_off = pos in set(drop_zones)

    # --- 1. On drop-off with deliverables → deliver ---
    if on_drop_off and deliverable:
        return {"bot": bot_id, "action": "drop_off"}, None

    # --- 2. Inventory full with deliverables → head to drop-off ---
    if len(inventory) >= 3 and deliverable:
        action = bfs_first_action(pos, nearest_dz, wall_set, width, height, soft_blocked)
        return {"bot": bot_id, "action": action}, None

    # --- 3. All active items collected, spare capacity → grab preview items? ---
    # Only on single-bot maps: with multiple bots another bot can complete the
    # active order while we're collecting preview items, leaving us with a full
    # inventory of non-deliverable items that we can never drop off.
    if deliverable and not remaining and len(inventory) < 3 and len(state["bots"]) == 1:
        plan, dz, plan_cost = plan_trip(
            pos, inventory, [], preview_needed,
            state["items"], assigned_items,
            wall_set, width, height, drop_zones,
        )
        if plan:
            # Is the detour worth it?
            direct = dist_to_nearest_dz(pos, drop_zones, wall_set, width, height)
            detour = plan_cost - direct
            if detour <= MAX_PREVIEW_DETOUR:
                return execute_first_step(bot_id, plan, state, pos, dz, wall_set, width, height, soft_blocked)

        # No worthwhile preview items → go deliver
        action = bfs_first_action(pos, nearest_dz, wall_set, width, height, soft_blocked)
        return {"bot": bot_id, "action": action}, None

    # --- 4. Still need items → plan cross-order trip ---
    # Disable preview pre-fetching on multi-bot maps: other bots can complete
    # the active order before we deliver, leaving us with stale non-deliverable
    # items that can never be dropped off.
    preview_for_plan = preview_needed if len(state["bots"]) == 1 else []
    plan, dz, cost = plan_trip(
        pos, inventory, active_needed, preview_for_plan,
        state["items"], assigned_items,
        wall_set, width, height, drop_zones,
        assigned_type_counts,
    )

    if not plan:
        if inventory:
            if on_drop_off:
                # Step aside — bot is blocking the drop-off for others
                avoid = wall_set | (soft_blocked or set())
                move_dir = {(0,1):"move_down",(1,0):"move_right",(0,-1):"move_up",(-1,0):"move_left"}
                for delta, act in move_dir.items():
                    nx, ny = pos[0]+delta[0], pos[1]+delta[1]
                    if (nx,ny) not in avoid and 0 <= nx < width and 0 <= ny < height:
                        return {"bot": bot_id, "action": act}, None
                # All preferred cells blocked — try ignoring soft_blocked
                for delta, act in move_dir.items():
                    nx, ny = pos[0]+delta[0], pos[1]+delta[1]
                    if (nx,ny) not in wall_set and 0 <= nx < width and 0 <= ny < height:
                        return {"bot": bot_id, "action": act}, None
            if deliverable:
                # Has items to deliver — head to drop-off
                action = bfs_first_action(pos, nearest_dz, wall_set, width, height, soft_blocked)
                return {"bot": bot_id, "action": action}, None
            # Stale inventory (items not needed by active order).
            # If there's spare capacity, try pre-fetching preview items.
            # Safe: preview items become deliverable when the order advances.
            if preview_needed and len(inventory) < 3 and allow_preview:
                preview_plan, preview_dz, _ = plan_trip(
                    pos, inventory, [], preview_needed,
                    state["items"], assigned_items,
                    wall_set, width, height, drop_zones,
                )
                if preview_plan:
                    return execute_first_step(bot_id, preview_plan, state, pos, preview_dz,
                                              wall_set, width, height, soft_blocked)
            return {"bot": bot_id, "action": "wait"}, None
        else:
            # No inventory, no active items to collect → pre-fetch preview if possible.
            # Safe: bot carries nothing stale, and preview items become deliverable
            # when the current active order completes.
            if preview_needed and allow_preview:
                preview_plan, preview_dz, _ = plan_trip(
                    pos, [], [], preview_needed,
                    state["items"], assigned_items,
                    wall_set, width, height, drop_zones,
                )
                if preview_plan:
                    return execute_first_step(bot_id, preview_plan, state, pos, preview_dz,
                                              wall_set, width, height, soft_blocked)
            # Nothing to do — move away from drop-off to clear delivery paths.
            # Idle bots parked near the drop-off block delivering bots for many rounds.
            nearest_dz = min(drop_zones, key=lambda z: manhattan(pos, z))
            dz_dist = manhattan(pos, nearest_dz)
            avoid = wall_set | (soft_blocked or set())
            move_dirs = [(0,1,"move_down"),(1,0,"move_right"),(0,-1,"move_up"),(-1,0,"move_left")]
            if dz_dist <= 8:
                # Move to cell that maximises distance from nearest drop-off zone
                best_act, best_dist = None, dz_dist
                for dx, dy, act in move_dirs:
                    nx, ny = pos[0]+dx, pos[1]+dy
                    if (nx,ny) not in avoid and 0 <= nx < width and 0 <= ny < height:
                        d = manhattan((nx, ny), nearest_dz)
                        if d > best_dist:
                            best_act, best_dist = act, d
                if best_act:
                    return {"bot": bot_id, "action": best_act}, None
                # Soft-blocked version failed — try ignoring soft_blocked
                for dx, dy, act in move_dirs:
                    nx, ny = pos[0]+dx, pos[1]+dy
                    if (nx,ny) not in wall_set and 0 <= nx < width and 0 <= ny < height:
                        d = manhattan((nx, ny), nearest_dz)
                        if d > dz_dist:
                            return {"bot": bot_id, "action": act}, None
            return {"bot": bot_id, "action": "wait"}, None

    return execute_first_step(bot_id, plan, state, pos, dz, wall_set, width, height, soft_blocked)


def execute_first_step(bot_id, plan, state, pos, dz, wall_set, width, height, soft_blocked=None):
    """Execute the first step of a plan: pick up if adjacent, else move."""
    target_id, goal_cell = plan[0]
    target_item = next((i for i in state["items"] if i["id"] == target_id), None)
    if target_item is None:
        return {"bot": bot_id, "action": "wait"}, None

    if manhattan(pos, tuple(target_item["position"])) == 1:
        return {"bot": bot_id, "action": "pick_up", "item_id": target_id}, target_id

    action = bfs_first_action(pos, goal_cell, wall_set, width, height, soft_blocked)
    return {"bot": bot_id, "action": action}, target_id


def dist_to_nearest_dz(pos, drop_zones, wall_set, width, height):
    """BFS distance from pos to nearest drop-off zone."""
    d = bfs_dists(pos, wall_set, width, height)
    return min(d.get(dz, float('inf')) for dz in drop_zones)


# ---------------------------------------------------------------------------
# Top-level per-round decision
# ---------------------------------------------------------------------------

def _next_pos(pos, action):
    """Cell a bot will occupy after taking an action."""
    d = {"move_up": (0,-1), "move_down": (0,1), "move_left": (-1,0), "move_right": (1,0)}
    if action in d:
        return (pos[0] + d[action][0], pos[1] + d[action][1])
    return pos


def decide_all(state):
    width = state["grid"]["width"]
    height = state["grid"]["height"]
    walls = state["grid"]["walls"]

    # Effective walls: grid walls + item shelf positions
    item_positions = [item["position"] for item in state["items"]]
    wall_set = set(map(tuple, walls + item_positions))

    all_bot_positions = {tuple(b["position"]) for b in state["bots"]}
    item_type_by_id = {item["id"]: item["type"] for item in state["items"]}

    # Pre-compute deliverable inventory per bot (items matching active order)
    active_needed, preview_needed_list = get_order_info(state)
    active_needed_ctr = Counter(active_needed)
    preview_needed_ctr = Counter(preview_needed_list)
    bot_deliverable = {
        bot["id"]: Counter(item for item in bot["inventory"] if item in active_needed_ctr)
        for bot in state["bots"]
    }
    all_deliverable = sum(bot_deliverable.values(), Counter())

    # Track preview items already held across ALL bots to avoid over-fetching.
    # Without this, every idle bot independently decides to grab flour/cheese/apples
    # even when multiple bots already carry those types.
    all_held_preview = Counter(
        item for bot in state["bots"]
        for item in bot["inventory"]
        if item in preview_needed_ctr
    )

    actions = []
    assigned_items = set()
    assigned_type_counts = Counter()  # types claimed for pickup this round
    assigned_preview_counts = Counter()  # preview types claimed this round
    reserved_next = set()  # cells already claimed by higher-priority bots this round
    drop_zone_set = set(tuple(z) for z in state.get("drop_off_zones", [state["drop_off"]]))

    for bot in sorted(state["bots"], key=lambda b: -len(b["inventory"])):
        pos = tuple(bot["position"])
        soft_blocked = (all_bot_positions - {pos}) | reserved_next

        # Other bots' deliverable inventory counts (capped by active order needs)
        other_deliverable = all_deliverable - bot_deliverable[bot["id"]]
        other_deliverable = +(other_deliverable & active_needed_ctr)
        effective_assigned = assigned_type_counts + other_deliverable

        # Compute remaining preview demand for this bot:
        # subtract what other bots already carry + what's been assigned this round.
        this_bot_preview = Counter(item for item in bot["inventory"] if item in preview_needed_ctr)
        other_held_preview = all_held_preview - this_bot_preview
        covered_preview = other_held_preview + assigned_preview_counts
        remaining_preview = list((+(preview_needed_ctr - covered_preview)).elements())

        action, claimed = decide_bot(bot, state, assigned_items, effective_assigned,
                                     wall_set, width, height, soft_blocked,
                                     allow_preview=bool(remaining_preview),
                                     preview_override=remaining_preview)

        # Prevent swap deadlock only when bots move in exactly opposite directions
        # (e.g. one going up, the other going down). Same-direction passing is fine.
        act_name = action["action"]
        next_c = _next_pos(pos, act_name)
        opposite = {"move_up":"move_down","move_down":"move_up",
                    "move_left":"move_right","move_right":"move_left"}
        # Don't apply anti-swap when stepping aside at a drop-off zone —
        # yielding to a delivering bot is cooperative, not a deadlock.
        if (act_name.startswith("move_") and next_c in all_bot_positions
                and pos in reserved_next and pos not in drop_zone_set):
            bot_at_next = next((b for b in state["bots"] if tuple(b["position"]) == next_c), None)
            if bot_at_next:
                their_act = next((a["action"] for a in actions if a["bot"] == bot_at_next["id"]), None)
                if their_act == opposite.get(act_name):
                    action = {"bot": bot["id"], "action": "wait"}
                    next_c = pos
                    claimed = None

        actions.append(action)
        if claimed:
            assigned_items.add(claimed)
            if claimed in item_type_by_id:
                t = item_type_by_id[claimed]
                assigned_type_counts[t] += 1
                if t not in active_needed_ctr:
                    assigned_preview_counts[t] += 1
        reserved_next.add(next_c)
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

            if round_num < 5 or round_num % 50 == 0:
                active_needed, preview_needed = get_order_info(state)
                for bot in state["bots"]:
                    a = next((a for a in actions if a["bot"] == bot["id"]), "?")
                    print(f"  R{round_num:3d} | Bot {bot['id']} @ {bot['position']} "
                          f"inv={bot['inventory']} | {a.get('action','?')}")
                print(f"       Active: {active_needed} | Preview: {preview_needed} | Score: {score}")

            await ws.send(json.dumps({"actions": actions}))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NM i AI 2026 Grocery Bot (v2)")
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    asyncio.run(play(args.token))
