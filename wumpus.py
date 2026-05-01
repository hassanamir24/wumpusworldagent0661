import random

# ─── HELPERS ────────────────────────────────────────────────────────────────

def key(r, c):
    """Convert row,col to a string key like '2,3'"""
    return f"{r},{c}"

def neighbors(r, c, rows, cols):
    """Get all valid up/down/left/right neighbors of cell (r,c)"""
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append((nr, nc))
    return result

# ─── WORLD SETUP ────────────────────────────────────────────────────────────

def create_world(rows, cols, num_pits):
    """Create a fresh game world with random hazards and gold"""
    hazards = {}  # key -> 'pit' or 'wumpus'

    # Place pits (not on start cell 0,0)
    placed = 0
    while placed < num_pits:
        r, c = random.randint(0, rows-1), random.randint(0, cols-1)
        if (r, c) == (0, 0) or key(r, c) in hazards:
            continue
        hazards[key(r, c)] = 'pit'
        placed += 1

    # Place Wumpus (not on start, not on a pit)
    while True:
        r, c = random.randint(0, rows-1), random.randint(0, cols-1)
        if (r, c) != (0, 0) and key(r, c) not in hazards:
            hazards[key(r, c)] = 'wumpus'
            break

    # Place gold (not on start, not on a hazard)
    while True:
        r, c = random.randint(0, rows-1), random.randint(0, cols-1)
        if (r, c) != (0, 0) and key(r, c) not in hazards:
            gold = (r, c)
            break

    world = {
        'rows': rows,
        'cols': cols,
        'hazards': hazards,
        'gold': gold,
        'agent': (0, 0),
        'visited': {key(0,0): True},   # cells agent has been to
        'safe': {key(0,0): True},      # cells proven safe by logic
        'kb': [],                       # Knowledge Base: list of clauses
        'inference_steps': 0,          # total resolution steps performed
        'agent_steps': 0,              # moves the agent made
        'percepts': [],                # percepts at current position
        'log': [],                     # history messages shown to user
        'done': False,
        'outcome': None,               # 'win', 'lose', or 'stuck'
    }

    # Sense and update KB at starting position
    p = get_percepts(world, 0, 0)
    world['percepts'] = p
    tell_kb(world, 0, 0, p)

    return world

# ─── PERCEPTS ───────────────────────────────────────────────────────────────

def get_percepts(world, r, c):
    """
    What does the agent sense at (r,c)?
    - Breeze  → a neighbor has a pit
    - Stench  → a neighbor has the wumpus
    - Glitter → gold is right here
    """
    percepts = []
    for nr, nc in neighbors(r, c, world['rows'], world['cols']):
        h = world['hazards'].get(key(nr, nc))
        if h == 'pit'    and 'Breeze' not in percepts: percepts.append('Breeze')
        if h == 'wumpus' and 'Stench' not in percepts: percepts.append('Stench')
    if (r, c) == world['gold']:
        percepts.append('Glitter')
    return percepts

# ─── PROPOSITIONAL LOGIC / KNOWLEDGE BASE ───────────────────────────────────
#
# A CLAUSE is a list of LITERALS.
# A LITERAL is a dict: { 'neg': bool, 'type': 'P'|'W', 'r': int, 'c': int }
#   neg=True  means NOT   (e.g. "no pit here")
#   neg=False means positive  (e.g. "pit here")
#   type 'P' = Pit, type 'W' = Wumpus
#
# Example clause: [{'neg':True, 'type':'P', 'r':1, 'c':2}]
#   → "There is NO pit at (1,2)"
#
# Example clause: [{'neg':False,'type':'P','r':1,'c':2}, {'neg':False,'type':'P','r':2,'c':1}]
#   → "Pit at (1,2)  OR  pit at (2,1)"  (added when agent senses Breeze)

def lit_eq(a, b):
    """Check if two literals are identical"""
    return a['neg'] == b['neg'] and a['type'] == b['type'] and a['r'] == b['r'] and a['c'] == b['c']

def lit_neg(lit):
    """Return the negation of a literal"""
    return {**lit, 'neg': not lit['neg']}

def clause_eq(a, b):
    """Check if two clauses contain the same literals"""
    if len(a) != len(b):
        return False
    return all(any(lit_eq(la, lb) for lb in b) for la in a)

def add_clause(kb, clause):
    """Add clause to KB only if it is not already there"""
    if not any(clause_eq(c, clause) for c in kb):
        kb.append(clause)

def tell_kb(world, r, c, percepts):
    """
    TELL the KB what we learned at (r,c) based on the percepts.
    This is the TELL step of the KBA cycle.
    """
    nbrs = neighbors(r, c, world['rows'], world['cols'])

    if 'Breeze' not in percepts:
        # No breeze → no pit in ANY neighbor
        for nr, nc in nbrs:
            add_clause(world['kb'], [{'neg': True,  'type': 'P', 'r': nr, 'c': nc}])
    else:
        # Breeze → pit in AT LEAST ONE neighbor (disjunction)
        add_clause(world['kb'], [{'neg': False, 'type': 'P', 'r': nr, 'c': nc} for nr, nc in nbrs])

    if 'Stench' not in percepts:
        # No stench → no wumpus in ANY neighbor
        for nr, nc in nbrs:
            add_clause(world['kb'], [{'neg': True,  'type': 'W', 'r': nr, 'c': nc}])
    else:
        # Stench → wumpus in AT LEAST ONE neighbor
        add_clause(world['kb'], [{'neg': False, 'type': 'W', 'r': nr, 'c': nc} for nr, nc in nbrs])

# ─── RESOLUTION REFUTATION ──────────────────────────────────────────────────
#
# To PROVE that literal `goal` is true:
#   1. Add the NEGATION of `goal` to the KB (assume goal is false)
#   2. Repeatedly resolve pairs of clauses
#   3. If we derive an EMPTY CLAUSE → contradiction → goal must be true ✓
#
# resolve(c1, c2): if c1 has literal L and c2 has ¬L,
#   produce: (c1 without L) ∪ (c2 without ¬L)

def resolve(c1, c2):
    """Try to resolve two clauses. Returns list of new clauses produced."""
    results = []
    for i, lit in enumerate(c1):
        for j, lit2 in enumerate(c2):
            if lit_eq(lit, lit_neg(lit2)):
                # Found complementary pair — combine the rest
                resolvent = c1[:i] + c1[i+1:] + c2[:j] + c2[j+1:]

                # Remove duplicate literals
                deduped = []
                for l in resolvent:
                    if not any(lit_eq(l, x) for x in deduped):
                        deduped.append(l)

                # Skip tautologies (contains both L and ¬L)
                tautology = any(
                    any(la['neg'] != lb['neg'] and la['type'] == lb['type']
                        and la['r'] == lb['r'] and la['c'] == lb['c']
                        for lb in deduped)
                    for la in deduped
                )
                if not tautology:
                    results.append(deduped)
    return results

def refute(kb, goal, max_steps=600):
    """
    Resolution Refutation: try to prove `goal` literal.
    Returns (proved: bool, steps: int)
    """
    neg_goal = lit_neg(goal)
    # Start with all KB clauses + the negated goal as a unit clause
    clauses = [list(c) for c in kb] + [[neg_goal]]

    seen = [list(c) for c in clauses]
    queue = list(seen)
    steps = 0

    while queue and steps < max_steps:
        c1 = queue.pop(0)
        for c2 in list(seen):
            for r in resolve(c1, c2):
                steps += 1
                if len(r) == 0:
                    return True, steps          # Empty clause = contradiction = proved!
                if not any(clause_eq(r, s) for s in seen):
                    seen.append(r)
                    queue.append(r)
            if steps >= max_steps:
                break

    return False, steps

def is_safe(world, r, c):
    """
    ASK the KB: is cell (r,c) safe?
    Proves both: no pit AND no wumpus at (r,c)
    """
    proved_no_pit,    s1 = refute(world['kb'], {'neg': True, 'type': 'P', 'r': r, 'c': c})
    proved_no_wumpus, s2 = refute(world['kb'], {'neg': True, 'type': 'W', 'r': r, 'c': c})
    world['inference_steps'] += s1 + s2
    return proved_no_pit and proved_no_wumpus

# ─── AGENT STEP ─────────────────────────────────────────────────────────────

def agent_step(world):
    """
    Move the agent one step using propositional logic.
    Returns updated world dict.
    """
    if world['done']:
        return world

    r, c = world['agent']

    # 1. Get percepts at current position
    percepts = get_percepts(world, r, c)
    world['percepts'] = percepts

    # 2. Check for gold
    if 'Glitter' in percepts:
        world['done'] = True
        world['outcome'] = 'win'
        world['log'].insert(0, '🏆 Found the gold! Agent wins!')
        return world

    # 3. Check for hazard (shouldn't happen with good logic, safety check)
    hazard = world['hazards'].get(key(r, c))
    if hazard == 'pit':
        world['done'] = True; world['outcome'] = 'lose'
        world['log'].insert(0, '🕳 Fell into a pit!')
        return world
    if hazard == 'wumpus':
        world['done'] = True; world['outcome'] = 'lose'
        world['log'].insert(0, '👹 Eaten by the Wumpus!')
        return world

    # 4. Tell KB about current percepts
    tell_kb(world, r, c, percepts)
    world['log'].insert(0, f"At ({r},{c}) — percepts: {', '.join(percepts) if percepts else 'none'}")

    # 5. Try to move to an adjacent unvisited safe cell
    for nr, nc in neighbors(r, c, world['rows'], world['cols']):
        k = key(nr, nc)
        if world['visited'].get(k):
            continue  # already visited
        if world['safe'].get(k) or is_safe(world, nr, nc):
            world['safe'][k] = True
            world['log'].insert(0, f"✓ Proved ({nr},{nc}) is safe → moving there")
            world['agent'] = (nr, nc)
            world['visited'][k] = True
            world['agent_steps'] += 1
            return world

    # 6. No safe neighbor — BFS to find nearest unvisited safe cell via safe path
    from collections import deque
    queue = deque([((r, c), [])])
    seen_bfs = {key(r, c)}

    while queue:
        (br, bc), path = queue.popleft()
        for nr, nc in neighbors(br, bc, world['rows'], world['cols']):
            k = key(nr, nc)
            if k in seen_bfs:
                continue
            seen_bfs.add(k)
            safe = world['safe'].get(k) or is_safe(world, nr, nc)
            if safe:
                world['safe'][k] = True
                if not world['visited'].get(k):
                    # Move one step along the path toward this cell
                    next_step = path[0] if path else (nr, nc)
                    nk = key(*next_step)
                    world['log'].insert(0, f"↩ Backtracking toward safe cell ({nr},{nc})")
                    world['agent'] = next_step
                    world['visited'][nk] = True
                    world['agent_steps'] += 1
                    return world
                # Cell is visited and safe — continue BFS through it
                queue.append(((nr, nc), path + [(nr, nc)]))

    # 7. Truly stuck
    world['done'] = True
    world['outcome'] = 'stuck'
    world['log'].insert(0, '🤔 Stuck — no safe moves left. Try a new game!')
    return world

# ─── KB DISPLAY ─────────────────────────────────────────────────────────────

def clause_to_string(clause):
    """Convert a clause to a readable string for display"""
    if not clause:
        return '⊥ (contradiction)'
    parts = []
    for lit in clause:
        neg = '¬' if lit['neg'] else ''
        parts.append(f"{neg}{lit['type']}({lit['r']},{lit['c']})")
    return ' ∨ '.join(parts)
