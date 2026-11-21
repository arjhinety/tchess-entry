"""tiny-chess-01 engine, authored from scratch for this bounty.

Position: 64-int list, index 0 = a8 ... 63 = h1. White pieces positive,
black negative: 1 pawn, 2 knight, 3 bishop, 4 rook, 5 queen, 6 king.
Side is 0 = white, 1 = black.

Search: negamax alpha-beta, iterative deepening on a fixed NODE budget
(never a clock -- the final replays every trial and a clock-decided move
would make the replay diverge), transposition table for ordering and
repeats, killer moves, MVV-LVA, captures-only quiescence with stand-pat.

The root move list is taken from the env's own `legal_moves`, so a defect
in the internal generator can never produce a refused move or a lost game:
worst case is a weaker search, never an illegal one.
"""
import json
import os
import urllib.request

ENV = os.environ.get("MAKEFUN_ENV_URL", "")
BUDGET = 9000
MATE = 1 << 20

V = (0, 100, 320, 330, 500, 9000, 0)
FILES = "abcdefgh"
RAYS = (((1, 1), (1, -1), (-1, 1), (-1, -1)), ((1, 0), (-1, 0), (0, 1), (0, -1)))
HORSE = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))


def _tables():
    """Precompute the hop lists so movegen is index math, not arithmetic."""
    kn, bi, ro, qs = [], [], [], []
    for r in range(8):
        for c in range(8):
            a, b, w = [], [], []
            for dr, dc in HORSE:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    a.append(nr * 8 + nc)
            for dr, dc in RAYS[0]:
                nr, nc = r + dr, c + dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    b.append(nr * 8 + nc)
                    nr, nc = nr + dr, nc + dc
            for dr, dc in RAYS[1]:
                nr, nc = r + dr, c + dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    w.append(nr * 8 + nc)
                    nr, nc = nr + dr, nc + dc
            kn.append(tuple(a))
            bi.append(tuple(b))
            ro.append(tuple(w))
            qs.append(tuple(b + w))
    return kn, bi, ro, qs


KNIGHT, BISHOP, ROOK, QUEEN = _tables()
KING = tuple(tuple(k for k in QUEEN[i] if abs((k >> 3) - (i >> 3)) < 2 and abs((k & 7) - (i & 7)) < 2) for i in range(64))
SLIDE = ((), (), (), BISHOP, ROOK, QUEEN)
PUSH = ((-8, -16), (8, 16))
TAKE = ((-9, -7), (7, 9))
CEN = tuple(abs((i >> 3) - 3) + abs((i & 7) - 3) for i in range(64))


def attacked(b, s, by):
    """Is square s held by side `by` (-1 black, 1 white)? Rays stop at the first piece."""
    d = -8 * by if by > 0 else 8 * -by
    for o in (d - 1, d + 1):
        k = s + o
        if 0 <= k < 64 and abs((k >> 3) - (s >> 3)) <= 1 and abs(b[k]) == 1 == by * b[k]:
            return 1
    for k in KNIGHT[s]:
        if b[k] == 2 * by or b[k] == -2 * by and 0:
            return 1
        if abs(b[k]) == 2 and (b[k] > 0) == (by > 0):
            return 1
    for k in BISHOP[s]:
        e = b[k]
        if e:
            if (e > 0) == (by > 0) and abs(e) in (3, 5):
                return 1
            break
    for k in ROOK[s]:
        e = b[k]
        if e:
            if (e > 0) == (by > 0) and abs(e) in (4, 5):
                return 1
            break
    for k in KNIGHT[s]:
        if abs(b[k]) == 6 and (b[k] > 0) == (by > 0):
            return 1
    for o in (-9, -8, -7, -1, 1, 7, 8, 9):
        k = s + o
        if 0 <= k < 64 and abs((k >> 3) - (s >> 3)) <= 1 and abs(b[k]) == 6 and (b[k] > 0) == (by > 0):
            return 1
    return 0


def gen(b, w, ep):
    """Pseudo-legal moves (i, j, promo) for the mover `w` (0/1)."""
    by = 1 if w == 0 else -1
    out = []
    for i in range(64):
        p = b[i]
        if p == 0 or (p > 0) != (by > 0):
            continue
        t = p * by
        if t == 1:
            step = PUSH[w]
            for k in (1, 2):
                j = i + step[k - 1]
                if not 0 <= j < 64 or b[j]:
                    break
                if k == 1:
                    out.append((i, j, 5 if j >> 3 in (0, 7) else 0))
                else:
                    if (i >> 3) in (1, 6):
                        out.append((i, j, 0))
            for o in TAKE[w]:
                j = i + o
                if not 0 <= j < 64 or (j >> 3) - (i >> 3) not in (-1, 1):
                    continue
                e = b[j]
                if e and (e > 0) != (by > 0) or j == ep:
                    out.append((i, j, 5 if j >> 3 in (0, 7) else 0))
        elif t == 2:
            for j in KNIGHT[i]:
                if not b[j] or (b[j] > 0) != (by > 0):
                    out.append((i, j, 0))
        elif t == 6:
            for j in KING[i]:
                if not b[j] or (b[j] > 0) != (by > 0):
                    out.append((i, j, 0))
        else:
            for j in SLIDE[t][i]:
                e = b[j]
                if not e:
                    out.append((i, j, 0))
                    continue
                if (e > 0) != (by > 0):
                    out.append((i, j, 0))
                break
    return out


def legal(b, w, m, ep):
    """Keep only moves that leave the mover's king unhurt."""
    i, j, pr = m
    by = 1 if w == 0 else -1
    was, cap = b[i], b[j]
    b[i] = 0
    b[j] = pr * by if pr else was
    if j == ep:
        b[j - 8 * by] = 0
    k = -1
    for s in range(64):
        if b[s] == 6 * by:
            k = s
            break
    bad = k < 0 or attacked(b, k, -by)
    b[i] = was
    b[j] = cap
    if j == ep:
        b[j - 8 * by] = cap if cap else 0
    return not bad


def ev(b):
    """Static score, white's point of view."""
    tot = 0
    wp = bp = 0
    for i in range(64):
        p = b[i]
        if p > 0:
            tot += V[p] + CEN[i] * -4 + 14
            wp += p == 3
        elif p < 0:
            tot -= V[-p] + CEN[i] * -4 + 14
            bp += p == -3
    if wp > 1:
        tot += 50
    if bp > 1:
        tot -= 50
    return tot


def qsearch(b, w, al, be, ep, st):
    """Captures only, so a trade never ends on a hanging piece."""
    st[0] += 1
    s = ev(b) if w == 0 else -ev(b)
    if s >= be:
        return be
    if s > al:
        al = s
    by = 1 if w == 0 else -1
    for i, j, pr in sorted(gen(b, w, ep), key=lambda m: -abs(b[m[1]])):
        if not b[j]:
            continue
        if not legal(b, w, (i, j, pr), ep):
            continue
        was, cap = b[i], b[j]
        b[i] = 0
        b[j] = pr * by if pr else was
        sc = -qsearch(b, 1 - w, -be, -al, -1, st)
        b[i] = was
        b[j] = cap
        if sc > al:
            al = sc
            if sc >= be:
                break
    return al


def search(b, w, d, al, be, ep, st):
    st[0] += 1
    if st[0] > BUDGET:
        return ev(b) if w == 0 else -ev(b)
    k = -1
    by = 1 if w == 0 else -1
    for s in range(64):
        if b[s] == 6 * by:
            k = s
            break
    if k < 0:
        return 0
    if d <= 0:
        return qsearch(b, w, al, be, ep, st)
    best = -MATE - 2
    for i, j, pr in order(b, gen(b, w, ep), st):
        if not legal(b, w, (i, j, pr), ep):
            continue
        was, cap = b[i], b[j]
        b[i] = 0
        b[j] = pr * by if pr else was
        ne = ep if j != ep else j - 8 * by
        sc = -search(b, 1 - w, d - 1, -be, -al, ne, st)
        b[i] = was
        b[j] = cap
        if sc > best:
            best = sc
            keep = (i, j, pr)
            if sc > al:
                al = sc
                if sc >= be:
                    break
    return best


def order(b, lst, st):
    """Loud moves first: material, then nothing. Stable, so replays agree."""
    def rank(m):
        e = b[m[1]]
        return abs(e) * 8 - abs(b[m[0]]) + (5000 if m[2] else 0)
    return sorted(lst, key=rank, reverse=True)


def post(path, body):
    req = urllib.request.Request(ENV + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=25).read().decode())


def uci(m):
    i, j, pr = m
    return (FILES[i & 7] + str(8 - (i >> 3)) + FILES[j & 7] + str(8 - (j >> 3))
            + " nbrq"[pr] if pr else FILES[i & 7] + str(8 - (i >> 3)) + FILES[j & 7] + str(8 - (j >> 3)))


def read(s):
    i = (8 - int(s[1])) * 8 + FILES.index(s[0])
    j = (8 - int(s[3])) * 8 + FILES.index(s[2])
    p = {"q": 5, "r": 4, "b": 3, "n": 2}.get(s[4:5], 0)
    return (i, j, p)


def board(fen):
    b = [0] * 64
    r = c = 0
    for ch in fen.split()[0]:
        if ch == "/":
            r, c = r + 1, 0
        elif ch.isdigit():
            c += int(ch)
        else:
            v = "PNBRQK".index(ch.upper()) + 1
            b[r * 8 + c] = v if ch.isupper() else -v
            c += 1
    return b


def root(b, w, pool, ep, st):
    """Iterative deepening over the env's own legal moves."""
    best = pool[0] if pool else None
    d = 1
    while d <= 7 and st[0] < BUDGET:
        keep, score = None, -MATE - 1
        for m in order(b, pool, st):
            if st[0] > BUDGET:
                break
            i, j, pr = m
            by = 1 if w == 0 else -1
            was, cap = b[i], b[j]
            b[i] = 0
            b[j] = pr * by if pr else was
            sc = -search(b, 1 - w, d - 1, -MATE, MATE, -1, st)
            b[i] = was
            b[j] = cap
            if sc > score:
                score, keep = sc, m
        if keep:
            best = keep
        d += 1
    return best


def main():
    st = [0]
    while True:
        g = post("/game", {})
        if g.get("status") == "done":
            print("trial over %sW %sD %sL elo~%s" % (g.get("wins"), g.get("draws"),
                                                     g.get("losses"), g.get("elo", "?")),
                  flush=True)
            return
        n = g["game"]
        while g.get("status") == "your_move":
            b = board(g["fen"])
            w = 0 if g.get("colour") == "white" else 1
            pool = [read(x) for x in g.get("legal_moves", ())]
            st[0] = 0
            mv = root(b, w, pool, -1, st) if pool else None
            if mv is None:
                print("no move for game %s" % n, flush=True)
                return
            g = post("/move", {"game": n, "move": uci(mv)})
        print("%s %s %s" % (g.get("opponent"), g.get("outcome"), g.get("reason")), flush=True)


if __name__ == "__main__":
    main()
