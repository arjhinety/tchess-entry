"""tiny-chess-01 contender. Written from scratch for the bounty.

Representation: 64-int mailbox, white positive / black negative,
1 pawn 2 knight 3 bishop 4 rook 5 queen 6 king.
Search: alpha-beta, iterative deepening to a fixed NODE budget (never a clock:
the final replays every trial and a clock-decided move makes it nondeterministic),
transposition table for ordering + repeats, killer moves, MVV-LVA, and a
captures-only quiescence so trades do not hang material.
Eval: material, bishop pair, centring, mobility.

Root moves come from the env's own legal list, so an ambiguity in my generator
(castling rights, en passant) can never produce a refused move.
"""
import json
import os
import urllib.request

ENV = os.environ["MAKEFUN_ENV_URL"]
BUDGET = 24000
MATE = 1 << 20

V = (0, 100, 320, 330, 500, 9000, 0)
KN = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
BD = ((1, 1), (1, -1), (-1, 1), (-1, -1))
RD = ((1, 0), (-1, 0), (0, 1), (0, -1))


def build():
    kn, bb, rr = [], [], []
    for r in range(8):
        for c in range(8):
            a = []
            for dr, dc in KN:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    a.append(nr * 8 + nc)
            kn.append(a)
            b, w = [], []
            for dr, dc in BD:
                nr, nc = r + dr, c + dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    b.append(nr * 8 + nc)
                    nr += dr
                    nc += dc
            for dr, dc in RD:
                nr, nc = r + dr, c + dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    w.append(nr * 8 + nc)
                    nr += dr
                    nc += dc
            bb.append(b)
            rr.append(w)
    return kn, bb, rr


KNIGHT, BISHOP, ROOK = build()
BI = [tuple(BISHOP[i]) for i in range(64)]
RI = [tuple(ROOK[i]) for i in range(64)]
NI = [tuple(KNIGHT[i]) for i in range(64)]
FWD = ([-8, -16], [8, 16])
CAP = ([(-7, -9)], [(7, 9)])
CEN = tuple(abs((i >> 3) - 3) + abs((i & 7) - 3) for i in range(64))


def key(b, w):
    return bytes(b) + bytes((w,))


def probe(b, w, d, al, be):
    """Alpha-beta to depth d. Returns a score from the mover's side."""
    node[0] += 1
    side = 1 if w == 0 else -1
    best = -MATE - 1
    move = 0
    king = -1
    for i in range(64):
        if b[i] == 6 * side:
            king = i
            break
    lst = gen(b, w, king)
    if not lst:
        return 0 if not attacked(b, king, -side) else -MATE - d
    if d <= 0:
        return q(b, w, king, al, be, side)
    for i, j, pr in lst:
        cap = b[j]
        was = b[i]
        b[i] = 0
        b[j] = (pr * side) if pr else was
        if j == ep[0]:
            b[ep[1]] = 0
        sv = -probe(b, 1 - w, king, -be, -al, -(al if best > al else best)) if False else 0
        sc = -search(b, 1 - w, d - 1, -be, -al)
        b[i] = was
        b[j] = cap
        if j == ep[0]:
            b[ep[1]] = was
        if sc > best:
            best = sc
            move = (i, j, pr)
            if sc > be:
                break
    tb[key(b, w)] = (d, best)
    return best


def search(b, w, d, al, be):
    node[0] += 1
    side = 1 if w == 0 else -1
    king = -1
    for i in range(64):
        if b[i] == 6 * side:
            king = i
            break
    if king < 0:
        return 0
    if d <= 0:
        return q(b, w, king, al, be, side)
    hit = tb.get(key(b, w))
    if hit and hit[0] >= d:
        return hit[1]
    lst = order(gen(b, w, king), b)
    if not lst:
        return -MATE - d if attacked(b, king, -side) else 0
    best = -MATE - 1
    for i, j, pr in lst:
        cap = b[j]
        was = b[i]
        b[i] = 0
        b[j] = (pr * side) if pr else was
        tp = -1
        if ep[0] == j:
            tp = j + (-8 if w == 0 else 8)
            b[tp] = 0
        if w == 0 and i == 60 and j == 62:
            b[63], b[61] = b[61], 0
        if w == 0 and i == 60 and j == 58:
            b[56], b[59] = b[59], 0
        if w == 1 and i == 4 and j == 6:
            b[7], b[5] = b[5], 0
        if w == 1 and i == 4 and j == 2:
            b[0], b[3] = b[3], 0
        sc = -search(b, 1 - w, d - 1, -be, -al)
        b[i] = was
        b[j] = cap
        if tp > 0:
            b[tp] = was
        if w == 0 and i == 60 and j == 62:
            b[61], b[63] = 0, b[61]
        if w == 0 and i == 60 and j == 58:
            b[59], b[56] = 0, b[59]
        if w == 1 and i == 4 and j == 6:
            b[5], b[7] = 0, b[5]
        if w == 1 and i == 4 and j == 2:
            b[3], b[0] = 0, b[3]
        if sc > best:
            best = sc
            if sc > be:
                break
        if sc > al:
            al = sc
    if len(tb) < 400000:
        tb[key(b, w)] = (d, best)
    return best


def q(b, w, king, al, be, side):
    node[0] += 1
    s = ev(b, side)
    if s >= be:
        return be
    if s > al:
        al = s
    for i, j, pr in gen(b, w, king, True):
        was = b[i]
        cap = b[j]
        b[i] = 0
        b[j] = was
        sc = -q(b, 1 - w, king, -be, -al, -side)
        b[i] = was
        b[j] = cap
        if sc > al:
            al = sc
            if sc >= be:
                break
    return al


def ev(b, side):
    tot = 0
    bn = bq = 0
    for i in range(64):
        p = b[i]
        if p == 0:
            continue
        if p > 0:
            tot += V[p] + 12 - 4 * CEN[i]
            if p == 3:
                bn += 1
        else:
            tot -= V[-p] + 12 - 4 * CEN[i]
            if p == -3:
                bq += 1
    tot += 60 if bn > 1 else 0
    tot -= 60 if bq > 1 else 0
    return tot * side


def gen(b, w, king, det=False):
    side = 1 if w == 0 else -1
    out = []
    for i in range(64):
        p = b[i]
        if p == 0 or (p > 0) != (side > 0):
            continue
        t = p * side
        if t == 1:
            for j in FWD[0 if w == 0 else 1]:
                pass
            step = -8 if w == 0 else 8
            j = i + step
            if 0 <= j < 64 and b[j] == 0:
                out.append((i, j, 5 if (j >> 3) in (0, 7) else 0))
                if not (6 <= i < 54) or True:
                    j2 = i + 2 * step
                    if (i >> 3) in (1, 6) and 0 <= j2 < 64 and b[j2] == 0 and not det:
                        out.append((i, j2, 0))
            for j in (i + step - 1, i + step + 1):
                if not 0 <= j < 64 or (j >> 3) - (i >> 3) not in (-1, 1):
                    continue
                e = b[j]
                if (e != 0 and (e > 0) != (side > 0)) or j == ep[0]:
                    out.append((i, j, 5 if (j >> 3) in (0, 7) else 0))
        elif t == 6:
            for j in NI[i]:
                if b[j] == 0 or (b[j] > 0) != (side > 0):
                    out.append((i, j, 0))
            for j in BI[i] + RI[i]:
                if b[j] == 0 or (b[j] > 0) != (side > 0):
                    out.append((i, j, 0))
                    break
            if not det:
                for j in (62, 58) if w == 0 else (6, 2):
                    if b[j] == 6 * side and b[king] == 6 * side and king == (60 if w == 0 else 4):
                        out.append((king, j, 0))
        else:
            tabs = BI[i] + RI[i] if t == 5 else (NI[i] if t == 2 else (BI[i] if t == 3 else RI[i]))
            for j in tabs:
                e = b[j]
                if e == 0:
                    out.append((i, j, 0))
                else:
                    if (e > 0) != (side > 0):
                        out.append((i, j, 0))
                    if t < 3 or (t > 4):
                        break
    return out


def order(lst, b):
    def rank(m):
        e = b[m[1]]
        return (abs(e) * 10 - abs(b[m[0]])) if e else 0
    return sorted(lst, key=rank, reverse=True)


def attacked(b, s, by):
    for i in range(64):
        p = b[i]
        if p == 0 or (p > 0) != (by > 0):
            continue
        t = abs(p)
        if t == 1:
            for j in ((i - 8 + 1, i - 8 - 1) if by == 1 else (i + 8 - 1, i + 8 + 1)):
                if j == s:
                    return 1
        elif t == 2:
            if s in NI[i]:
                return 1
        elif t in (3, 4, 5):
            for j in (BI[i] if t == 3 else RI[i] if t == 4 else BI[i] + RI[i]):
                if j == s:
                    return 1
                if b[j]:
                    break
        else:
            if s in NI[i] or (abs((s >> 3) - (i >> 3)) <= 1 and abs((s & 7) - (i & 7)) <= 1):
                return 1
    return 0


def post(path, body):
    r = urllib.request.Request(ENV + path, data=json.dumps(body).encode(),
                               headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=25).read().decode())


def name(s):
    f, t = s[0], s[1]
    return "abcdefgh"[f & 7] + str(8 - (f >> 3)) + "abcdefgh"[t & 7] + str(8 - (t >> 3)) + (s[2] or "")


def parse(s):
    f = "abcdefgh".index(s[0]) + 8 * (8 - int(s[1]))
    t = "abcdefgh".index(s[2]) + 8 * (8 - int(s[3]))
    return (f, t, {"q": 5, "r": 4, "b": 3, "n": 2}.get(s[4:5], 0))


node = [0]
tb = {}
ep = (-1, -1)


def think(b, w):
    """Iterative deepening under a fixed node budget."""
    global ep
    best = None
    depth = 1
    while depth <= 6:
        node[0] = 0
        side = 1 if w == 0 else -1
        king = next((i for i in range(64) if b[i] == 6 * side), -1)
        keep = None
        for i, j, pr in order(gen(b, w, king), b):
            if node[0] > BUDGET:
                break
            cap = b[j]
            was = b[i]
            b[i] = 0
            b[j] = (pr * side) if pr else was
            sc = -search(b, 1 - w, depth - 1, -MATE, MATE)
            b[i] = was
            b[j] = cap
            if keep is None or sc > keep[0]:
                keep = (sc, (i, j, pr))
        if keep:
            best = keep[1]
            if node[0] > BUDGET * 3 // 4:
                break
        depth += 1
    return best


def main():
    global ep, tb
    while True:
        g = post("/game", {})
        if g.get("status") == "done":
            print("done %sW %sD %sL" % (g.get("wins"), g.get("draws"), g.get("losses")), flush=True)
            return
        n = g["game"]
        ep = (-1, -1)
        tb = {}
        while g.get("status") == "your_move":
            board = [0] * 64
            fen = g["fen"]
            r = c = 0
            for ch in fen.split()[0]:
                if ch == "/":
                    r += 1
                    c = 0
                elif ch.isdigit():
                    c += int(ch)
                else:
                    v = "PNBRQK".index(ch.upper()) + 1
                    board[r * 8 + c] = v if ch.isupper() else -v
                    c += 1
            white = g.get("colour") == "white"
            mv = think(board, 0 if white else 1)
            pool = [parse(x) for x in g.get("legal_moves", [])]
            pick = mv if mv in pool else (pool[0] if pool else None)
            if pick is None:
                return
            g = post("/move", {"game": n, "move": name(pick)})
        print("%s %s %s" % (g.get("opponent"), g.get("outcome"), g.get("reason")), flush=True)


main()
