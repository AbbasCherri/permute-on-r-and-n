from fractions import Fraction
from math import gcd

# Roughly the memory cost in bits of one entry in a Python set, used only to
# choose between holding the totals as bits and holding them as values.
_SET_OVERHEAD = 256

# Below this many bits the whole reachable set fits comfortably in one mask, so
# the windowed shortcut has nothing to gain.
_WINDOW_FROM = 1 << 22

# Totals are only put in int64 arrays while they cannot overflow one.
_INT64_CEILING = 1 << 62

# More totals than this and no machine is going to hold them either.
_SPARSE_CEILING = 1 << 40

# Fewer totals than this and a Python set beats the array machinery.
_ARRAY_FROM = 1 << 14

# Elements per intermediate array, which bounds the working set on the GPU.
_ARRAY_CHUNK = 1 << 24

# Peak working memory runs to a few times whatever is being held, so only this
# fraction of what is free can be counted on.
_SAFETY = 4


def count_distinct_sums(r, n, ndigits=9, return_sums=False, max_bits=None,
                        progress=None):
    if n < 0:
        raise ValueError("n must be non-negative")

    # progress is anything tqdm-shaped: called with a total, then updated and
    # closed. It is owned here rather than by the loops below so that it can be
    # filled to n on the way out, terms that were skipped included.
    bar = _Progress(progress, n)
    answer = _counted(r, n, ndigits, return_sums, max_bits, bar)
    bar.finish()
    return answer


class _Progress:
    def __init__(self, factory, total):
        self.bar = None if factory is None else factory(total=total)
        self.total = total
        self.done = 0

    def step(self):
        self.done += 1
        if self.bar is not None:
            self.bar.update(1)

    def finish(self):
        # Skipping the rest of the terms still settles them, so the bar ends
        # full rather than wherever the shortcut happened to stop.
        if self.bar is not None:
            self.bar.update(self.total - self.done)
            self.bar.close()


def _counted(r, n, ndigits, return_sums, max_bits, bar):
    if max_bits is None:
        max_bits = free_memory() * 8 // _SAFETY

    scale, values = _on_grid(r, ndigits)
    if n == 0:
        zero = Fraction(0) if ndigits is None else 0.0
        return (1, [zero]) if return_sums else 1
    if not values:
        return (0, []) if return_sums else 0

    # Slide the smallest value to 0 and divide out the common spacing, which is
    # what keeps the grid small: [1, 2, 3] at ndigits=9 becomes [0, 1, 2]
    # instead of three points a billion apart.
    base = values[0]
    spacing = 0
    for value in values:
        spacing = gcd(spacing, value - base)

    def decode(index):  # grid index back to a value of the original kind
        total = n * base + spacing * index
        return Fraction(total, scale) if ndigits is None else total / scale

    if spacing == 0:  # a single distinct value, so a single possible total
        return (1, [decode(0)]) if return_sums else 1

    ws = [(value - base) // spacing for value in values]
    top = ws[-1]

    # One bit per grid point in [0, n*top] against one entry per reachable
    # total: widely spaced values with a small n leave the grid mostly empty,
    # so compare the two sizes before committing to either.
    dense_bits = n * top + 1
    expected = _multiset_bound(n, len(ws), dense_bits // _SET_OVERHEAD + 1)

    def by_totals():
        totals = _sparse_totals(ws, n, expected, n * top, bar)
        if return_sums:
            return len(totals), [decode(int(y)) for y in totals]
        return len(totals)

    if expected * _SET_OVERHEAD < dense_bits:
        return by_totals()

    # For a large n only the two ends of the reachable set have to be looked at.
    if not return_sums and dense_bits > _WINDOW_FROM:
        count = _count_from_ends(ws, top, n, max_bits)
        if count is not None:
            return count

    if dense_bits > max_bits:
        # No mask that size will fit, so the totals are the only way left. The
        # count of them is an upper bound and often a loose one, so this is
        # worth trying unless even that bound is hopeless.
        if expected > _SPARSE_CEILING:
            raise MemoryError(
                f"needs {dense_bits} bits, or upwards of {expected} totals: the"
                " values sit on too fine a grid. Round them with ndigits.")
        return by_totals()

    count, mask = _dp_count(_runs(ws), top, n, return_sums, bar)
    if return_sums:
        return count, [decode(index) for index in _set_bits(mask)]
    return count


def max_ndigits(values, n, budget=None, ceiling=30):
    # The finest decimal grid whose totals this machine can still hold, which
    # is what caps ndigits: a finer grid means a larger integer per value, and
    # so more grid points between the smallest and largest total. Lists of
    # integers are unaffected by the grid and come back with the ceiling.
    if budget is None:
        budget = free_memory() // _SAFETY
    bits = budget * 8

    ratios = [value.as_integer_ratio() for value in values]
    low, high = 0, ceiling
    while low < high:
        middle = (low + high + 1) // 2
        if n * _grid_top(ratios, middle) + 1 <= bits:
            low = middle
        else:
            high = middle - 1
    return low


def gpu_ready():
    # Whether the totals will be counted on a GPU rather than in a Python set.
    return _array_module() is not None


def free_memory():
    # Bytes free for the totals, which is the GPU's when they would live there.
    free = _free_ram()
    xp = _array_module()
    if xp is not None:
        free = min(free, _free_gpu(xp))
    return free


def _free_ram():
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import os
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return 1 << 31  # no way to tell, so assume a modest 2 GiB


def _free_gpu(xp):
    # What the device has left, plus whatever cupy holds in its pool without
    # using, since that comes back for free.
    try:
        return int(xp.cuda.Device().mem_info[0]
                   + xp.get_default_memory_pool().free_bytes())
    except Exception:
        return 1 << 62  # cannot tell, so leave it to the allocator


def _grid_top(ratios, ndigits):
    # Largest grid index the values reach once slid to start at 0 and divided
    # by their common spacing: the width the totals are counted over.
    _, grid = _quantise(ratios, ndigits)
    if len(grid) < 2:
        return 0
    spacing = 0
    for value in grid:
        spacing = gcd(spacing, value - grid[0])
    return (grid[-1] - grid[0]) // spacing


def _on_grid(values, ndigits):
    # as_integer_ratio is exact for ints, floats and Fractions alike, so a
    # value sitting right on a rounding boundary cannot fall the wrong side of
    # it and throw the common spacing off.
    return _quantise([value.as_integer_ratio() for value in values], ndigits)


def _quantise(ratios, ndigits):
    # Returns the scale the grid counts in and the values as exact multiples.
    if ndigits is None:
        # Exact: a common denominator rounds nothing away at all.
        scale = 1
        for _, denominator in ratios:
            scale = scale // gcd(scale, denominator) * denominator
        return scale, sorted({num * (scale // den) for num, den in ratios})

    scale = 10 ** ndigits
    grid = {num * scale if den == 1 else (2 * num * scale + den) // (2 * den)
            for num, den in ratios}
    return scale, sorted(grid)

def _sparse_totals(ws, n, expected, biggest, bar):
    # Deduplicating int64 arrays by sorting only beats a Python set when the
    # sort is a GPU's, so this asks for one and otherwise stays with the set,
    # which has no size ceiling either. Totals too large for an int64 likewise
    # stay in Python ints, which cannot overflow.
    xp = _array_module() if expected > _ARRAY_FROM and biggest < _INT64_CEILING else None
    if xp is None:
        totals = {0}
        for _ in range(n):
            totals = {total + w for total in totals for w in ws}
            bar.step()
            _room_for(len(totals) * _SET_OVERHEAD // 8, _free_ram(), len(totals))
        return sorted(totals)

    values = xp.asarray(ws, dtype=xp.int64)
    totals = xp.zeros(1, dtype=xp.int64)
    try:
        for _ in range(n):
            # How many totals there are is what the answer is, so it cannot be
            # known ahead of time. Check there is still room for them instead.
            _room_for(totals.size * 8 * _SAFETY, _free_gpu(xp), totals.size)
            # Fold the values in a chunk at a time, keeping the intermediate a
            # bounded size and never holding more than two large arrays at once.
            step = max(1, _ARRAY_CHUNK // totals.size)
            grown = None
            for i in range(0, values.size, step):
                part = xp.unique(totals[:, None] + values[i:i + step])
                grown = part if grown is None else xp.unique(xp.concatenate((grown, part)))
            totals = grown
            bar.step()
    except Exception as failure:  # cupy's own out-of-memory, as a backstop
        if type(failure).__name__ != "OutOfMemoryError":
            raise
        _room_for(totals.size * 8 * _SAFETY, 0, totals.size)
    return totals


def _room_for(needed, free, reached):
    if needed > free:
        raise MemoryError(
            f"{reached} totals reached and the next term needs {needed} bytes"
            f" against {free} free: lower ndigits, or n, or r.")


_ARRAY_MODULE = ...  # not looked up yet


def _array_module():
    # cupy, but only when a GPU is really there to run it (a Colab T4, say).
    # On a CPU its sort is no faster than the set it would replace.
    global _ARRAY_MODULE
    if _ARRAY_MODULE is ...:
        _ARRAY_MODULE = None
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() > 0:
                _ARRAY_MODULE = cupy
        except Exception:  # not installed, or installed without a device
            pass
    return _ARRAY_MODULE


# --- the totals held as bits ------------------------------------------------


def _runs(ws):
    # Group the values into maximal runs of consecutive grid points, so a run
    # costs one shift plus log2(length) smears instead of one shift per value.
    # Large value sets are dense on the grid, which makes the runs long.
    runs = []
    start = previous = ws[0]
    for w in ws[1:]:
        if w == previous + 1:
            previous = w
        else:
            runs.append((start, previous - start + 1))
            start = previous = w
    runs.append((start, previous - start + 1))
    return runs


def _one_more_term(mask, runs):
    # Adding a term is an OR of shifted copies of the mask, all of it inside
    # CPython's big-int code rather than a Python loop over the totals.
    nxt = 0
    for start, length in runs:
        block = mask << start
        covered = 1
        while covered < length:  # block |= block << 1 ... << length-1
            grow = min(covered, length - covered)
            block |= block << grow
            covered += grow
        nxt |= block
    return nxt


def _dp_count(runs, top, n, need_totals, bar):
    # Only ever called once the caller knows the full mask will fit.
    mask = 1
    shape = None
    shape_round = 0
    holes = -1

    for k in range(1, n + 1):
        mask = _one_more_term(mask, runs)
        bar.step()
        if need_totals or k == n:
            continue

        # A settled reachable set has the same number of holes from one term to
        # the next. That test is a single pass over the mask, so use it to
        # decide whether the structural check below is worth running at all.
        holes, previous_holes = k * top + 1 - _popcount(mask), holes
        if holes != previous_holes:
            continue

        # Once both ends have settled into a repeating pattern of gaps with a
        # solid run between them, every later term only widens that run, so the
        # count is linear in n from here and the remaining terms can be skipped.
        shape, previous = _shape(mask, k * top, top), shape
        if shape is not None and shape == previous and shape_round == k - 1:
            return n * top + 1 - holes, None
        shape_round = k

    return _popcount(mask), mask


def _shape(mask, tot, top):
    # The mask as "gaps near 0", a solid run, "gaps near the top". None while
    # the solid run is too short to guarantee that the next term reproduces the
    # same two end patterns.
    size = tot + 1
    mid = size >> 1
    gaps = ~mask & ((1 << size) - 1)

    low_gaps = gaps & ((1 << mid) - 1)
    high_gaps = gaps >> mid
    last_low = low_gaps.bit_length() - 1
    first_high = mid + (high_gaps & -high_gaps).bit_length() - 1 if high_gaps else size
    if first_high - last_low - 1 < top:
        return None

    return (last_low, mask & ((1 << (last_low + 1)) - 1),
            tot - first_high, mask >> first_high)


# --- the ends alone, which is all a large n needs ---------------------------


def _count_from_ends(ws, top, n, max_bits):
    # For a large enough n the reachable totals are one solid block with a
    # fixed pattern of gaps at either end, so counting the gaps at the two ends
    # is enough, and each end settles inside a window that does not grow with n.
    # None when that has not happened yet, leaving the caller to walk the DP.
    step = max(b - a for a, b in zip(ws, ws[1:])) if len(ws) > 1 else top
    low = _settled_gaps(_runs(ws), step, n, max_bits)
    if low is None:
        return None

    # The top end is the bottom end of the same problem with every value
    # mirrored, since a total of x from n terms is a total of n*top - x there.
    # Mirroring reverses the gaps between values, so the widest is unchanged.
    high = _settled_gaps(_runs([top - w for w in reversed(ws)]), step, n, max_bits)
    if high is None:
        return None

    settled_low, gaps_low = low
    settled_high, gaps_high = high
    if n < settled_low + settled_high:  # the two ends may not have met yet
        return None
    return n * top + 1 - gaps_low - gaps_high


def _settled_gaps(runs, step, n, max_bits):
    # Repeat the DP inside a fixed window until the totals in it stop changing.
    # The window is wide enough only if it ends in a solid run of at least step
    # totals, step being the widest gap between neighbouring values: a run that
    # long, shifted by each value in turn, lands back-to-back and so can only
    # ever grow. Widen and retry when the window does not show one.
    width = 2 * step + 2
    while width <= max_bits:
        limit = (1 << width) - 1
        # Values at or past the window can only shift totals straight out of
        # it, and a run may as well stop at the edge.
        near = [(start, min(length, width - start))
                for start, length in runs if start < width]
        mask = 1
        # A term that changes nothing has settled and a term that changes
        # something sets at least one more bit, so width + 1 terms is a cap.
        for k in range(1, min(n, width + 1) + 1):
            nxt = _one_more_term(mask, near) & limit
            if nxt == mask:
                holes = ~mask & limit
                if width - holes.bit_length() >= step:
                    return k, _popcount(holes)
                break  # window too narrow to show the solid run
            mask = nxt
        else:
            return None  # n terms is not enough for this end to settle
        width *= 2
    return None


# --- odds and ends ----------------------------------------------------------


def _multiset_bound(n, count, cap):
    # C(n + count - 1, count - 1), the number of multisets and so an upper
    # bound on the number of distinct totals. Stops early once it passes cap,
    # since past that point only the comparison against cap matters.
    total = 1
    for i in range(1, count):
        total = total * (n + i) // i
        if total > cap:
            break
    return total


def _set_bits(mask):
    # Positions of the set bits, read a byte at a time so the cost stays linear
    # in the size of the mask rather than quadratic.
    data = mask.to_bytes((mask.bit_length() + 7) // 8, "little")
    return [i * 8 + b for i, byte in enumerate(data) if byte
            for b in range(8) if byte >> b & 1]


try:
    _popcount = int.bit_count  # Python 3.10+
except AttributeError:
    def _popcount(x):
        return bin(x).count("1")


if __name__ == "__main__":
    import random
    import time

    r = [1, 2, 3]
    n = 2
    count = count_distinct_sums(r, n)
    print(f"Number of distinct sums with {n} terms from {r}: {count}")

    count, sums = count_distinct_sums(r, n, return_sums=True)
    print(f"The sums themselves: {sums}")

    # Rounded to 9 decimals the three values sit on a 0.1 grid; exactly as
    # given they are three unrelated binary fractions and nothing coincides.
    print(f"[0.1, 0.2, 0.3], n=3, rounded: {count_distinct_sums([0.1, 0.2, 0.3], 3)}")
    print(f"[0.1, 0.2, 0.3], n=3, exact:   "
          f"{count_distinct_sums([0.1, 0.2, 0.3], 3, ndigits=None)}")

    r = [random.randint(0, 10 ** 6) for _ in range(100000)]
    n = 10 ** 12
    started = time.perf_counter()
    count = count_distinct_sums(r, n)
    print(f"{len(r)} values, n={n}: {count} distinct sums"
          f" in {time.perf_counter() - started:.2f}s")
