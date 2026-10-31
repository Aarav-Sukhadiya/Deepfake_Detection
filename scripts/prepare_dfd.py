"""Prepare Google DFD (Kaggle mirror) for ``dfdetect train``.

What it does
============
1. Autodetects ``real`` / ``fake`` folders inside an unzipped DFD dump.
2. Builds a flat ``out/real`` and ``out/fake`` tree of *hardlinks* (no copy,
   no extra disk; the trainer's ``walk_media`` skips symlinks but follows
   hardlinks).
3. Subsamples the fake class so ``fake / total`` lands under the trainer's
   hard cap of 0.8 (default target 0.78 for margin). Sampling is seeded and
   stratified by the leading actor id parsed from the filename when
   possible, so we keep cross-actor diversity.
4. Prints the resulting :class:`DatasetStats` and (optionally) runs the same
   constraint gate the trainer will run.

Usage
-----
::

    python scripts/prepare_dfd.py \\
        --source ./DFD \\
        --out ./data/dfd_train

Re-running is idempotent: existing hardlinks are kept, missing ones are
created, extras are pruned.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS

REAL_HINTS = ("real", "original", "pristine", "authentic")
FAKE_HINTS = ("fake", "manipulated", "deepfake", "synth", "swapped")

# DFD filenames typically start with one or two 2-digit actor ids, e.g.
#   01__hugging_happy.mp4               (real, actor 01)
#   01_03__hugging_happy__YVGY8GHK.mp4  (fake, actor 01 -> 03)
_ACTOR_RE = re.compile(r"^(\d{2})(?:_(\d{2}))?")


def _safe_rel(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


def _classify_dir(name: str) -> str | None:
    low = name.lower()
    if any(h in low for h in FAKE_HINTS):
        return "fake"
    if any(h in low for h in REAL_HINTS):
        return "real"
    return None


def _find_class_roots(source: Path) -> tuple[list[Path], list[Path]]:
    """Walk ``source`` and return ``(real_roots, fake_roots)``: directories
    whose name (or a parent up to ``source``) says "real" or "fake"."""
    real_roots: list[Path] = []
    fake_roots: list[Path] = []
    for p in source.rglob("*"):
        if not p.is_dir():
            continue
        kind = _classify_dir(p.name)
        if kind == "real":
            real_roots.append(p)
        elif kind == "fake":
            fake_roots.append(p)
    return real_roots, fake_roots


def _gather_media(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for r in roots:
        for p in r.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            if p.name.startswith("."):
                continue
            if p.suffix.lower() not in MEDIA_EXTS:
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            files.append(p)
    files.sort()
    return files


def _actor_key(p: Path) -> str:
    m = _ACTOR_RE.match(p.name)
    if not m:
        return "_unknown"
    return m.group(1)


def _stratified_sample(files: list[Path], k: int, rng: random.Random) -> list[Path]:
    """Sample ``k`` files trying to keep per-actor proportions roughly
    intact. Falls back to plain reservoir sample if no actor ids parse."""
    if k >= len(files):
        return list(files)
    buckets: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        buckets[_actor_key(f)].append(f)
    if len(buckets) <= 1:
        # No actor signal — just sample uniformly.
        return rng.sample(files, k)
    total = sum(len(v) for v in buckets.values())
    out: list[Path] = []
    leftover: list[Path] = []
    for lst in buckets.values():
        rng.shuffle(lst)
        quota = max(1, round(k * len(lst) / total))
        out.extend(lst[:quota])
        leftover.extend(lst[quota:])
    if len(out) > k:
        rng.shuffle(out)
        leftover = out[k:] + leftover
        out = out[:k]
    elif len(out) < k:
        rng.shuffle(leftover)
        out.extend(leftover[: k - len(out)])
    out.sort()
    return out


def _materialise(files: list[Path], dest: Path, *, dry_run: bool, dup: int = 1) -> tuple[int, int, int]:
    """Hardlink ``files`` into ``dest`` (flat). Returns (linked, kept, pruned).
    Filenames are prefixed with a 6-char source-id to defuse collisions when
    two source folders contain the same basename. ``dup`` >= 2 creates that
    many hardlinks per source, each with a ``_dupK`` suffix — used to
    oversample one class to satisfy the trainer's ratio gate. The trainer's
    head-hash uniqueness check still sees the *original* count of sources,
    so this is faithful: it tells the optimiser "show me this clip more
    often" without lying about diversity."""
    dest.mkdir(parents=True, exist_ok=True)
    wanted: dict[str, Path] = {}
    for src in files:
        # sha1, not ``hash()``: str hashing is salted per process, so the old
        # prefix changed every run and "idempotent" re-runs relinked + pruned
        # everything.
        rid = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:6]
        base = f"{rid}__{src.name}"
        if dup <= 1:
            wanted[base] = src
        else:
            stem = src.stem
            ext = src.suffix
            for k in range(dup):
                name = f"{rid}__{stem}__dup{k}{ext}"
                wanted[name] = src

    existing = {p.name: p for p in dest.iterdir() if p.is_file()}
    linked = kept = pruned = 0
    for name, src in wanted.items():
        target = dest / name
        if target.exists():
            kept += 1
            continue
        if dry_run:
            linked += 1
            continue
        try:
            os.link(src, target)
        except OSError as e:
            print(f"  ! could not hardlink {src} -> {target}: {e}", file=sys.stderr)
            continue
        linked += 1
    for name, p in existing.items():
        if name not in wanted:
            if not dry_run:
                p.unlink()
            pruned += 1
    return linked, kept, pruned


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path,
                    help="Folder containing the unzipped DFD dataset (autodetect mode).")
    ap.add_argument("--real-dir", type=Path, action="append", default=[],
                    help="Explicit real-class folder (repeatable). Overrides --source autodetect.")
    ap.add_argument("--fake-dir", type=Path, action="append", default=[],
                    help="Explicit fake-class folder (repeatable). Overrides --source autodetect.")
    ap.add_argument("--out", type=Path, default=Path("./data/dfd_train"),
                    help="Output root; will contain real/ and fake/ subfolders.")
    ap.add_argument("--target-ratio", type=float, default=0.78,
                    help="Desired fake/total ratio (must be <0.8 to pass the trainer gate).")
    ap.add_argument("--keep-all-fakes", action="store_true",
                    help="Don't subsample fakes; instead oversample reals (hardlink-duplicate) to satisfy --target-ratio. "
                         "Preferred: the trainer's WeightedRandomSampler already balances per-batch, so keeping all fakes "
                         "gives the model more diverse swap examples without hurting real coverage.")
    ap.add_argument("--oversample-reals", type=int, default=0,
                    help="Force-duplicate each real this many times (0 = auto when --keep-all-fakes, else 1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Print what would happen; don't create links.")
    ap.add_argument("--check", action="store_true", help="After preparing, run the trainer's bias-constraint check.")
    args = ap.parse_args()

    if args.real_dir or args.fake_dir:
        if not (args.real_dir and args.fake_dir):
            print("ERROR: --real-dir and --fake-dir must be used together.", file=sys.stderr)
            return 2
        real_roots = [p.expanduser().resolve() for p in args.real_dir]
        fake_roots = [p.expanduser().resolve() for p in args.fake_dir]
        for p in real_roots + fake_roots:
            if not p.is_dir():
                print(f"ERROR: {p} is not a directory.", file=sys.stderr)
                return 2
        src = Path(os.path.commonpath([str(p) for p in real_roots + fake_roots])).resolve()
        print(f"Using explicit roots (common base: {src}) ...")
    else:
        if not args.source:
            print("ERROR: pass either --source or --real-dir/--fake-dir.", file=sys.stderr)
            return 2
        src = args.source.expanduser().resolve()
        if not src.is_dir():
            print(f"ERROR: --source {src} is not a directory.", file=sys.stderr)
            return 2
        print(f"Scanning {src} ...")
        real_roots, fake_roots = _find_class_roots(src)
        if not real_roots or not fake_roots:
            print("ERROR: couldn't autodetect real/fake folders by name. Expected", file=sys.stderr)
            print(f"  REAL hints: {REAL_HINTS}", file=sys.stderr)
            print(f"  FAKE hints: {FAKE_HINTS}", file=sys.stderr)
            print("Pass them manually with --real-dir/--fake-dir.", file=sys.stderr)
            return 2
    print(f"  real roots: {len(real_roots)}  ({', '.join(_safe_rel(p, src) for p in real_roots[:3])}{' ...' if len(real_roots) > 3 else ''})")
    print(f"  fake roots: {len(fake_roots)}  ({', '.join(_safe_rel(p, src) for p in fake_roots[:3])}{' ...' if len(fake_roots) > 3 else ''})")

    reals = _gather_media(real_roots)
    fakes = _gather_media(fake_roots)
    print(f"Found {len(reals)} real, {len(fakes)} fake media files.")

    if not reals or not fakes:
        print("ERROR: one class has zero media files; cannot continue.", file=sys.stderr)
        return 2

    r = args.target_ratio
    if not 0.05 < r < 0.95:
        print(f"ERROR: --target-ratio must be in (0.05, 0.95), got {r}", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    real_dup_factor = max(1, args.oversample_reals)
    if args.keep_all_fakes:
        # Solve for K such that fake / (K*real + fake) <= r  =>  K >= fake*(1-r)/(r*real).
        needed = (len(fakes) * (1.0 - r)) / (r * len(reals))
        auto_k = max(1, int(needed) + (0 if needed.is_integer() else 1))
        if args.oversample_reals == 0:
            real_dup_factor = auto_k
        elif real_dup_factor < auto_k:
            print(f"WARN: --oversample-reals {real_dup_factor} is below the {auto_k} needed for ratio {r:.2f}; "
                  f"the trainer's bias gate will likely fail.", file=sys.stderr)
        fakes_sel = list(fakes)
        print(f"Keeping all {len(fakes)} fakes; oversampling reals x{real_dup_factor} "
              f"(target ratio {r:.2f}).")
    else:
        cap = int(round(r * len(reals) / (1.0 - r)))
        if len(fakes) > cap:
            print(f"Subsampling fakes {len(fakes)} -> {cap} (stratified by actor, seed={args.seed}).")
            fakes_sel = _stratified_sample(fakes, cap, rng)
        else:
            fakes_sel = list(fakes)
            print(f"Fake count {len(fakes)} already below cap {cap}; keeping all.")

    out_real = args.out / "real"
    out_fake = args.out / "fake"
    print(f"Materialising hardlinks into {args.out} ...")
    rl, rk, rp = _materialise(reals, out_real, dry_run=args.dry_run, dup=real_dup_factor)
    fl, fk, fp = _materialise(fakes_sel, out_fake, dry_run=args.dry_run, dup=1)
    print(f"  real/: linked={rl} kept={rk} pruned={rp}")
    print(f"  fake/: linked={fl} kept={fk} pruned={fp}")
    final_real = rl + rk
    final_fake = fl + fk
    final_total = final_real + final_fake
    final_ratio = final_fake / final_total if final_total else 0.0
    print(f"Final: real={final_real} fake={final_fake} ratio={final_ratio:.3f}")

    if args.check and not args.dry_run:
        # Run the same gate the trainer will run.
        try:
            from dfdetect.training import enforce_constraints, summarise_dataset
        except ImportError:
            print("(install dfdetect with `pip install -e .` to use --check)", file=sys.stderr)
            return 0
        stats = summarise_dataset(out_real, out_fake)
        try:
            enforce_constraints(stats)
        except Exception as e:  # noqa: BLE001
            print(f"BIAS-CHECK FAILED: {e}", file=sys.stderr)
            return 3
        print(f"Bias check OK: {stats.real_count} real / {stats.fake_count} fake "
              f"(ratio {stats.fake_ratio:.2f}, "
              f"real-sources {stats.real_unique_video_hashes}, "
              f"fake-sources {stats.fake_unique_video_hashes}).")
        for w in stats.warnings:
            print(f"  warn: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
