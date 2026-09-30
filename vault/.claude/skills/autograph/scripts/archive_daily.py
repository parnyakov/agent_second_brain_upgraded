#!/usr/bin/env python3
"""
autograph archive_daily — move fully-processed, fully-reflected daily notes
out of the active daily/ folder into daily/archive/YYYY-MM/.

Nothing is ever deleted (git keeps full history regardless); this only
declutters the active daily/ folder. A note is archived ONLY when all hard
gates pass — otherwise it's left alone and reported, never silently moved:

  1. age       — note's date is at least --days days in the past (default 30)
  2. processed — the file carries a `processed:` marker block written by the
                  daily pipeline (see daily-format rule). The block is found
                  ANYWHERE in the file, not only at the very end: entries
                  that arrive after the evening run are appended below it,
                  and some older blocks are not closed with `---`. The
                  pipeline's `<!-- ✓ processed -->` comment also counts.
  3. reflected — that ISO week has a weekly reflection in either of the two
                  formats the vault actually uses:
                  a) personal weekly reflection
                     personal/reflection/<year>/<month>/Неделя N, DD.MM - DD.MM.md
                     (date range in the file name covers the note's date;
                     day may lack a leading zero, comma may be missing)
                  b) system reflection
                     thoughts/reflections/YYYY-WNN-system-reflection.md

If only (b) exists the note archives but is flagged in the report as having
no personal reflection for that week.

On archive, any reference elsewhere in the vault to `daily/YYYY-MM-DD`
(wikilink or literal path, with or without .md) is rewritten to the new
archived path, so provenance links (e.g. thought notes' `source:` field)
don't quietly break.

Commands:
  archive_daily.py run <vault-dir> [--days N] [--dry-run]
"""

import re
import sys
from datetime import date, timedelta
from pathlib import Path

from common import walk_vault, rel_path

DAILY_NAME_RE = re.compile(r'^(\d{4})-(\d{2})-(\d{2})\.md$')
# `---` line followed by `processed: <timestamp>` anywhere in the file
# (entries appended after the evening run sit below the block; some blocks
# were never closed with `---`), or the pipeline's HTML-comment marker.
PROCESSED_MARKER_RE = re.compile(r'(?:^|\n)---[ \t]*\nprocessed:[ \t]*\d{4}-\d{2}-\d{2}\S*')
PROCESSED_COMMENT_RE = re.compile(r'<!--\s*✓\s*processed\b')


def iso_week_reflection_path(vault_dir: Path, d: date) -> Path:
    iso_year, iso_week, _ = d.isocalendar()
    return vault_dir / 'thoughts' / 'reflections' / f'{iso_year}-W{iso_week:02d}-system-reflection.md'


PERSONAL_WEEK_RE = re.compile(
    r'^Неделя\s+(?:\d+\s*,?\s*)?(\d{1,2})\.(\d{1,2})\s*-\s*(\d{1,2})\.(\d{1,2})'
)


def _personal_week_ranges(vault_dir: Path):
    """Yield (start, end, path) for every personal weekly reflection file
    named `Неделя N, D.MM - D.MM.md` under personal/reflection/<year>/."""
    refl_dir = vault_dir / 'personal' / 'reflection'
    if not refl_dir.is_dir():
        return
    for f in refl_dir.rglob('*.md'):
        m = PERSONAL_WEEK_RE.match(f.stem)
        if not m:
            continue
        year = next((int(p) for p in f.relative_to(refl_dir).parts[:-1]
                     if p.isdigit() and len(p) == 4), None)
        if year is None:
            continue
        d1, m1, d2, m2 = (int(x) for x in m.groups())
        try:
            if (m2, d2) < (m1, d1):  # week crosses New Year
                # folder may be named after either the start or the end year
                candidates = [(date(year, m1, d1), date(year + 1, m2, d2)),
                              (date(year - 1, m1, d1), date(year, m2, d2))]
            else:
                candidates = [(date(year, m1, d1), date(year, m2, d2))]
        except ValueError:
            continue
        for start, end in candidates:
            yield start, end, f


def personal_reflection_for(vault_dir: Path, d: date, _cache={}) -> Path | None:
    """Personal weekly reflection file whose date range covers `d`, or None."""
    key = str(vault_dir)
    if key not in _cache:
        _cache[key] = list(_personal_week_ranges(vault_dir))
    for start, end, f in _cache[key]:
        if start <= d <= end:
            return f
    return None


def is_processed(content: str) -> bool:
    return bool(PROCESSED_MARKER_RE.search(content) or PROCESSED_COMMENT_RE.search(content))


def rewrite_references(vault_dir: Path, old_rel: str, new_rel: str) -> int:
    """Rewrite `daily/YYYY-MM-DD` references (wikilink or literal path,
    with/without .md) across the vault to point at the archived path.
    Returns number of files touched."""
    old_stem = old_rel[:-3]  # drop .md
    new_stem = new_rel[:-3]
    touched = 0
    for md in walk_vault(vault_dir):
        try:
            text = md.read_text(errors='replace')
        except Exception:
            continue
        if old_stem not in text:
            continue
        new_text = text.replace(old_rel, new_rel).replace(old_stem, new_stem)
        if new_text != text:
            md.write_text(new_text)
            touched += 1
    return touched


def run(vault_dir: Path, days: int, dry_run: bool) -> None:
    daily_dir = vault_dir / 'daily'
    threshold = date.today() - timedelta(days=days)

    archived, skipped_recent, skipped_unprocessed, skipped_unreflected = [], [], [], []
    flagged_no_personal = []

    for md in sorted(daily_dir.glob('????-??-??.md')):
        m = DAILY_NAME_RE.match(md.name)
        if not m:
            continue
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

        if d > threshold:
            skipped_recent.append(md.name)
            continue

        content = md.read_text(errors='replace')
        if not is_processed(content):
            skipped_unprocessed.append(md.name)
            continue

        refl_path = iso_week_reflection_path(vault_dir, d)
        personal = personal_reflection_for(vault_dir, d)
        if personal is None and not refl_path.exists():
            skipped_unreflected.append((md.name, rel_path(refl_path, vault_dir)))
            continue

        if personal is None:
            flagged_no_personal.append(md.name)

        target_dir = daily_dir / 'archive' / f'{d.year:04d}-{d.month:02d}'
        target = target_dir / md.name
        old_rel = f'daily/{md.name}'
        new_rel = f'daily/archive/{d.year:04d}-{d.month:02d}/{md.name}'

        if dry_run:
            archived.append((md.name, new_rel))
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        md.rename(target)
        touched = rewrite_references(vault_dir, old_rel, new_rel)
        archived.append((md.name, new_rel, touched))

    print(f"archive_daily — vault: {vault_dir}, threshold: {threshold.isoformat()} (days={days}), dry_run={dry_run}")
    print(f"  archived:            {len(archived)}")
    for row in archived:
        if dry_run:
            print(f"    {row[0]} -> {row[1]} (dry-run, not moved)")
        else:
            print(f"    {row[0]} -> {row[1]} ({row[2]} file(s) with references rewritten)")
    if flagged_no_personal:
        print(f"  no personal reflection found for week of (archived anyway, review if needed):")
        for name in flagged_no_personal:
            print(f"    {name}")
    print(f"  skipped (too recent): {len(skipped_recent)}")
    print(f"  skipped (not yet processed by daily pipeline): {len(skipped_unprocessed)}")
    for name in skipped_unprocessed:
        print(f"    {name}")
    print(f"  skipped (week has no weekly reflection):        {len(skipped_unreflected)}")
    for name, refl in skipped_unreflected:
        print(f"    {name} (needs personal/reflection/<year>/<month>/Неделя N, DD.MM - DD.MM.md or {refl})")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help') or args[0] != 'run':
        print(__doc__)
        sys.exit(0 if args and args[0] in ('-h', '--help') else 1)

    vault_dir = Path(args[1]) if len(args) > 1 else None
    if not vault_dir or not vault_dir.is_dir():
        print("Error: vault directory required", file=sys.stderr)
        sys.exit(1)

    days = 30
    dry_run = False
    for a in args[2:]:
        if a == '--dry-run':
            dry_run = True
        elif a.startswith('--days'):
            if '=' in a:
                days = int(a.split('=', 1)[1])
            else:
                idx = args.index(a)
                days = int(args[idx + 1])

    run(vault_dir, days, dry_run)


if __name__ == '__main__':
    main()
