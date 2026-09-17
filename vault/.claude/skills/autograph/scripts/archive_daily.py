#!/usr/bin/env python3
"""
autograph archive_daily — move fully-processed, fully-reflected daily notes
out of the active daily/ folder into daily/archive/YYYY-MM/.

Nothing is ever deleted (git keeps full history regardless); this only
declutters the active daily/ folder. A note is archived ONLY when all hard
gates pass — otherwise it's left alone and reported, never silently moved:

  1. age       — note's date is at least --days days in the past (default 30)
  2. processed — the file ends with the `processed:` marker block that the
                  daily pipeline writes after CAPTURE/EXECUTE/REFLECT
                  (see .claude/rules/daily-format.md)
  3. reflected — a system weekly reflection file exists for that ISO week
                  (thoughts/reflections/YYYY-WNN-system-reflection.md)

A softer, non-blocking check also looks for a personal weekly reflection
covering that week (personal/reflection/**) — if none is found, the note
still archives (that process is intentionally freeform/optional) but it's called out in the
report so a human can decide whether to do that reflection first.

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
PROCESSED_MARKER_RE = re.compile(r'\n---\s*\nprocessed:\s*\S+.*?\n---\s*$', re.DOTALL)


def iso_week_reflection_path(vault_dir: Path, d: date) -> Path:
    iso_year, iso_week, _ = d.isocalendar()
    return vault_dir / 'thoughts' / 'reflections' / f'{iso_year}-W{iso_week:02d}-system-reflection.md'


def personal_reflection_exists(vault_dir: Path, d: date) -> bool:
    """Best-effort: does any personal/reflection/** file's name mention this
    week's Mon/Sun day.month? Freeform naming, so this is advisory only."""
    refl_dir = vault_dir / 'personal' / 'reflection'
    if not refl_dir.is_dir():
        return False
    iso_year, iso_week, _ = d.isocalendar()
    monday = date.fromisocalendar(iso_year, iso_week, 1)
    sunday = date.fromisocalendar(iso_year, iso_week, 7)
    monday_tag = monday.strftime('%d.%m')
    sunday_tag = sunday.strftime('%d.%m')
    for f in refl_dir.rglob('*.md'):
        name = f.name
        if monday_tag in name or sunday_tag in name:
            return True
    return False


def is_processed(content: str) -> bool:
    return bool(PROCESSED_MARKER_RE.search(content))


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
        if not refl_path.exists():
            skipped_unreflected.append((md.name, rel_path(refl_path, vault_dir)))
            continue

        if not personal_reflection_exists(vault_dir, d):
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
    print(f"  skipped (week not yet system-reflected):        {len(skipped_unreflected)}")
    for name, refl in skipped_unreflected:
        print(f"    {name} (needs {refl})")


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
