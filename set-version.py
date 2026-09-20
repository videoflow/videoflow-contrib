#!/usr/bin/env python3
"""Set, or check, the one version every contrib sub-package shares.

All sub-packages are released in lockstep with the core ``videoflow`` release: contrib vX.Y.Z
ships with core vX.Y.Z, and each package pins ``videoflow>=X.Y.Z``. The version is spelled
out in four places per component, none of which reads the others:

    <component>/pyproject.toml    version = "X.Y.Z"
                                  "videoflow>=X.Y.Z"           (the core floor)
    <component>/component.yaml    metadata.version: "X.Y.Z"
                                  spec.runtime.images.{cpu,gpu}: ghcr.io/videoflow/contrib-<name>:X.Y.Z[-cuda]

This script is the only thing that should edit those strings. It rewrites lines in place with
regexes rather than parsing and re-dumping TOML/YAML, so comments and formatting survive.

    ./set-version.py 1.2.0               rewrite every file to 1.2.0 (refuses to go backwards)
    ./set-version.py --check [1.2.0]     exit 1 if any file disagrees (with the others, or with the argument)
    ./set-version.py --current           print the version every file agrees on

Normally run by .github/workflows/release.yml; CI runs ``--check`` on every push so a new
component copied in at a stale version fails early.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# X.Y.Z with an optional PEP 440 pre-release suffix (1.2.0rc1). No dashes: the image tag
# regex below relies on '-' meaning only the '-cuda' suffix.
VERSION_RE = re.compile(r'^\d+\.\d+\.\d+([a-z]+\d+)?$')

# (glob, pattern, replacement template, min matches, max matches). Group 'v' is the version.
# Every pattern must match at least once per file: a zero-match file is a component that
# drifted from the layout, not one with nothing to update.
RULES : list[tuple[str, re.Pattern[str], str, int, int]] = [
    ('*/pyproject.toml',
     re.compile(r'^version = "(?P<v>[^"]+)"$', re.MULTILINE),
     'version = "{v}"', 1, 1),
    ('*/pyproject.toml',
     re.compile(r'"videoflow>=(?P<v>[^"]+)"'),
     '"videoflow>={v}"', 1, 1),
    # Exactly two spaces of indent: metadata.version, not a nested key elsewhere.
    ('*/component.yaml',
     re.compile(r'^  version: "(?P<v>[^"]+)"$', re.MULTILINE),
     '  version: "{v}"', 1, 1),
    # Anchored on the registry path so spec.resources.gpu: (a mapping key) is never touched.
    ('*/component.yaml',
     re.compile(r'^(?P<pre>\s+(?:cpu|gpu): ghcr\.io/videoflow/contrib-[a-z0-9-]+:)(?P<v>[^\s-]+)(?P<suffix>-cuda)?$',
                re.MULTILINE),
     '{pre}{v}{suffix}', 1, 2),
]


def _files(glob : str) -> list[Path]:
    # Depth-1 globs: component dirs only, never solutions/ (which have no pyproject).
    return sorted(ROOT.glob(glob))


def collect() -> dict[str, list[tuple[Path, int]]]:
    '''Every version string found, keyed by version -> [(file, line)]. Raises on a file that
    matches a rule too few or too many times.'''
    found : dict[str, list[tuple[Path, int]]] = {}
    problems : list[str] = []
    for glob, pattern, _, lo, hi in RULES:
        for path in _files(glob):
            text = path.read_text()
            matches = list(pattern.finditer(text))
            if not lo <= len(matches) <= hi:
                problems.append(f'{path.relative_to(ROOT)}: expected {lo}-{hi} matches of '
                                f'{pattern.pattern!r}, found {len(matches)}')
                continue
            for m in matches:
                line = text.count('\n', 0, m.start()) + 1
                found.setdefault(m.group('v'), []).append((path, line))
    if problems:
        raise SystemExit('set-version: layout drift:\n  ' + '\n  '.join(problems))
    if not found:
        raise SystemExit('set-version: no version strings found — run from the repo root?')
    return found


def current() -> str:
    found = collect()
    if len(found) == 1:
        return next(iter(found))
    lines = []
    for version, places in sorted(found.items()):
        for path, line in places:
            lines.append(f'  {path.relative_to(ROOT)}:{line}: {version}')
    raise SystemExit('set-version: sub-packages disagree on the version:\n' + '\n'.join(lines))


def _key(version : str) -> tuple[int, ...]:
    # Compare on the numeric part only; a pre-release suffix has already been validated away.
    return tuple(int(p) for p in version.split('.')[:2] + [re.sub(r'[a-z].*', '', version.split('.')[2])])


def set_version(new : str, allow_downgrade : bool) -> None:
    if not VERSION_RE.match(new):
        raise SystemExit(f'set-version: {new!r} is not X.Y.Z (optionally with a pre-release suffix like rc1)')
    old = current()
    if new == old:
        print(f'set-version: already at {new}; nothing to do')
        return
    if _key(new) < _key(old) and not allow_downgrade:
        raise SystemExit(f'set-version: refusing to move from {old} back to {new} (pass --allow-downgrade)')
    changed : set[Path] = set()
    for glob, pattern, template, _, _ in RULES:
        def rewrite(m : re.Match[str], template : str = template) -> str:
            return template.format(**{**m.groupdict(default=''), 'v': new})
        for path in _files(glob):
            text = path.read_text()
            new_text = pattern.sub(rewrite, text)
            if new_text != text:
                path.write_text(new_text)
                changed.add(path)
    for path in sorted(changed):
        print(f'  {path.relative_to(ROOT)}')
    print(f'set-version: {old} -> {new} ({len(changed)} files)')


def main(argv : list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('version', nargs='?', help='the new version, X.Y.Z')
    group.add_argument('--check', nargs='?', const='', metavar='EXPECTED',
                       help='verify every file agrees (and equals EXPECTED, if given)')
    group.add_argument('--current', action='store_true', help='print the version every file agrees on')
    parser.add_argument('--allow-downgrade', action='store_true',
                        help='permit a version lower than the current one')
    args = parser.parse_args(argv)

    if args.current:
        print(current())
        return 0
    if args.check is not None:
        got = current()
        if args.check and got != args.check:
            print(f'set-version: files are at {got}, expected {args.check}', file=sys.stderr)
            return 1
        print(f'set-version: all sub-packages at {got}')
        return 0
    set_version(args.version, args.allow_downgrade)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
