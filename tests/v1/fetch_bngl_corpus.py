#!/usr/bin/env python3
"""Materialize the BNGL test corpus backing ``tests/v1/test_bngl_corpus.py``.

The corpus is 21 third-party ``.bngl`` fixture files.
Rather than vendoring ~3200 lines of model text, this script fetches them on
demand at test/CI time. Sources:

    rulehub       RuleWorld/RuleHub
    bngl_models   wshlavacek/BNGL-Models

Only a tiny manifest (the ``CORPUS`` table below) is committed -- which
model, which pinned commit, which path, its sha256 -- and the real bytes are
fetched at test/CI time. Same idea as lanl/bngsim's
``parity_checks/bng_parity/vendor_corpus.py``.

Usage:
    python tests/v1/fetch_bngl_corpus.py [--dest DIR] [--force] [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly makes the interpreter prepend its own directory
# to sys.path. The `math/` directory then shadows the stdlib `math` module,
# resulting in an ImportError. Drop that entry before importing anything that
# could pull in `math`.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    del sys.path[0]

import argparse  # noqa: E402 -- must follow the sys.path fix above
import hashlib  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

JSDELIVR_GH = "https://cdn.jsdelivr.net/gh/{repo}@{sha}/{path}"

# GitHub repo pins
_RULEHUB = ("RuleWorld/RuleHub", "479d6d62a175572f28b2f6b6d7a376b6d1132da2")
_BNGL_MODELS = (
    "wshlavacek/BNGL-Models",
    "81c90d8f58354a925651859c94142149afefc4b1",
)

# (dest filename, (repo, sha), upstream path, expected sha256 of the RAW
# upstream bytes, optional (find, replace, expected_count) repair applied
# after the hash check). Transcribed from
# tests/v1/bngl_corpus/README.md's provenance table.
CORPUS = [
    (
        "An_2009.bngl",
        _RULEHUB,
        "Published/An2009/An_2009.bngl",
        "1b48efd6fc190988faa5eb28fed78af492f49f8fdf67f064436fbf0f14b634e9",
        None,
    ),
    (
        "Barua_2009__PATCHED.bngl",
        _RULEHUB,
        "Published/Barua2009/Barua_2009.bngl",
        "26ca5053c4a340b597b2d839edd736469fc14cc2a03cf96b0447b7537089c454",
        ("atoll=>", "atol=>", 1),
    ),
    (
        "Chattaraj_2021.bngl",
        _RULEHUB,
        "Published/Chattaraj2021/Chattaraj_2021.bngl",
        "86d43fdf6d5490acb0aa8ba473d92822b18b4af6f2f28a00a3b2ed91505399ed",
        None,
    ),
    (
        "LR.bngl",
        _RULEHUB,
        "Tutorials/NativeTutorials/LR/LR.bngl",
        "e1c72a8f760110a354e0e3e4d41c1fbba504d4d6888e4df2ae91413146d529cb",
        None,
    ),
    (
        "LRR.bngl",
        _RULEHUB,
        "Tutorials/NativeTutorials/LRR/LRR.bngl",
        "b67d551f6d77a74414c2afc9c390b650ccf77d51cebfee45cba9d6ba91f255f3",
        None,
    ),
    (
        "Motivating_example_cBNGL.bngl",
        _RULEHUB,
        "Tutorials/MotivatingexamplecBNGL/Motivating_example_cBNGL.bngl",
        "b3086d656e089a57bbf6d8f31e5ddb6b1cdade598b3bc062a31e2af99ea7482e",
        None,
    ),
    (
        "Ras_bistability_v2.bngl",
        _BNGL_MODELS,
        "my_models/ode/Ras_bistability_v2.bngl",
        "de36f9a416dcc42189d848c32ba3600f07952b375b2cb7248546a11b7b80f731",
        None,
    ),
    (
        "Rule_based_egfr_tutorial.bngl",
        _RULEHUB,
        "Published/Rulebasedegfrtutorial/Rule_based_egfr_tutorial.bngl",
        "56a51fed773324a063c7a1b5e33b0426c1e9d70ab726d20f1b02aa5798fe747a",
        None,
    ),
    (
        "akt-signaling.bngl",
        _RULEHUB,
        "Examples/biology/aktsignaling/akt-signaling.bngl",
        "c89be2edd80d4406571b770c2f1ac39e37e5a2a84a843a5d06390e2ca0070f4f",
        None,
    ),
    (
        "apoptosis-cascade.bngl",
        _RULEHUB,
        "Examples/biology/apoptosiscascade/apoptosis-cascade.bngl",
        "4a8c3bc5248a3245cbce35dc274f4f8f9906f8389da352b0f994e739fc1be050",
        None,
    ),
    (
        "bcr-signaling.bngl",
        _RULEHUB,
        "Examples/biology/bcrsignaling/bcr-signaling.bngl",
        "c07947fbe462dba01d0b44ccc77b4e91d3d1bf5a90371944beb4ef96c5ccc50e",
        None,
    ),
    (
        "blood-coagulation-thrombin.bngl",
        _RULEHUB,
        "Examples/biology/bloodcoagulationthrombin/blood-coagulation-thrombin.bngl",
        "a0ba1f78f4e48592c94d45aff44305370d9a0ca597301ffe593400ad5f6d8c94",
        None,
    ),
    (
        "bmp-signaling.bngl",
        _RULEHUB,
        "Examples/biology/bmpsignaling/bmp-signaling.bngl",
        "fdacb14cddae7fdbdae472800602e03a55642adc92e6a51fc4636867236a2a61",
        None,
    ),
    (
        "brusselator-oscillator.bngl",
        _RULEHUB,
        "Examples/biology/brusselatoroscillator/brusselator-oscillator.bngl",
        "2c86990fc8e36a4a814cdb48a076fae4a359de5b5ffe8dfead15980c515aef67",
        None,
    ),
    (
        "catalysis.bngl",
        _BNGL_MODELS,
        "my_models/ode/catalysis.bngl",
        "7910543d941a1de6b4e8010014a2ee0bf4bc6660665c54bafc56432c5a458256",
        None,
    ),
    (
        "egg.bngl",
        _RULEHUB,
        "Published/Hlavacek2018Egg/egg.bngl",
        "603abf5da09d5eab5ec40816ba558d4b963a0b7faebd676398790980ded8b9ab",
        None,
    ),
    (
        "elephant_EFA.bngl",
        _RULEHUB,
        "Published/Hlavacek2018Elephant/elephant_EFA.bngl",
        "79846c26b8250d6b470a4481d02a2a5d2bebc9fc56c72450f7af7659e7dfc07c",
        None,
    ),
    (
        "energy_transport_pump.bngl",
        _RULEHUB,
        "Examples/energy/energytransportpump/energy_transport_pump.bngl",
        "c5ec8034bf0286942c7cc63c1382d49ec11b6b6cb8366b55e4b838e98a9b2ea3",
        None,
    ),
    (
        "example1.bngl",
        _RULEHUB,
        "Tutorials/example1/example1.bngl",
        "e9aa128804069907a7b8e8b64c498de88d9339cbc312dd5c8abc0cb40a089d64",
        None,
    ),
    (
        "genetic_bistability_energy.bngl",
        _RULEHUB,
        "Examples/genetics/geneticbistabilityenergy/genetic_bistability_energy.bngl",
        "c1b41ff63d7b3209cb30157f7e8be0f34b32afd8ebbd779165fcecffbfc6d59e",
        None,
    ),
    (
        "immob_equiv_lig_sites.bngl",
        _BNGL_MODELS,
        "my_models/nf/immob_equiv_lig_sites.bngl",
        "3ec97bac71f19cc65b8762dc9d6baae091b725b528366a80718d17aeab3221e3",
        None,
    ),
]


class CorpusError(Exception):
    pass


def fetch_one(
    dest_name, repo_pin, upstream_path, expected_sha256, repair, *, dry_run
):
    """Fetch, verify, and (optionally) repair one corpus file.

    Returns the final bytes.
    """
    repo, sha = repo_pin
    url = JSDELIVR_GH.format(repo=repo, sha=sha, path=upstream_path)
    if dry_run:
        print(f"  [dry-run] would fetch {url}")
        return None

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 -- pinned https CDN URL
            raw = resp.read()
    except urllib.error.URLError as exc:
        raise CorpusError(
            f"fetch failed for {dest_name} ({url}): {exc}"
        ) from exc

    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise CorpusError(
            f"sha256 mismatch for {dest_name}: expected {expected_sha256}, "
            f"got {digest} -- upstream {repo}@{sha[:12]}:{upstream_path} "
            "changed or the fetch was tampered with; re-verify before "
            "trusting this content."
        )

    text = raw.decode("utf-8", errors="replace")
    if repair is not None:
        find, replace, expected_count = repair
        n = text.count(find)
        if n != expected_count:
            raise CorpusError(
                f"repair for {dest_name} expected {expected_count}x "
                f"{find!r} but found {n} -- upstream changed; re-verify "
                "the repair before applying it"
            )
        text = text.replace(find, replace)

    return text.encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).parent / "bngl_corpus",
        help="directory to write fetched .bngl files into (default: "
        "bngl_corpus/ next to this script, i.e. tests/v1/bngl_corpus)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-fetch even if the file already exists",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be fetched; write nothing",
    )
    args = ap.parse_args()

    n_fetched = n_skipped = n_failed = 0
    failures = []

    for dest_name, repo_pin, upstream_path, expected_sha256, repair in CORPUS:
        dest_path = args.dest / dest_name
        if dest_path.exists() and not args.force and not args.dry_run:
            print(
                f"skip     {dest_name}  "
                "(already present; use --force to re-fetch)"
            )
            n_skipped += 1
            continue

        print(
            f"fetch    {dest_name}  <- "
            f"{repo_pin[0]}@{repo_pin[1][:12]}:{upstream_path}"
        )
        try:
            content = fetch_one(
                dest_name,
                repo_pin,
                upstream_path,
                expected_sha256,
                repair,
                dry_run=args.dry_run,
            )
        except CorpusError as exc:
            print(f"  FAILED: {exc}")
            failures.append(str(exc))
            n_failed += 1
            continue

        if args.dry_run:
            continue

        args.dest.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(content)
        print(f"  wrote {len(content)} bytes, sha256 verified")
        n_fetched += 1

    print(
        f"\nfetched {n_fetched}, skipped {n_skipped}, failed {n_failed} "
        f"(of {len(CORPUS)} total)"
    )
    if failures:
        print("\nfailures:")
        for f in failures:
            print(f"  - {f}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
