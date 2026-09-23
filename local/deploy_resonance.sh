#!/bin/bash
# Deploy the resonance probe modules to the printer host, correctly.
#
# Exists because doing this by hand caused three MCU shutdowns in one session:
#   - py_compile does NOT catch scope errors.  An edit that re-parents part of
#     a method into a new function compiles fine and dies at runtime, mid-probe.
#   - FIRMWARE_RESTART clears an MCU shutdown but does NOT reload Python, so a
#     "fixed" module can sit on disk while the old one keeps running - and the
#     traceback line numbers still point at the old file, which is maximally
#     confusing.
#   - A service restart reloads Python but CANNOT recover a shut-down MCU
#     ("Can not update MCU config as it is shutdown").
# After a crash caused by a code bug you need BOTH, in this order.
#
# DEPLOYS A COMMIT, NOT THE WORKING TREE.  The files are taken from `git archive
# HEAD`, so whatever reaches the printer is reconstructible later from its SHA
# alone.  That matters because the trace corpus outlives the code that produced
# it: on 2026-08-07 a trace could not be attributed to a detector version,
# because the only record of what had been deployed was an md5 of a working tree
# that no longer existed anywhere.  A SHA is a state you can check out; a hash of
# uncommitted files is not.
#
# Consequence: an uncommitted change CANNOT be deployed.  That is the point -
# commit first.  ALLOW_DIRTY=1 overrides for a genuine emergency, and then the
# deploy is unreconstructible, which is why it says so loudly.
set -euo pipefail
HOST="${HOST:-who@zer0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODULES="resonance_probe.py resonance_probe_calibrate.py analog_contact.py"
TESTS="test_resonance_moving_stats.py test_analog_contact.py test_verify_ramp.py test_air_guard.py test_verify_verdict.py test_rank_pairs.py"
DEPLOY_LOG="$ROOT/local/deploy_history.log"

echo "== commit check =="
DIRTY=$(git -C "$ROOT" status --porcelain -- klippy/extras docs)   # excludes the untracked self-checks
if [ -n "$DIRTY" ]; then
    if [ "${ALLOW_DIRTY:-0}" = "1" ]; then
        echo "   WARNING: deploying with uncommitted changes in the tree."
        echo "   The archive still comes from HEAD, so these are NOT deployed:"
        echo "$DIRTY" | sed 's/^/     /'
    else
        echo "   Uncommitted changes under klippy/extras or docs:"
        echo "$DIRTY" | sed 's/^/     /'
        echo
        echo "   Commit them first - the deploy takes its modules from HEAD, so"
        echo "   these would NOT reach the printer, and the deployed state would"
        echo "   have no SHA to reconstruct it from."
        echo "   (The self-checks are exempt: they are deliberately untracked,"
        echo "   ship from the working tree by scp, and are md5-verified.)"
        echo "   Override with ALLOW_DIRTY=1 only if you accept that."
        exit 1
    fi
fi
SHA=$(git -C "$ROOT" rev-parse HEAD)
SHORT=$(git -C "$ROOT" rev-parse --short HEAD)
BRANCH=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)
echo "   deploying $BRANCH @ $SHORT"

# Extract HEAD to a scratch dir and deploy THAT, so the bytes that land on the
# printer are provably the commit's and not the working tree's.
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
git -C "$ROOT" archive "$SHA" klippy/extras | tar -x -C "$STAGE"
SRC="$STAGE/klippy/extras"
for f in $MODULES; do
    [ -f "$SRC/$f" ] || { echo "   $f is not in HEAD - commit it first"; exit 1; }
done
# The self-checks are deliberately NOT tracked (local dev aids, no place in the
# upstream diff), so they cannot come from the archive.  They ship from the
# working tree and are md5-verified on target like everything else - what makes
# them safe to deploy uncommitted is that they gate the deploy rather than run
# on the printer in service.
TSRC="$ROOT/klippy/extras"
for f in $TESTS; do
    [ -f "$TSRC/$f" ] || { echo "   missing self-check $f"; exit 1; }
done

echo "== scope check =="
python3 "$(dirname "$0")/scopecheck.py" $(for m in $MODULES; do echo "$SRC/$m"; done)

echo "== compile =="
for m in $MODULES; do python3 -m py_compile "$SRC/$m"; done

echo "== copy =="
scp -q $(for m in $MODULES; do echo "$SRC/$m"; done) \
       $(for t in $TESTS;   do echo "$TSRC/$t"; done) "$HOST:~/klipper/klippy/extras/"

echo "== unit tests (on target, klippy-env has numpy) =="
# This MUST gate the deploy.  It used to be `ssh ... && echo "tests pass"`,
# which only skipped the echo on failure and then restarted klippy anyway -
# a gate that silently passes everything is worse than no gate.
if ssh "$HOST" 'cd ~/klipper/klippy && for t in extras.test_resonance_moving_stats extras.test_analog_contact extras.test_verify_ramp extras.test_air_guard extras.test_verify_verdict extras.test_rank_pairs; do ~/klippy-env/bin/python -m $t >/dev/null || exit 1; done'; then
  echo "   tests pass"
else
  echo "   TESTS FAILED - not restarting klippy (the running process keeps the"
  echo "   previous code; the new files are already copied, so re-run after a fix)"
  exit 1
fi

echo "== reload python (service restart) =="
ssh "$HOST" 'rm -f ~/klipper/klippy/extras/__pycache__/resonance_probe*.pyc ~/klipper/klippy/extras/__pycache__/analog_contact*.pyc; curl -s -X POST "localhost:7125/machine/services/restart?service=klipper" >/dev/null'
sleep 30

STATE=$(ssh "$HOST" 'curl -s localhost:7125/printer/info' | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["state"])')
if [ "$STATE" != "ready" ]; then
    echo "== state '$STATE' -> FIRMWARE_RESTART to clear the MCU =="
    ssh "$HOST" 'curl -s -m 90 -G localhost:7125/printer/gcode/script --data-urlencode "script=FIRMWARE_RESTART"' >/dev/null
    sleep 28
    STATE=$(ssh "$HOST" 'curl -s localhost:7125/printer/info' | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["state"])')
fi

echo "== verify =="
# Modules against the COMMIT's bytes, self-checks against the working tree's -
# each compared with the source it was actually deployed from.
for m in $MODULES; do
    L=$(md5 -q "$SRC/$m"); R=$(ssh "$HOST" "md5sum ~/klipper/klippy/extras/$m" | cut -d' ' -f1)
    [ "$L" = "$R" ] && echo "   $m ok" || { echo "   $m MISMATCH"; exit 1; }
done
for t in $TESTS; do
    L=$(md5 -q "$TSRC/$t"); R=$(ssh "$HOST" "md5sum ~/klipper/klippy/extras/$t" | cut -d' ' -f1)
    [ "$L" = "$R" ] && echo "   $t ok (untracked)" || { echo "   $t MISMATCH"; exit 1; }
done
UP=$(ssh "$HOST" 'ps -o etimes= -p $(pgrep -f klippy.py | head -1)' | tr -d ' ')
echo "   state=$STATE klippy_uptime=${UP}s"
[ "$STATE" = "ready" ] || { echo "NOT READY"; exit 1; }

# Record what landed, so the deploy log in the wiki is transcribed rather than
# reconstructed from memory - and so a trace can be traced back to its code.
WHEN=$(date '+%Y-%m-%dT%H:%M:%S%z')
{
    printf '%s  %s  %s@%s  host=%s%s\n' "$WHEN" "$SHA" "$BRANCH" "$SHORT" "$HOST" \
        "$([ -n "$DIRTY" ] && echo '  DIRTY-OVERRIDE')"
    printf '    %s\n' "$(git -C "$ROOT" log -1 --pretty=%s "$SHA")"
    # The self-checks have no SHA - they are untracked by design - so their md5
    # is the only record of which version gated this deploy.
    for t in $TESTS; do printf '    selfcheck %s  %s\n' "$(md5 -q "$TSRC/$t")" "$t"; done
} >> "$DEPLOY_LOG"
echo
echo "== deployed =="
echo "   $BRANCH @ $SHORT   $WHEN"
echo "   git checkout $SHORT     # reconstructs exactly this code"
echo "   logged to local/deploy_history.log - copy into the wiki deploy log"
