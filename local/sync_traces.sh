#!/bin/sh
# Pull new probe traces from the printer host to this machine for offline
# analysis.  Run it after any probing session - the corpus is the only record
# of a contact that cannot be re-created without wearing the plate again, and
# analysis is repeatedly wanted while the printer is off or busy printing.
#
# Safe to run any time, and safe to run twice:
#   --ignore-existing   never overwrites a trace already here, so a local copy
#                       is authoritative and re-syncing cannot rewrite history
#   no --delete         removing a trace on the host never removes it here
#   *.csv only          LABELS.md lives HERE and is hand-maintained; the host
#                       has no business overwriting it
#
# HOST=who@<ip> ./sync_traces.sh   to bypass MagicDNS (it has failed before;
#                                  the tailnet IP has been 100.123.190.58)
set -e

HOST="${HOST:-who@zer0}"
REMOTE="${REMOTE:-probe_traces/}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
LOCAL="$HERE/probe_traces/"

before=$(find "$LOCAL" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')

if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$HOST" true 2>/dev/null; then
    echo "sync_traces: $HOST is not reachable (printer off, or MagicDNS again)."
    echo "             Local corpus untouched: $before traces."
    echo "             Retry with: HOST=who@100.123.190.58 $0"
    exit 1
fi

echo "== pulling new traces from $HOST:$REMOTE =="
rsync -a --ignore-existing \
      --include='*.csv' --exclude='*' \
      "$HOST:$REMOTE" "$LOCAL"

after=$(find "$LOCAL" -maxdepth 1 -name '*.csv' | wc -l | tr -d ' ')
echo "== traces: $before -> $after (+$((after - before))) =="

# Surface what arrived, since new traces usually mean there is something to
# label in LABELS.md while the run is still fresh in mind.
if [ "$after" -gt "$before" ]; then
    echo "newest:"
    ls -t "$LOCAL"/*.csv | head -"$((after - before))" | while read -r f; do
        printf '  %s' "$(basename "$f")"
        grep -h '^# note=' "$f" 2>/dev/null | sed 's/^# /  /' || echo
        echo
    done
    echo
    echo "If any of these are labelled runs, record them in probe_traces/LABELS.md"
    echo "now - an unlabelled overshoot trace cannot be used for depth later."
fi
