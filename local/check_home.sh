#!/bin/sh
# Home, then grab a camera frame at the POST-HOME position for comparison
# against local/home_reference.jpg.
#
# X and Y home sensorlessly on this machine (tmc2209 virtual_endstop,
# driver_sgthrs 100) and INTERMITTENTLY FALSE-TRIGGER: G28 returns success,
# klippy reports homed_axes='xyz', and the toolhead is physically somewhere
# else.  There is no software symptom - no error, no log line, and the reported
# position looks perfect.  The camera is the only check that catches it, and it
# only works at the post-home position BEFORE anything else moves: a frame taken
# after the toolhead has driven off shows nothing comparable.
#
# Durable copies of the reference: local/home_reference.jpg (working) and
# ~/AI Brain/raw/assets/klipper-zer0-home-reference.jpg (synced + backed up).
# The previous reference was lost to non-durable storage - do not "helpfully"
# park a new one in /tmp.
set -eu
HOST="${HOST:-who@zer0}"
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="${1:-$HERE/home_latest.jpg}"

ssh -o ConnectTimeout=10 "$HOST" \
    'curl -s -X POST "http://localhost:7125/printer/gcode/script" \
       -H "Content-Type: application/json" -d "{\"script\": \"G28\"}" -m 180 >/dev/null
     sleep 3
     curl -s -m 15 "http://localhost:8080/?action=snapshot" -o /tmp/posthome.jpg'
scp -q "$HOST:/tmp/posthome.jpg" "$OUT"

# The camera CANNOT tell a fresh home from a home that never happened: a failed
# or refused G28 leaves the toolhead parked exactly where the LAST home left
# it, so the reference frame matches perfectly and the next move dies on "Must
# home axis first".  Cost the 2026-09-21 session a wasted probe run.  So ask
# klippy what it thinks, too.
HOMED=$(ssh -o ConnectTimeout=10 "$HOST" \
    'curl -s "http://localhost:7125/printer/objects/query?toolhead=homed_axes"' \
    2>/dev/null | sed -n 's/.*"homed_axes": *"\([a-z]*\)".*/\1/p')
case "$HOMED" in
  *x*y*z*) echo "homed_axes=$HOMED" ;;
  *) echo "HOME FAILED: klippy reports homed_axes='$HOMED' (want xyz)."
     echo "The frame below may still LOOK right - it is the previous home's"
     echo "position.  Do not run anything until this says xyz."
     exit 1 ;;
esac

echo "post-home frame: $OUT"
echo "compare against: $HERE/home_reference.jpg"
echo "A false home puts the nozzle somewhere VISIBLY different - if the two do"
echo "not match, re-home before running anything."
