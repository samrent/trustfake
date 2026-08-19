#!/bin/bash
# Wait for the validation job, then pull all 249 train shards and verify everything.
cd /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1
while pgrep -f 'fetch_sid_set.py validation' >/dev/null; do sleep 10; done
echo "=== validation done, starting train $(date -Is) ==="
.venv/bin/python src/fetch_sid_set.py train
echo "=== all done $(date -Is) ==="
