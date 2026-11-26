#!/bin/sh
# Wait for THIS training container (by ID) to finish, then fine-tune 2 more hours.
# Guard: only relaunch after 01:30, so a manual restart/removal earlier never triggers it.
cd /e/DevForge/repo
cid=$1
code=$(docker wait "$cid" 2>&1)
echo "$(date) container $cid exited with $code"
now=$(date +%s); earliest=$(date -d "01:30" +%s)
[ $earliest -lt $(( now - 43200 )) ] && earliest=$(date -d "tomorrow 01:30" +%s)
if [ $now -lt $earliest ]; then echo "$(date) before 01:30 -> not relaunching (manual restart?)"; exit 0; fi
powershell -ExecutionPolicy Bypass -File training/start_training.ps1 -Hours 2 \
  -Extra "--maps-per-bin 500 --dmg-pen 0.03 --death-pen 5 --cell-rew 0.02 --shape-k 0.1 --hard-floor 0.2 --key-gate 0.3 --weapon-pen 0.05 --shot-pen 0.005 --door-use 0.1 --lr 1.5e-4 --ent 0.005"
echo "$(date) relaunched for 2h fine-tune"
