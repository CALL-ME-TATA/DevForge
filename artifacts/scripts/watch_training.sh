#!/bin/sh
# Training health watcher: appends to E:/DevForge/training_health.log until 4:30 AM.
#   SUMMARY  every 15 min: iteration, steps, avg speed, entropy, exit rates per map type
#   ALERT    immediately: crash text in the log, container down, no progress for 5 min
LOG=/e/DevForge/repo/training/ckpt/train.log
OUT=/e/DevForge/training_health.log
stop=$(date -d "tomorrow 04:30" +%s)
last_sum=0; alerted_stall=0; alerted_down=0
seen_err=$(grep -c -E "Traceback|Error|Killed|nan |CUDA out of memory" $LOG)
while [ $(date +%s) -lt $stop ]; do
  now=$(date +%s); ts=$(date +%H:%M)
  # crash text
  n=$(grep -c -E "Traceback|Error|Killed|nan |CUDA out of memory" $LOG)
  if [ "$n" -gt "$seen_err" ]; then
    echo "$ts ALERT error in train.log: $(grep -E "Traceback|Error|Killed|nan |CUDA out of memory" $LOG | tail -1 | cut -c1-160)" >> $OUT
    seen_err=$n
  fi
  # container status (between runs at ~1:45 a gap of <1 min is expected)
  if ! timeout 20 docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^doomrun$'; then
    [ $alerted_down -eq 0 ] && echo "$ts ALERT doomrun container is not running" >> $OUT
    alerted_down=1
  else
    [ $alerted_down -eq 1 ] && echo "$ts OK doomrun container is running again" >> $OUT
    alerted_down=0
  fi
  # stall: train.log not written for 5 min while the container is up
  age=$(( now - $(stat -c %Y $LOG) ))
  if [ $age -gt 300 ] && [ $alerted_down -eq 0 ]; then
    [ $alerted_stall -eq 0 ] && echo "$ts ALERT no training progress for $((age/60)) min" >> $OUT
    alerted_stall=1
  else
    alerted_stall=0
  fi
  # periodic summary
  if [ $((now - last_sum)) -ge 900 ]; then
    last=$(grep "^it " $LOG | tail -1)
    it=$(echo "$last" | awk '{print $2}'); st=$(echo "$last" | awk '{print $4}')
    sps=$(grep "^it " $LOG | tail -30 | awk '{s+=$6;n++} END{printf "%.0f", s/n}')
    ent=$(echo "$last" | grep -oE "ent [0-9.]+" | awk '{print $2}')
    ex=$(echo "$last" | sed 's/.*| exit //')
    echo "$ts SUMMARY it $it | $st steps | $sps sps | ent $ent | exit $ex" >> $OUT
    last_sum=$now
  fi
  sleep 30
done
echo "$(date +%H:%M) watcher finished" >> $OUT
