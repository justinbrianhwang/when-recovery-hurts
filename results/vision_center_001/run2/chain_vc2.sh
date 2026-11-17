#!/usr/bin/env bash
# Vision-CENTER run2 (one approved retry): thread limit, install, thread-limit check on stored frames, then
# env -> scenes -> audit -> shadow (B1) -> pilot (B2) -> pack. Stops at the first failure; pack always runs.
# Needs T0 (instance creation, epoch seconds), USD_PER_HOUR, PRIOR_USD (0: the cap of this run counts this run only).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
W=/workspace/robot_harm_intervention_starter_v0_3
mkdir -p $W && cd $W && tar xzf /root/code_vc2.tgz && chmod +x scripts/*.sh scripts/*/*.sh && mkdir -p logs artifacts/ledger
ST=logs/vision_center_001.status
echo "RUN2 threads OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS BLIS=$BLIS_NUM_THREADS nproc=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg)" >> $ST
echo "install start=$(date -u +%FT%TZ)" >> $ST
bash scripts/00_system_packages.sh > logs/vc_install_sys.log 2>&1; rc=$?; echo "install_sys exit=$rc end=$(date -u +%FT%TZ)" >> $ST
[[ $rc -eq 0 ]] || { echo "PHASE_EXIT install 1" >> $ST; exit 1; }
bash scripts/01_bootstrap_vast.sh > logs/vc_bootstrap.log 2>&1; rc=$?; echo "bootstrap exit=$rc end=$(date -u +%FT%TZ)" >> $ST
[[ $rc -eq 0 ]] || { echo "PHASE_EXIT install 1" >> $ST; exit 1; }
echo "PHASE_EXIT install 0" >> $ST
ok=1
# thread-limit check: unchanged perception code on stored frames against the existing outputs; a difference stops everything before B1
( source .venv/bin/activate && python /root/vc_precheck.py check /root/vc_ref.json runs/vision_center_001/thread_limit_check.json ) > logs/vc_thread_check.log 2>&1; rc=$?
echo "thread_check exit=$rc end=$(date -u +%FT%TZ)" >> $ST; echo "PHASE_EXIT thread_check $rc" >> $ST; [[ $rc -eq 0 ]] || { ok=0; echo "THREAD CHECK FAILED: TECHNICAL_BLOCKED before B1" >> $ST; }
for ph in env scenes audit shadow pilot; do
  if [[ $ok -eq 1 ]]; then
    if [[ $ph == shadow || $ph == pilot ]]; then
      # spent so far = hourly price x time since the instance was created, plus a fixed US$0.05 for transfer, plus PRIOR_USD; inputs are time and money only
      export SPENT_USD=$(python3 -c "import time;print(round((time.time()-$T0)/3600*$USD_PER_HOUR+0.05+$PRIOR_USD,4))")
      echo "$ph SPENT_USD=$SPENT_USD USD_PER_HOUR=$USD_PER_HOUR load=$(cut -d' ' -f1-3 /proc/loadavg)" >> $ST
    fi
    bash scripts/vision_center/run.sh $ph; rc=$?; echo "PHASE_EXIT $ph $rc" >> $ST; [[ $rc -eq 0 ]] || ok=0
  else echo "PHASE_SKIPPED $ph (an earlier phase did not pass)" >> $ST; fi
done
echo "CHAIN_END_BEFORE_PACK $(date -u +%FT%TZ)" >> $ST
bash scripts/vision_center/run.sh pack > /root/vc_pack.out 2>&1; echo "PHASE_EXIT pack $?" >> $ST
echo "CHAIN_END $(date -u +%FT%TZ)" >> $ST
