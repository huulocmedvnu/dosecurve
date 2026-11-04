#!/bin/zsh
# tools/gtest_one.sh tag a b tau gamma : Song, Mixscale, joint v4 and anchored joint on one sim
cd "$(dirname "$0")/.."
O=out/gtest; t=$1; B=${DOSECURVE:-tools/dosecurve/target/release/dosecurve}
[[ -f $O/${t}_anch.tsv ]] && exit 0
python3 -I tools/simcache.py cache/k562ess sim/gt_$t --a $2 --b $3 --tau $4 --gamma $5 --seed 2 --groups 150 > /dev/null || exit 1
$B ps sim/gt_$t $O/${t}_song --axis lin --bins rank > /dev/null 2>&1
$B mixscale sim/gt_$t $O/${t}_mix > /dev/null 2>&1
DOSE_BOOT=20 $B joint sim/gt_$t $O/${t}_v4 --threads 4 2>&1 | grep -o "classified.*" | sed "s/^/$t v4 /"
DOSE_BOOT=20 DOSE_ANCHOR=1 $B joint sim/gt_$t $O/${t}_anch --threads 4 2>&1 | grep -o "classified.*" | sed "s/^/$t anchor /"
rm -rf sim/gt_$t
