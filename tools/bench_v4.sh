#!/bin/zsh
# v4 benchmark: tools/bench_v4.sh <cache_dir> <tag> <min_cells>
cd "$(dirname "$0")/.."
C=$1; T=$2; MC=$3; N=1000
B=${DOSECURVE:-tools/dosecurve/target/release/dosecurve}
O=out/bench_v4; mkdir -p $O
run(){ name=$1; shift; /usr/bin/time -l "$@" 2> $O/$name.log; }
run ${T}_song_full $B ps $C $O/${T}_song_full --axis lin --bins rank --null $N --min-cells $MC
run ${T}_mix_full  $B mixscale $C $O/${T}_mix_full --null $N --min-cells $MC
run ${T}_joint_full env DOSE_BOOT=20 $B joint $C $O/${T}_joint_full --null $N --min-cells $MC
for h in 0 1; do
  run ${T}_song_h$h  $B ps $C $O/${T}_song_h$h --axis lin --bins rank --half $h --min-cells $MC
  run ${T}_mix_h$h   $B mixscale $C $O/${T}_mix_h$h --half $h --min-cells $MC
  run ${T}_joint_h$h env DOSE_BOOT=20 $B joint $C $O/${T}_joint_h$h --half $h --min-cells $MC
done
