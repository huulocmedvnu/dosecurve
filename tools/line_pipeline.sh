#!/bin/zsh
# full analysis of one cell line: tools/line_pipeline.sh name cache mincells
cd "$(dirname "$0")/.."
N=$1; C=$2; MC=$3; B=${DOSECURVE:-tools/dosecurve/target/release/dosecurve}; R=out/$N; G=out/simgrid_$N
mkdir -p $R $G
/usr/bin/time -l $B ps $C $R/${N}_song --axis lin --bins rank --min-cells $MC > $R/song.log 2>&1
/usr/bin/time -l $B mixscale $C $R/${N}_mix --min-cells $MC > $R/mix.log 2>&1
/usr/bin/time -l env DOSE_BOOT=20 $B joint $C $R/${N}_joint --min-cells $MC > /dev/null 2> $R/joint.log
python3 -I tools/target_mix.py $C $R/target_mix.tsv all > $R/target_mix.log 2>&1
awk -v c=$C -v g=$G -v m=$MC '{print c, g, m, $0}' configs/line_grid.txt | xargs -P 3 -L 1 tools/simone_line.sh > $G/run.log 2>&1
SIMMED=out/empty SIMEFF=$G python3 -I tools/realmap2.py $R/${N}_joint.tsv $R/${N}_song.genes.tsv $R/${N}_mix.tsv $R/regime_native.tsv > $R/realmap_native.log 2>&1
echo PIPELINE_DONE >> $R/joint.log
