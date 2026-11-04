#!/bin/zsh
# native grid config for any cache: tools/simone_line.sh cache outdir mincells tag a b tau gamma seed eff ngenes [escape]
cd "$(dirname "$0")/.."
B=${DOSECURVE:-tools/dosecurve/target/release/dosecurve}; C=$1; O=$2; MC=$3; shift 3; t=$1
[[ -f $O/${t}_joint.tsv ]] && exit 0
python3 -I tools/simcache.py $C sim/${O:t}_$t --a $2 --b $3 --tau $4 --gamma $5 --seed $6 --eff $7 --ngenes $8 --escape ${9:-0.1} --minsrc $MC --groups 150 > /dev/null || exit 1
cp sim/${O:t}_$t/truth.tsv $O/${t}_truth.tsv
$B ps sim/${O:t}_$t $O/${t}_song --axis lin --bins rank --min-cells $MC > /dev/null 2>&1
$B mixscale sim/${O:t}_$t $O/${t}_mix --min-cells $MC > /dev/null 2>&1
DOSE_BOOT=20 $B joint sim/${O:t}_$t $O/${t}_joint --min-cells $MC --threads 4 2>&1 | grep -o "classified.*" | sed "s/^/$t /"
rm -rf sim/${O:t}_$t
