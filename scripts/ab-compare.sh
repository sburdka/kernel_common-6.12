#!/system/bin/sh
# ab-compare.sh - capture and diff the counters that matter for the
# 1.5 GB goals: free memory, deferred splits, compaction.
#
# Usage:
#   ab-compare.sh snap baseline      # on stock boot, after workload
#   ab-compare.sh snap tuned         # on tuned boot, same workload
#   ab-compare.sh diff baseline tuned
#
# Snapshots land in /data/local/tmp/mthp-ab/

D=/data/local/tmp/mthp-ab
mkdir -p $D

MEMINFO_KEYS="MemFree MemAvailable AnonPages AnonHugePages Mapped"
VMSTAT_KEYS="thp_fault_alloc thp_fault_fallback thp_deferred_split_page \
thp_split_page compact_stall compact_fail compact_success \
compact_daemon_wake pgsteal_direct pgscan_direct pgmajfault"

snap() {
	out=$D/$1.txt
	: > $out
	for k in $MEMINFO_KEYS; do
		grep "^$k:" /proc/meminfo | awk -v k=$k '{print "meminfo."k" "$2}' >> $out
	done
	for k in $VMSTAT_KEYS; do
		grep "^$k " /proc/vmstat >> $out
	done
	# per-size mTHP stats (6.12 has split_deferred per size)
	for dir in /sys/kernel/mm/transparent_hugepage/hugepages-*kB; do
		sz=$(basename $dir)
		for st in anon_fault_alloc anon_fault_fallback split split_deferred; do
			f=$dir/stats/$st
			[ -e "$f" ] && echo "$sz.$st $(cat $f)" >> $out
		done
	done
	# bestfit decision counters, if loaded
	for f in /sys/kernel/debug/mthp_bestfit/live /sys/kernel/debug/mthp_bestfit_live; do
		[ -e "$f" ] && sed 's/^/bf./' "$f" | grep -v '^bf.#' >> $out && break
	done
	cp /proc/buddyinfo $D/$1.buddyinfo 2>/dev/null
	echo "snapshot -> $out"
}

diff_ab() {
	awk 'NR==FNR { a[$1]=$2; next }
	     {
		b=$2; d=b-a[$1];
		printf "%-34s %12s -> %-12s  (%+d)\n", $1, a[$1], b, d
	     }' $D/$1.txt $D/$2.txt
	echo ""
	echo "Read: meminfo.* are absolute kB (higher MemFree/MemAvailable = win)."
	echo "      *_deferred_split / split_deferred and compact_* are event"
	echo "      counters since boot - compare deltas over the same workload."
}

case "$1" in
	snap) snap "$2" ;;
	diff) diff_ab "$2" "$3" ;;
	*) echo "usage: $0 snap <name> | diff <a> <b>" ;;
esac
