#!/system/bin/sh
# tune-1p5gb-v4.sh - frag-fit policy for a 1.5 GB ARM64 Android target
# (GKI 6.12, 4K pages, mthp_bestfit v4.1)
#
# Strategy: the buddy free-list SHAPE picks the folio order. Faults fill
# existing holes (exact fit) before anything is ever split; orders the
# shape cannot absorb fall to base pages instead of compacting.

THP=/sys/kernel/mm/transparent_hugepage
BF=/proc/sys/vm/mthp_bestfit

set_one() { [ -e "$1" ] && echo "$2" > "$1" && echo "  $1 = $2"; }

echo "== mTHP size policy =="
# Wide size menu ON PURPOSE in v4: frag-fit needs candidate orders to
# match whatever hole sizes exist. The module is the size limiter now;
# sysfs only defines the menu. PMD 2 MB stays off at 1.5 GB.
set_one $THP/hugepages-16kB/enabled   always
set_one $THP/hugepages-32kB/enabled   always
set_one $THP/hugepages-64kB/enabled   always
set_one $THP/hugepages-128kB/enabled  always
set_one $THP/hugepages-256kB/enabled  always
set_one $THP/hugepages-512kB/enabled  never
set_one $THP/hugepages-1024kB/enabled never
set_one $THP/hugepages-2048kB/enabled never
set_one $THP/enabled madvise
set_one $THP/defrag  never    # faults never direct-compact; frag-fit makes this free

echo "== mthp_bestfit v4.0 frag-fit =="
set_one $BF/enabled           1
set_one $BF/frag_fit          1     # shape-driven selection (0 = legacy v3.3 A/B)
set_one $BF/exact_min_blocks  8     # holes needed at an order before it's picked
set_one $BF/split_slack       2     # pass-2 may split blocks at most 2 orders up
set_one $BF/refresh_ms        100   # free-shape cache refresh
set_one $BF/min_pages         8     # VMA ceiling: 8 full folios required
set_one $BF/exec_boost        1
set_one $BF/lifetime_aware    1
set_one $BF/dry_run           0

echo "== done =="
echo "Watch: cat /sys/kernel/debug/mthp_bestfit/stats"
echo "  'Buddy free shape' = what frag-fit is steering into."
echo "  Exact-fit%% high + compact_stall flat in vmstat = working as designed."
