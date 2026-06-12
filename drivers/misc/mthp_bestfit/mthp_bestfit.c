// SPDX-License-Identifier: GPL-2.0
/*
 * mthp_bestfit v4.1 - fragmentation-shape best-fit mTHP order selection.
 *
 * v4.1 closes the v4.0 review findings (see EXPERT-REVIEW.md):
 *
 *  F1  strict_mask=1: emit BIT(best) only. The 0..best mask made every
 *      retained bit a full allocation attempt in alloc_anon_folio();
 *      with defrag != never each attempt can direct-compact. One large
 *      attempt, then base pages.
 *  F2  no periodic kworker. The free-shape refresh is self-clocked from
 *      the fault path (jiffies gate + cmpxchg election): zero idle
 *      wakeups, refresh rate tracks fault rate.
 *  F3  herd_guard: per-window consumption budgets so a fault storm
 *      cannot stampede one order off a stale snapshot.
 *  OPT decision is O(1): eligibility precomputed at refresh time as two
 *      packed 16-bit bitmaps in ONE 64-bit word (atomically consistent
 *      snapshot by construction); selection is mask + fls. Geometric
 *      ceiling is one fls_long. Stats use this_cpu ops, no preempt
 *      toggling.
 *
 * Target: Android GKI 6.12 (arm64), Clang/LLVM, external module.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/mm_types.h>
#include <linux/huge_mm.h>
#include <linux/sysctl.h>
#include <linux/debugfs.h>
#include <linux/seq_file.h>
#include <linux/percpu.h>
#include <linux/tracepoint.h>
#include <linux/uaccess.h>
#include <linux/sizes.h>
#include <linux/mmzone.h>
#include <linux/minmax.h>
#include <linux/bitops.h>
#include <linux/atomic.h>
#include <linux/jiffies.h>
#include <linux/swap.h>

#include <trace/hooks/mm.h>

#include "profile.h"

#define MOD_NAME	"mthp_bestfit"
#define MOD_VER		"4.3"

MODULE_SOFTDEP("pre: mthp_monitor");

#if PAGE_SIZE == SZ_4K
# define CONTPTE_ORDER 4	/* 64 KB = 16 x 4 KB */
#elif PAGE_SIZE == SZ_16K
# define CONTPTE_ORDER 0
#else
# define CONTPTE_ORDER 0
#endif

/* -- VMA type classification --------------------------------------------- */
#define BF_VMA_STACK	0
#define BF_VMA_EXEC	1	/* see review F5: rarely live on fault path */
#define BF_VMA_ANON	2
#define BF_VMA_FILE	3
#define BF_VMA_TYPES	4

static int bf_vma_type(const struct vm_area_struct *vma)
{
	if (vma->vm_flags & VM_GROWSDOWN)
		return BF_VMA_STACK;
	if ((vma->vm_flags & VM_EXEC) && vma->vm_file)
		return BF_VMA_EXEC;
	if (vma_is_anonymous(vma))
		return BF_VMA_ANON;
	return BF_VMA_FILE;
}

/* -- sysctls -------------------------------------------------------------- */
static int sysctl_mthp_bestfit_enabled		__read_mostly = 1;
static int sysctl_mthp_bestfit_min_pages	__read_mostly = 4;
static int sysctl_mthp_bestfit_contpte_bias	__read_mostly = 0;
static int sysctl_mthp_bestfit_buddy_guard	__read_mostly = 32;
static int sysctl_mthp_bestfit_frag_cap		__read_mostly = 70;
static int sysctl_mthp_bestfit_exec_boost	__read_mostly = 1;
static int sysctl_mthp_bestfit_pressure_aware	__read_mostly = 1;
static int sysctl_mthp_bestfit_pressure_min	__read_mostly = 4;
static int sysctl_mthp_bestfit_dry_run		__read_mostly = 0;
static int sysctl_mthp_bestfit_lifetime_aware	__read_mostly = 1;
static int sysctl_mthp_bestfit_frag_fit		__read_mostly = 1;
static int sysctl_mthp_bestfit_exact_min	__read_mostly = 8;
static int sysctl_mthp_bestfit_split_slack	__read_mostly = 2;
static int sysctl_mthp_bestfit_refresh_ms	__read_mostly = 100;
/* v4.1 */
static int sysctl_mthp_bestfit_strict_mask	__read_mostly = 1;
static int sysctl_mthp_bestfit_herd_guard	__read_mostly = 1;
/* v4.3: PMD eligibility threshold — buddy must have >= pmd_min_blocks
 * free 2 MB blocks before PMD is selected. Higher than exact_min_blocks
 * because each 2 MB block is order-7 times costlier than a 16 KB block. */
static int sysctl_mthp_bestfit_pmd_min		__read_mostly = 16;

static int bf_int_zero		= 0;
static int bf_int_one		= 1;
static int bf_min_pages_max	= 16;
static int bf_buddy_guard_max	= 4096;
static int bf_frag_cap_max	= 100;
static int bf_pressure_min_max	= 64;
static int bf_exact_min_max	= 4096;
static int bf_slack_max		= 4;
static int bf_refresh_ms_min	= 50;
static int bf_refresh_ms_max	= 5000;
static int bf_pmd_min_max	= 65536;

/* -- per-CPU statistics ---------------------------------------------------- */
struct bestfit_counters {
	u64 decisions;
	u64 order_hist[PMD_ORDER + 1];
	u64 contpte_hits;
	u64 capped_down;
	u64 stack_capped;
	u64 exec_boosted;
	u64 frag_capped;
	u64 buddy_guarded;
	u64 pressure_downgraded;
	u64 compact_stall_avoided;
	u64 lifetime_conserved;
	u64 exact_fit;
	u64 split_fit;
	u64 frag_suppressed;
	u64 budget_skips;	/* v4.1: herd guard rejected an exact bit */
	u64 by_type[BF_VMA_TYPES];
};

static DEFINE_PER_CPU(struct bestfit_counters, bestfit_pcpu);

struct bf_flags {
	bool contpte, capped, stack, exec_boosted;
	bool frag_cap, buddy_guard;
	bool pdg, sa, lc;
	bool exact, split_fit, suppressed;
	int  budget_skips;
};

/*
 * OPT: this_cpu ops are individually preempt/IRQ-safe single
 * instructions on arm64 - no preempt_disable()/enable() pair, no
 * cross-field consistency needed for statistics.
 */
static inline void bestfit_count(int order, int vtype,
				 const struct bf_flags *f)
{
	this_cpu_inc(bestfit_pcpu.decisions);
	if (order >= 0 && order <= PMD_ORDER)
		this_cpu_inc(bestfit_pcpu.order_hist[order]);
	if (vtype >= 0 && vtype < BF_VMA_TYPES)
		this_cpu_inc(bestfit_pcpu.by_type[vtype]);
	if (f->contpte)		this_cpu_inc(bestfit_pcpu.contpte_hits);
	if (f->capped)		this_cpu_inc(bestfit_pcpu.capped_down);
	if (f->stack)		this_cpu_inc(bestfit_pcpu.stack_capped);
	if (f->exec_boosted)	this_cpu_inc(bestfit_pcpu.exec_boosted);
	if (f->frag_cap)	this_cpu_inc(bestfit_pcpu.frag_capped);
	if (f->buddy_guard)	this_cpu_inc(bestfit_pcpu.buddy_guarded);
	if (f->pdg)		this_cpu_inc(bestfit_pcpu.pressure_downgraded);
	if (f->sa)		this_cpu_inc(bestfit_pcpu.compact_stall_avoided);
	if (f->lc)		this_cpu_inc(bestfit_pcpu.lifetime_conserved);
	if (f->exact)		this_cpu_inc(bestfit_pcpu.exact_fit);
	if (f->split_fit)	this_cpu_inc(bestfit_pcpu.split_fit);
	if (f->suppressed)	this_cpu_inc(bestfit_pcpu.frag_suppressed);
	if (f->budget_skips)
		this_cpu_add(bestfit_pcpu.budget_skips, f->budget_skips);
}

/* -- mthp_monitor optional integration (frag_index only) ------------------ */
static int (*monitor_get_fn)(struct system_info *);
static struct system_info g_frag_cache;

/* -- free-shape state -------------------------------------------------------
 *
 * Hot state is ONE 64-bit word:
 *   bits  0..15  exact-fit eligibility bitmap   (free[k] >= exact_min)
 *   bits 32..47  bounded-fit eligibility bitmap (sum window >= exact_min)
 * Published with a single WRITE_ONCE, read with a single READ_ONCE: the
 * fault path always sees an internally consistent snapshot, no seqlock.
 *
 * bf_free_raw[] (u32, cold) feeds debugfs and the legacy v3.3 path.
 * bf_budget[] are the per-window herd-guard consumption budgets.
 */
static u64 bf_eligible;
static u32 bf_free_raw[PMD_ORDER + 1];
static atomic_t bf_budget[PMD_ORDER + 1];
static unsigned long bf_next_refresh;
static atomic_t bf_refresh_busy = ATOMIC_INIT(0);

/*
 * Racy per-order snapshot over NODE_DATA(0) zones. nr_free sums all
 * migratetypes (CMA/HIGHATOMIC inflate it) and excludes per-CPU lists
 * (orders <= COSTLY undercount) - see review F4. Over-counts are
 * absorbed by the allocator fallback; under-counts cost hit rate only.
 */
static unsigned long bf_order_free_count(int order)
{
	unsigned long total = 0;
	int z;

	for (z = 0; z < MAX_NR_ZONES; z++) {
		struct zone *zone = &NODE_DATA(0)->node_zones[z];

		if (populated_zone(zone))
			total += READ_ONCE(zone->free_area[order].nr_free);
	}
	return total;
}

static void bf_recompute_shape(void)
{
	u32 raw[PMD_ORDER + 1];
	u32 exact = 0, bounded = 0;
	u32 want = (u32)READ_ONCE(sysctl_mthp_bestfit_exact_min);
	int slack = READ_ONCE(sysctl_mthp_bestfit_split_slack);
	int k, j, top;
	u64 sum;

	for (k = 0; k <= PMD_ORDER; k++) {
		raw[k] = (u32)min(bf_order_free_count(k),
				  (unsigned long)U32_MAX);
		WRITE_ONCE(bf_free_raw[k], raw[k]);
		atomic_set(&bf_budget[k], raw[k]);
	}

	for (k = 2; k < PMD_ORDER; k++) {
		if (raw[k] >= want)
			exact |= BIT(k);
		top = min(k + slack, PMD_ORDER - 1);
		sum = 0;
		for (j = k; j <= top; j++)
			sum += raw[j];
		if (sum >= want)
			bounded |= BIT(k);
	}

	/* PMD: exact-fit only (no bounded-fit — there is nothing larger to
	 * split down from). Requires pmd_min_blocks free 2 MB buddy blocks. */
	if (raw[PMD_ORDER] >= (u32)READ_ONCE(sysctl_mthp_bestfit_pmd_min))
		exact |= BIT(PMD_ORDER);

	WRITE_ONCE(bf_eligible, ((u64)bounded << 32) | exact);

	if (monitor_get_fn)
		monitor_get_fn(&g_frag_cache);
}

/*
 * F2: self-clocked refresh. Called at hook entry; one jiffies compare
 * when fresh. When stale, one CPU wins the cmpxchg and pays the ~40
 * lockless loads inline (sub-microsecond, safe under per-VMA lock - no
 * sleeping, READ_ONCE only). No faults -> no work -> zero idle wakeups.
 * Fault storm -> refresh tracks fault rate -> staleness window shrinks
 * exactly when the herd is largest.
 */
static void bf_maybe_refresh(void)
{
	if (time_before(jiffies, READ_ONCE(bf_next_refresh)))
		return;
	if (atomic_cmpxchg(&bf_refresh_busy, 0, 1) != 0)
		return;
	if (time_after_eq(jiffies, READ_ONCE(bf_next_refresh))) {
		bf_recompute_shape();
		WRITE_ONCE(bf_next_refresh, jiffies + msecs_to_jiffies(
			READ_ONCE(sysctl_mthp_bestfit_refresh_ms)));
	}
	atomic_set(&bf_refresh_busy, 0);
}

/* -- v4 core: O(1) fragmentation-shape best fit -----------------------------
 *
 * One 64-bit load, two fls(). The refresh path did the O(orders*slack)
 * work; the buddy's segregated lists did the sort long before that.
 */
static int bf_fragfit_order(int ceiling, unsigned long allowed,
			    struct bf_flags *f)
{
	u64 elig = READ_ONCE(bf_eligible);
	u32 cmask = (u32)(allowed & (BIT(ceiling + 1) - 1)) & ~3u;
	u32 exact = (u32)elig & cmask;
	u32 bounded = (u32)(elig >> 32) & cmask;
	int k;

	if (READ_ONCE(sysctl_mthp_bestfit_herd_guard)) {
		/* F3: spend the window budget; skip exhausted orders. */
		while (exact) {
			k = fls(exact) - 1;
			if (atomic_dec_if_positive(&bf_budget[k]) >= 0) {
				f->exact = true;
				return k;
			}
			f->budget_skips++;
			exact &= ~BIT(k);
		}
	} else if (exact) {
		f->exact = true;
		return fls(exact) - 1;
	}

	if (bounded) {
		f->split_fit = true;
		return fls(bounded) - 1;
	}

	f->suppressed = true;
	return 0;
}

/* -- v3.3 legacy pressure gate (frag_fit=0 A/B path) ----------------------- */
static int bf_pressure_limit(int order, int geo_order, bool *stall_avoided)
{
	unsigned long free_count;
	int min_free;

	*stall_avoided = false;
	if (!READ_ONCE(sysctl_mthp_bestfit_pressure_aware))
		return order;

	min_free = READ_ONCE(sysctl_mthp_bestfit_pressure_min);

	free_count = READ_ONCE(bf_free_raw[order]);
	if (free_count >= (unsigned long)min_free)
		return order;

	if (order != geo_order) {
		if (free_count > 0)
			*stall_avoided = true;
		free_count = READ_ONCE(bf_free_raw[geo_order]);
		if (free_count >= (unsigned long)min_free)
			return geo_order;
	}

	if (free_count > 0)
		*stall_avoided = true;
	return -1;
}

/* -- decision pipeline ------------------------------------------------------ */
static int mthp_bestfit_order(struct vm_area_struct *vma,
			      unsigned long allowed,
			      int *vtype_out, struct bf_flags *f)
{
	unsigned long vma_size, min_pages;
	int order, geo_order, ceiling, vtype;

	memset(f, 0, sizeof(*f));
	*vtype_out = BF_VMA_ANON;

	if (!READ_ONCE(sysctl_mthp_bestfit_enabled))
		return -1;

	vma_size  = vma->vm_end - vma->vm_start;
	min_pages = (unsigned long)READ_ONCE(sysctl_mthp_bestfit_min_pages);
	if (min_pages < 1)
		min_pages = 1;

	vtype      = bf_vma_type(vma);
	*vtype_out = vtype;

	/* Stack cap */
	if (vma->vm_flags & VM_GROWSDOWN) {
		f->stack = true;
		if (vma_size >= (min_pages << (PAGE_SHIFT + 2)))
			return 2;
		return 0;
	}

	/* Reject VMAs too small for order 2 (16 KB minimum). */
	if (vma_size < (min_pages << (PAGE_SHIFT + 2)))
		return 0;

	/*
	 * OPT: O(1) geometric ceiling. Largest geo with
	 *   vma_size >= min_pages << (PAGE_SHIFT + geo)
	 * <=> geo = ilog2(vma_size / min_pages) - PAGE_SHIFT.
	 * The size reject above guarantees geo >= 2.
	 */
	geo_order = fls_long(vma_size / min_pages) - 1 - PAGE_SHIFT;
	if (geo_order > PMD_ORDER)
		geo_order = PMD_ORDER;

	ceiling = geo_order;

	/* Exec boost: ceiling +1. See review F5 - verify by_type[exec]
	 * before trusting this branch; it is likely dead at fault time. */
	if ((vma->vm_flags & VM_EXEC) && vma->vm_file &&
	    READ_ONCE(sysctl_mthp_bestfit_exec_boost) &&
	    ceiling < PMD_ORDER - 1)
		ceiling++;

	/* Lifetime: first-fault demotion (anon_vma not yet instantiated).
	 * Honest scope per review F6: one folio per VMA lifetime. */
	if (READ_ONCE(sysctl_mthp_bestfit_lifetime_aware) &&
	    !vma->anon_vma && ceiling > 2) {
		ceiling--;
		f->lc = true;
	}

	if (READ_ONCE(sysctl_mthp_bestfit_frag_fit)) {
		order = bf_fragfit_order(ceiling, allowed, f);
		f->exec_boosted = (order == ceiling && ceiling > geo_order);
		f->sa = f->suppressed;
	} else {
		order   = geo_order;
		f->capped = (order < PMD_ORDER);

		if (READ_ONCE(sysctl_mthp_bestfit_contpte_bias) &&
		    CONTPTE_ORDER > 0 &&
		    order > 0 && order < CONTPTE_ORDER &&
		    vma_size >= (1UL << (PAGE_SHIFT + CONTPTE_ORDER))) {
			order      = CONTPTE_ORDER;
			f->contpte = true;
		}

		if ((vma->vm_flags & VM_EXEC) && vma->vm_file &&
		    READ_ONCE(sysctl_mthp_bestfit_exec_boost) &&
		    order < PMD_ORDER - 1) {
			order++;
			f->exec_boosted = true;
		}

		{
			int limited = bf_pressure_limit(order, geo_order,
							&f->sa);

			if (limited < 0) {
				if (!READ_ONCE(sysctl_mthp_bestfit_dry_run))
					return 0;
			} else if (limited < order) {
				f->pdg = true;
				order  = limited;
			}
		}

		if (monitor_get_fn && order > 0) {
			u32 frag     = READ_ONCE(g_frag_cache.frag_index);
			int frag_cap = READ_ONCE(sysctl_mthp_bestfit_frag_cap);

			if (CONTPTE_ORDER > 0 && (int)frag > frag_cap &&
			    order > CONTPTE_ORDER) {
				order = CONTPTE_ORDER;
				f->frag_cap = true;
			}
		}

		if (order > 2 && order < PMD_ORDER) {
			int guard = READ_ONCE(sysctl_mthp_bestfit_buddy_guard);
			u32 free_at = READ_ONCE(bf_free_raw[order]);

			if ((int)free_at < guard) {
				order--;
				f->buddy_guard = true;
			}
		}
	}

	f->capped  = (order < geo_order) || (order < PMD_ORDER);
	f->contpte = (CONTPTE_ORDER > 0 && order == CONTPTE_ORDER);

	return order;
}

static unsigned long __mthp_bestfit_mask(struct vm_area_struct *vma,
					 unsigned long allowed_orders)
{
	struct bf_flags f;
	int best, vtype;

	if (!READ_ONCE(sysctl_mthp_bestfit_enabled))
		return allowed_orders;

	/* PMD bypass: see v3.3 rationale (4x compact_stall fallthrough). */
	if (!(allowed_orders & (BIT(PMD_ORDER) - 1)))
		return allowed_orders;

	/* MemAvailable pressure gate: under memory pressure, suppress 2 MB
	 * folios specifically (strip PMD from the candidate set) so sub-PMD
	 * frag-fit still runs. Only a complete memory emergency returns 0. */
	if (READ_ONCE(sysctl_mthp_bestfit_pressure_aware)) {
		unsigned long avail = si_mem_available();
		unsigned long min_avail = (unsigned long)
			READ_ONCE(sysctl_mthp_bestfit_pressure_min) << (20 - PAGE_SHIFT);

		if (avail < min_avail)
			allowed_orders &= ~BIT(PMD_ORDER);
	}

	bf_maybe_refresh();

	best = mthp_bestfit_order(vma, allowed_orders, &vtype, &f);

	if (best < 0)
		return allowed_orders;

	bestfit_count(best, vtype, &f);

	if (READ_ONCE(sysctl_mthp_bestfit_dry_run))
		return allowed_orders;

	/*
	 * F1: strict_mask=1 emits exactly one bit. alloc_anon_folio()
	 * attempts every bit it is handed with the same gfp; under
	 * defrag != never each extra bit is a potential direct
	 * reclaim/compaction attempt. One large attempt, then the
	 * implicit order-0 path. strict_mask=0 restores the v3.3/v4.0
	 * 0..best shape for byte-for-byte legacy A/B.
	 */
	if (READ_ONCE(sysctl_mthp_bestfit_strict_mask))
		return (best >= 2) ? (allowed_orders & BIT(best)) : 0;
	return allowed_orders & ((1UL << (best + 1)) - 1);
}

/* -- vendor hook handler --------------------------------------------------- */
static void bestfit_vh_allowable_orders(void *data,
					struct vm_area_struct *vma,
					unsigned long *orders)
{
	(void)data;
	if (unlikely(!vma || !orders))
		return;
	if (!*orders)
		return;
	*orders = __mthp_bestfit_mask(vma, *orders);
}

/* -- sysctl table ----------------------------------------------------------- */
#define BF_CTL(nm, var, lo, hi) {				\
	.procname	= nm,					\
	.data		= &var,					\
	.maxlen		= sizeof(int),				\
	.mode		= 0644,					\
	.proc_handler	= proc_dointvec_minmax,			\
	.extra1		= lo,					\
	.extra2		= hi,					\
}

static struct ctl_table mthp_bestfit_sysctls[] = {
	BF_CTL("enabled",	   sysctl_mthp_bestfit_enabled,	      &bf_int_zero, &bf_int_one),
	BF_CTL("min_pages",	   sysctl_mthp_bestfit_min_pages,     &bf_int_one,  &bf_min_pages_max),
	BF_CTL("contpte_bias",	   sysctl_mthp_bestfit_contpte_bias,  &bf_int_zero, &bf_int_one),
	BF_CTL("buddy_guard_min",  sysctl_mthp_bestfit_buddy_guard,   &bf_int_zero, &bf_buddy_guard_max),
	BF_CTL("frag_cap_pct",	   sysctl_mthp_bestfit_frag_cap,      &bf_int_zero, &bf_frag_cap_max),
	BF_CTL("exec_boost",	   sysctl_mthp_bestfit_exec_boost,    &bf_int_zero, &bf_int_one),
	BF_CTL("pressure_aware",   sysctl_mthp_bestfit_pressure_aware,&bf_int_zero, &bf_int_one),
	/* pressure_min_free: minimum MemAvailable in MB (si_mem_available()
	 * gate); below this, PMD folios are suppressed. */
	BF_CTL("pressure_min_free",sysctl_mthp_bestfit_pressure_min,  &bf_int_one,  &bf_pressure_min_max),
	BF_CTL("dry_run",	   sysctl_mthp_bestfit_dry_run,	      &bf_int_zero, &bf_int_one),
	BF_CTL("lifetime_aware",   sysctl_mthp_bestfit_lifetime_aware,&bf_int_zero, &bf_int_one),
	BF_CTL("frag_fit",	   sysctl_mthp_bestfit_frag_fit,      &bf_int_zero, &bf_int_one),
	BF_CTL("exact_min_blocks", sysctl_mthp_bestfit_exact_min,     &bf_int_one,  &bf_exact_min_max),
	BF_CTL("split_slack",	   sysctl_mthp_bestfit_split_slack,   &bf_int_zero, &bf_slack_max),
	BF_CTL("refresh_ms",	   sysctl_mthp_bestfit_refresh_ms,    &bf_refresh_ms_min, &bf_refresh_ms_max),
	BF_CTL("strict_mask",	   sysctl_mthp_bestfit_strict_mask,   &bf_int_zero, &bf_int_one),
	BF_CTL("herd_guard",	   sysctl_mthp_bestfit_herd_guard,    &bf_int_zero, &bf_int_one),
	BF_CTL("pmd_min_blocks",   sysctl_mthp_bestfit_pmd_min,       &bf_int_zero, &bf_pmd_min_max),
};

static struct ctl_table_header *mthp_bestfit_sysctl_hdr;

/* -- debugfs ----------------------------------------------------------------- */
#ifdef CONFIG_DEBUG_FS
static struct dentry *mthp_bestfit_debugfs;
static struct dentry *mthp_bestfit_dbg_stats;
static struct dentry *mthp_bestfit_dbg_live;
static struct dentry *mthp_bestfit_dbg_enabled;

static const char * const order_names[] = {
	[0] = "  4KB", [1] = "  8KB", [2] = " 16KB", [3] = " 32KB",
	[4] = " 64KB", [5] = "128KB", [6] = "256KB", [7] = "512KB",
	[8] = "  1MB", [9] = "  2MB",
};

static const char * const vtype_names[BF_VMA_TYPES] = {
	[BF_VMA_STACK]	= "stack",
	[BF_VMA_EXEC]	= "exec",
	[BF_VMA_ANON]	= "anon",
	[BF_VMA_FILE]	= "file",
};

static void collect_totals(struct bestfit_counters *total)
{
	int cpu, i;

	memset(total, 0, sizeof(*total));
	for_each_possible_cpu(cpu) {
		struct bestfit_counters *c = per_cpu_ptr(&bestfit_pcpu, cpu);

		total->decisions	    += READ_ONCE(c->decisions);
		total->contpte_hits	    += READ_ONCE(c->contpte_hits);
		total->capped_down	    += READ_ONCE(c->capped_down);
		total->stack_capped	    += READ_ONCE(c->stack_capped);
		total->exec_boosted	    += READ_ONCE(c->exec_boosted);
		total->frag_capped	    += READ_ONCE(c->frag_capped);
		total->buddy_guarded	    += READ_ONCE(c->buddy_guarded);
		total->pressure_downgraded  += READ_ONCE(c->pressure_downgraded);
		total->compact_stall_avoided += READ_ONCE(c->compact_stall_avoided);
		total->lifetime_conserved   += READ_ONCE(c->lifetime_conserved);
		total->exact_fit	    += READ_ONCE(c->exact_fit);
		total->split_fit	    += READ_ONCE(c->split_fit);
		total->frag_suppressed	    += READ_ONCE(c->frag_suppressed);
		total->budget_skips	    += READ_ONCE(c->budget_skips);
		for (i = 0; i <= PMD_ORDER; i++)
			total->order_hist[i] += READ_ONCE(c->order_hist[i]);
		for (i = 0; i < BF_VMA_TYPES; i++)
			total->by_type[i] += READ_ONCE(c->by_type[i]);
	}
}

static int bestfit_stats_show(struct seq_file *m, void *v)
{
	struct bestfit_counters total;
	u64 elig = READ_ONCE(bf_eligible);
	int i;

	collect_totals(&total);

#define PCT(n) (total.decisions > 0 ? (n) * 100 / total.decisions : 0)

	seq_printf(m, "mthp_bestfit v%s statistics\n", MOD_VER);
	seq_puts(m,  "===========================\n");
	seq_printf(m, "mode:           %s%s\n",
		   sysctl_mthp_bestfit_frag_fit ? "frag-fit" : "legacy v3.3",
		   sysctl_mthp_bestfit_strict_mask ? " · strict mask" : " · 0..best mask");
	seq_printf(m, "enabled:        %d   dry_run: %d\n",
		   sysctl_mthp_bestfit_enabled, sysctl_mthp_bestfit_dry_run);
	seq_printf(m, "min_pages:      %d   exact_min: %d   slack: %d\n",
		   sysctl_mthp_bestfit_min_pages,
		   sysctl_mthp_bestfit_exact_min,
		   sysctl_mthp_bestfit_split_slack);
	seq_printf(m, "refresh_ms:     %d (self-clocked)   herd_guard: %d\n",
		   sysctl_mthp_bestfit_refresh_ms,
		   sysctl_mthp_bestfit_herd_guard);
	seq_printf(m, "pressure:       aware=%d min_avail=%d MB (PMD suppressed below this)\n",
		   sysctl_mthp_bestfit_pressure_aware,
		   sysctl_mthp_bestfit_pressure_min);
	seq_printf(m, "pmd_min_blocks: %d (free 2MB buddy blocks needed for PMD selection)\n",
		   sysctl_mthp_bestfit_pmd_min);
	seq_printf(m, "eligible:       exact=0x%03x bounded=0x%03x\n",
		   (u32)elig & 0xffff, (u32)(elig >> 32) & 0xffff);
	seq_printf(m, "monitor active: %s\n",
		   monitor_get_fn ? "yes" : "no (local free-shape only)");

	seq_puts(m, "\nBuddy free shape (cached) / window budget left:\n");
	for (i = 2; i <= PMD_ORDER; i++)
		seq_printf(m, "  order %d (%s): %u free / %d budget%s%s\n",
			   i,
			   i < ARRAY_SIZE(order_names) ? order_names[i] : "?",
			   READ_ONCE(bf_free_raw[i]),
			   atomic_read(&bf_budget[i]),
			   (elig & BIT(i)) ? "  [exact]" : "",
			   ((elig >> 32) & BIT(i)) ? " [bounded]" : "");

	seq_printf(m, "\nTotal decisions:      %llu\n", total.decisions);
	seq_printf(m, "Exact-fit (hole fill): %llu (%llu%%)\n",
		   total.exact_fit, PCT(total.exact_fit));
	seq_printf(m, "Split-fit (bounded):  %llu (%llu%%)\n",
		   total.split_fit, PCT(total.split_fit));
	seq_printf(m, "Frag suppressed:      %llu (%llu%%)\n",
		   total.frag_suppressed, PCT(total.frag_suppressed));
	seq_printf(m, "Budget skips (herd):  %llu\n", total.budget_skips);
	seq_printf(m, "contPTE selected:     %llu (%llu%%)\n",
		   total.contpte_hits, PCT(total.contpte_hits));
	seq_printf(m, "Capped below max:     %llu (%llu%%)\n",
		   total.capped_down, PCT(total.capped_down));
	seq_printf(m, "Stack capped:         %llu\n", total.stack_capped);
	seq_printf(m, "Exec boosted:         %llu (%llu%%)\n",
		   total.exec_boosted, PCT(total.exec_boosted));
	seq_printf(m, "Pressure downgraded:  %llu (%llu%%)\n",
		   total.pressure_downgraded, PCT(total.pressure_downgraded));
	seq_printf(m, "Compact stall avoided: %llu (%llu%%)\n",
		   total.compact_stall_avoided, PCT(total.compact_stall_avoided));
	seq_printf(m, "Lifetime conserved:   %llu (%llu%%)\n",
		   total.lifetime_conserved, PCT(total.lifetime_conserved));
	seq_printf(m, "Frag cap fired:       %llu\n", total.frag_capped);
	seq_printf(m, "Buddy guard fired:    %llu\n", total.buddy_guarded);

	seq_puts(m, "\nBy VMA type:\n");
	for (i = 0; i < BF_VMA_TYPES; i++)
		seq_printf(m, "  %-6s : %llu (%llu%%)\n",
			   vtype_names[i],
			   total.by_type[i], PCT(total.by_type[i]));

	seq_puts(m, "\nOrder histogram (decided; cross-check per-size anon_fault_alloc):\n");
	for (i = 0; i <= PMD_ORDER; i++) {
		u64 cnt = total.order_hist[i];

		if (cnt == 0)
			continue;
		seq_printf(m, "  order %d (%s): %10llu  (%3llu%%)%s\n",
			   i,
			   i < ARRAY_SIZE(order_names) ? order_names[i] : "?",
			   cnt, PCT(cnt),
			   i == CONTPTE_ORDER && CONTPTE_ORDER > 0 ?
				"  [contPTE]" :
			   i == PMD_ORDER ? "  [PMD]" : "");
	}

	return 0;
}
#undef PCT

static int bestfit_stats_open(struct inode *inode, struct file *file)
{
	return single_open(file, bestfit_stats_show, NULL);
}

static ssize_t bestfit_stats_write(struct file *file, const char __user *buf,
				   size_t count, loff_t *ppos)
{
	int cpu;

	for_each_possible_cpu(cpu)
		memset(per_cpu_ptr(&bestfit_pcpu, cpu), 0,
		       sizeof(struct bestfit_counters));
	return count;
}

static const struct file_operations bestfit_stats_fops = {
	.open		= bestfit_stats_open,
	.read		= seq_read,
	.write		= bestfit_stats_write,
	.llseek		= seq_lseek,
	.release	= single_release,
};

static int bestfit_live_show(struct seq_file *m, void *v)
{
	struct bestfit_counters total;
	u64 elig = READ_ONCE(bf_eligible);
	int i;

	collect_totals(&total);
	seq_printf(m, "# mthp_bestfit v%s live view\n", MOD_VER);
	seq_printf(m, "enabled %d\n", sysctl_mthp_bestfit_enabled);
	seq_printf(m, "frag_fit %d\n", sysctl_mthp_bestfit_frag_fit);
	seq_printf(m, "strict_mask %d\n", sysctl_mthp_bestfit_strict_mask);
	seq_printf(m, "herd_guard %d\n", sysctl_mthp_bestfit_herd_guard);
	seq_printf(m, "pmd_min_blocks %d\n", sysctl_mthp_bestfit_pmd_min);
	seq_printf(m, "exact_bm 0x%03x\n", (u32)elig & 0xffff);
	seq_printf(m, "bounded_bm 0x%03x\n", (u32)(elig >> 32) & 0xffff);
	for (i = 2; i <= PMD_ORDER; i++)
		seq_printf(m, "free_order_%d %u\n", i,
			   READ_ONCE(bf_free_raw[i]));
	seq_printf(m, "decisions %llu\n", total.decisions);
	seq_printf(m, "exact_fit %llu\n", total.exact_fit);
	seq_printf(m, "split_fit %llu\n", total.split_fit);
	seq_printf(m, "frag_suppressed %llu\n", total.frag_suppressed);
	seq_printf(m, "budget_skips %llu\n", total.budget_skips);
	seq_printf(m, "lifetime_conserved %llu\n", total.lifetime_conserved);
	seq_printf(m, "compact_stall_avoided %llu\n", total.compact_stall_avoided);
	for (i = 0; i < BF_VMA_TYPES; i++)
		seq_printf(m, "by_%s %llu\n", vtype_names[i], total.by_type[i]);
	for (i = 0; i <= PMD_ORDER; i++) {
		if (total.order_hist[i])
			seq_printf(m, "order_%d %llu\n", i,
				   total.order_hist[i]);
	}
	return 0;
}

static int bestfit_live_open(struct inode *inode, struct file *file)
{ return single_open(file, bestfit_live_show, NULL); }

static const struct file_operations bestfit_live_fops = {
	.open		= bestfit_live_open,
	.read		= seq_read,
	.llseek		= seq_lseek,
	.release	= single_release,
};

static int bestfit_enabled_show(struct seq_file *m, void *v)
{
	seq_printf(m, "%d\n", sysctl_mthp_bestfit_enabled);
	return 0;
}

static int bestfit_enabled_open(struct inode *inode, struct file *file)
{ return single_open(file, bestfit_enabled_show, NULL); }

static ssize_t bestfit_enabled_write(struct file *file,
				     const char __user *buf,
				     size_t count, loff_t *ppos)
{
	char kbuf[8];
	int val;
	size_t n = min(count, sizeof(kbuf) - 1);

	if (copy_from_user(kbuf, buf, n))
		return -EFAULT;
	kbuf[n] = '\0';
	if (kstrtoint(strstrip(kbuf), 10, &val))
		return -EINVAL;
	WRITE_ONCE(sysctl_mthp_bestfit_enabled, val ? 1 : 0);
	return count;
}

static const struct file_operations bestfit_enabled_fops = {
	.open		= bestfit_enabled_open,
	.read		= seq_read,
	.write		= bestfit_enabled_write,
	.llseek		= seq_lseek,
	.release	= single_release,
};

static int mthp_bestfit_debugfs_init(void)
{
	mthp_bestfit_debugfs = debugfs_create_dir("mthp_bestfit", NULL);
	if (IS_ERR(mthp_bestfit_debugfs)) {
		int err = PTR_ERR(mthp_bestfit_debugfs);

		mthp_bestfit_debugfs = NULL;
		return err;
	}

	debugfs_create_file("stats", 0644, mthp_bestfit_debugfs,
			    NULL, &bestfit_stats_fops);
	debugfs_create_file("live", 0444, mthp_bestfit_debugfs,
			    NULL, &bestfit_live_fops);
	debugfs_create_file("enabled", 0644, mthp_bestfit_debugfs,
			    NULL, &bestfit_enabled_fops);

	/* Legacy flat paths so existing scripts keep working. */
	mthp_bestfit_dbg_stats =
		debugfs_create_file("mthp_bestfit_stats", 0644, NULL, NULL,
				    &bestfit_stats_fops);
	mthp_bestfit_dbg_live =
		debugfs_create_file("mthp_bestfit_live", 0444, NULL, NULL,
				    &bestfit_live_fops);
	mthp_bestfit_dbg_enabled =
		debugfs_create_file("mthp_bestfit_enabled", 0644, NULL, NULL,
				    &bestfit_enabled_fops);

	return 0;
}

static void mthp_bestfit_debugfs_exit(void)
{
	debugfs_remove_recursive(mthp_bestfit_debugfs);
	debugfs_remove(mthp_bestfit_dbg_stats);
	debugfs_remove(mthp_bestfit_dbg_live);
	debugfs_remove(mthp_bestfit_dbg_enabled);
}
#else
static inline int  mthp_bestfit_debugfs_init(void) { return 0; }
static inline void mthp_bestfit_debugfs_exit(void) {}
#endif /* CONFIG_DEBUG_FS */

/* -- module init / exit ------------------------------------------------------ */
static int __init mthp_bestfit_init(void)
{
	int ret;

	mthp_bestfit_sysctl_hdr = register_sysctl("vm/mthp_bestfit",
						  mthp_bestfit_sysctls);
	if (!mthp_bestfit_sysctl_hdr)
		pr_warn(MOD_NAME ": register_sysctl failed "
			"(hardened build?), continuing with debugfs only\n");

	ret = register_trace_android_vh_thp_vma_allowable_orders(
			bestfit_vh_allowable_orders, NULL);
	if (ret) {
		pr_err(MOD_NAME ": register vendor hook failed: %d\n", ret);
		if (mthp_bestfit_sysctl_hdr)
			unregister_sysctl_table(mthp_bestfit_sysctl_hdr);
		return ret;
	}

	ret = mthp_bestfit_debugfs_init();
	if (ret)
		pr_warn(MOD_NAME ": debugfs init failed: %d\n", ret);

	monitor_get_fn = symbol_get(mthp_monitor_get_sysinfo);
	pr_info(MOD_NAME ": monitor %s\n",
		monitor_get_fn ? "integration active (frag_index)" :
				 "not found, local free-shape only");

	/* Seed the shape once; afterwards refresh is fault-clocked
	 * (F2: no periodic kworker, zero idle wakeups). */
	bf_recompute_shape();
	WRITE_ONCE(bf_next_refresh, jiffies + msecs_to_jiffies(
		READ_ONCE(sysctl_mthp_bestfit_refresh_ms)));

	pr_info(MOD_NAME " v%s loaded (PMD_ORDER=%d CONTPTE_ORDER=%d "
		"frag_fit=%d strict_mask=%d herd_guard=%d exact_min=%d "
		"pmd_min=%d slack=%d refresh=%dms self-clocked)\n",
		MOD_VER, PMD_ORDER, CONTPTE_ORDER,
		sysctl_mthp_bestfit_frag_fit,
		sysctl_mthp_bestfit_strict_mask,
		sysctl_mthp_bestfit_herd_guard,
		sysctl_mthp_bestfit_exact_min,
		sysctl_mthp_bestfit_pmd_min,
		sysctl_mthp_bestfit_split_slack,
		sysctl_mthp_bestfit_refresh_ms);
	return 0;
}

static void __exit mthp_bestfit_exit(void)
{
	unregister_trace_android_vh_thp_vma_allowable_orders(
			bestfit_vh_allowable_orders, NULL);

	/* All shape refreshes run inside hook callbacks; once callbacks
	 * are drained there is nothing asynchronous left to cancel. */
	tracepoint_synchronize_unregister();

	if (monitor_get_fn) {
		symbol_put(mthp_monitor_get_sysinfo);
		monitor_get_fn = NULL;
	}

	mthp_bestfit_debugfs_exit();
	if (mthp_bestfit_sysctl_hdr)
		unregister_sysctl_table(mthp_bestfit_sysctl_hdr);

	pr_info(MOD_NAME " v" MOD_VER " unloaded\n");
}

module_init(mthp_bestfit_init);
module_exit(mthp_bestfit_exit);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Suyog Buradkar, Pradnya Dahiwale");
MODULE_DESCRIPTION("Fragmentation-shape best-fit mTHP order selection, O(1) fault path");
