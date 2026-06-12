// SPDX-License-Identifier: GPL-2.0
/*
 * mthp_monitor - mTHP fragmentation monitor
 *
 * Buddy/fragmentation polling, THP-split event counting via kprobe,
 * slab leak watchdog via the shrinker interface, and a seqlock-protected
 * system_info snapshot exported through profile.h.
 *
 * Primary target: Android GKI 6.12 (arm64), Clang/LLVM, external module.
 * The kprobe fallback table also covers <=6.7 and the 6.15/6.19 mm
 * reorganisations, so the same source loads on mainline up to v6.19.
 *
 * Pairing note: mthp_bestfit >= v4.0 sources per-order buddy data
 * locally and consumes only frag_index from mthp_monitor_get_sysinfo();
 * buddy_free[] in the exported struct stays at the INT_MAX
 * "unavailable" sentinel (first_online_pgdat()/next_zone() are not
 * exported to modules on GKI or mainline). The monitor remains useful
 * standalone for slab-leak watching, THP-split counting, and the
 * sysfs/debugfs fragmentation view.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/vmstat.h>
#include <linux/swap.h>
#include <linux/workqueue.h>
#include <linux/jiffies.h>
#include <linux/atomic.h>
#include <linux/shrinker.h>
#include <linux/kprobes.h>
#include <linux/seqlock.h>
#include <linux/minmax.h>
#include <linux/sysfs.h>
#include <linux/kobject.h>
#ifdef CONFIG_DEBUG_FS
#include <linux/debugfs.h>
#include <linux/seq_file.h>
#endif

#include "profile.h"

#define MOD_NAME	"mthp_monitor"
#define MOD_VER		"3.1"

/* ---- Config ------------------------------------------------------------ */

static unsigned int buddy_update_ms = 1000;
static int leak_thresh = 1000;
static bool auto_reclaim = true;

module_param(buddy_update_ms, uint, 0644);
module_param(leak_thresh, int, 0644);
module_param(auto_reclaim, bool, 0644);

/* ---- Global state ------------------------------------------------------ */

static unsigned long g_buddy_free[MAX_TRACKED_ORDER];
static seqlock_t g_buddy_lock;
static struct system_info g_system_info;

static atomic64_t g_slab_bytes = ATOMIC64_INIT(0);
static atomic64_t g_pages_in_use = ATOMIC64_INIT(0);
static atomic64_t g_thp_splits = ATOMIC64_INIT(0);
static atomic64_t g_shrinker_runs = ATOMIC64_INIT(0);

/* ---- Buddy polling workqueue ------------------------------------------- */

static struct delayed_work buddy_work;

static void buddy_update_work(struct work_struct *work)
{
	unsigned long total_pages, free_pages;
	unsigned int frag;
	int order;

	/*
	 * first_online_pgdat() and next_zone() are not exported to modules
	 * (GKI and mainline alike: mm/mmzone.c carries no EXPORT_SYMBOL),
	 * so zone->free_area[order].nr_free cannot be read from here. Use
	 * vmstat exports instead: totalram_pages() is an inline helper (no
	 * export needed) and global_zone_page_state(NR_FREE_PAGES) is
	 * exported.
	 */
	total_pages = totalram_pages();
	free_pages = global_zone_page_state(NR_FREE_PAGES);

	/* frag_index = (total - free) * 100 / total  (utilization proxy) */
	frag = total_pages > free_pages ?
		(unsigned int)(((total_pages - free_pages) * 100) /
			       total_pages) : 0;

	write_seqlock(&g_buddy_lock);
	for (order = 0; order < MAX_TRACKED_ORDER; order++) {
		/*
		 * buddy_free[] is set to INT_MAX to signal "data
		 * unavailable", which disables mthp_bestfit's legacy
		 * buddy_guard safely: the guard fires only when
		 * free_at < guard_min; INT_MAX satisfies free_at >=
		 * guard_min for any reasonable guard_min value.
		 * bestfit >= v4.0 ignores this array entirely.
		 */
		g_buddy_free[order] = (unsigned long)INT_MAX;
		g_system_info.buddy_free[order] = (u64)INT_MAX;
	}
	g_system_info.frag_index = frag;
	g_system_info.total_pages = total_pages;
	write_sequnlock(&g_buddy_lock);

	atomic64_set(&g_pages_in_use,
		     (s64)global_node_page_state(NR_ANON_MAPPED) +
		     (s64)global_node_page_state(NR_FILE_PAGES));
	atomic64_set(&g_slab_bytes,
		     (s64)global_node_page_state(NR_SLAB_RECLAIMABLE_B) +
		     (s64)global_node_page_state(NR_SLAB_UNRECLAIMABLE_B));

	schedule_delayed_work(&buddy_work, msecs_to_jiffies(buddy_update_ms));
}

/* ---- Kprobe handlers (event counting only - no function pointer calls) -- */

static int kp_thp_split(struct kprobe *p, struct pt_regs *regs)
{
	atomic64_inc(&g_thp_splits);
	return 0;
}

/*
 * Symbol fallback table, newest first. Which entry lands:
 *
 *   __folio_split                       v6.15+ common split path; static,
 *                                       may be LTO-renamed on Clang
 *                                       FULL_LTO builds (falls through)
 *   __split_huge_page_to_list_to_order  v6.19+ out-of-line global (the
 *                                       old name became a static inline)
 *   split_huge_page_to_list_to_order    v6.8 .. v6.18 -- this is the one
 *                                       that attaches on GKI 6.12; global,
 *                                       so FULL_LTO does not rename it
 *   split_huge_page_to_list             <= v6.7
 */
static const char * const sym_thp_split[] = {
	"__folio_split",
	"__split_huge_page_to_list_to_order",
	"split_huge_page_to_list_to_order",
	"split_huge_page_to_list",
};

struct kprobe_entry {
	struct kprobe kp;
	const char * const *names;
	int name_count;
	bool registered;
};

static struct kprobe_entry kp_table[] = {
	{
		.kp = { .pre_handler = kp_thp_split },
		.names = sym_thp_split,
		.name_count = ARRAY_SIZE(sym_thp_split),
	},
};

static void register_kprobes_all(void)
{
	int i, j, ret;

	for (i = 0; i < ARRAY_SIZE(kp_table); i++) {
		for (j = 0; j < kp_table[i].name_count; j++) {
			kp_table[i].kp.symbol_name = kp_table[i].names[j];
			/* a failed attempt may leave kp.addr resolved */
			kp_table[i].kp.addr = NULL;
			ret = register_kprobe(&kp_table[i].kp);
			if (ret == 0) {
				kp_table[i].registered = true;
				pr_info(MOD_NAME ": kprobe %s attached\n",
					kp_table[i].names[j]);
				break;
			}
		}
		if (!kp_table[i].registered)
			pr_warn(MOD_NAME ": kprobe %s unavailable\n",
				kp_table[i].names[0]);
	}
}

static void unregister_kprobes_all(void)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(kp_table); i++)
		if (kp_table[i].registered)
			unregister_kprobe(&kp_table[i].kp);
}

/* ---- Exported interface ------------------------------------------------- */

/**
 * mthp_monitor_get_sysinfo - read current fragmentation snapshot
 * @out: caller-allocated buffer to fill
 *
 * Seqlock-protected; caller always sees a consistent snapshot.
 * Safe from workqueue or process context. Not for use in IRQ context.
 */
int mthp_monitor_get_sysinfo(struct system_info *out)
{
	unsigned int seq;

	if (!out)
		return -EINVAL;

	do {
		seq = read_seqbegin(&g_buddy_lock);
		memcpy(out, &g_system_info, sizeof(*out));
	} while (read_seqretry(&g_buddy_lock, seq));

	return 0;
}
EXPORT_SYMBOL_GPL(mthp_monitor_get_sysinfo);

/* ---- Shrinker (6.7+ API) ------------------------------------------------ */

static struct shrinker *g_shrinker;

static unsigned long mthp_shrinker_count(struct shrinker *s,
					 struct shrink_control *sc)
{
	long net = atomic64_read(&g_slab_bytes);

	net = max(net, 0L);
	return (net > leak_thresh) ? (unsigned long)net : 0;
}

static unsigned long mthp_shrinker_scan(struct shrinker *s,
					struct shrink_control *sc)
{
	long net = atomic64_read(&g_slab_bytes);

	net = max(net, 0L);
	if (net > leak_thresh && auto_reclaim) {
		atomic64_inc(&g_shrinker_runs);
		/*
		 * The kernel will reclaim slab pages through normal shrinker
		 * chaining. Explicit drop_slab() is not needed here and was
		 * removed because it required a kprobe-resolved function
		 * pointer, which triggers KCFI violations on GKI kernels.
		 */
	}
	return min_t(unsigned long, (unsigned long)net, sc->nr_to_scan);
}

static int shrinker_setup(void)
{
	g_shrinker = shrinker_alloc(0, MOD_NAME);
	if (!g_shrinker)
		return -ENOMEM;

	g_shrinker->count_objects = mthp_shrinker_count;
	g_shrinker->scan_objects = mthp_shrinker_scan;
	g_shrinker->seeks = DEFAULT_SEEKS;

	shrinker_register(g_shrinker);
	return 0;
}

/* ---- debugfs ------------------------------------------------------------ */

#ifdef CONFIG_DEBUG_FS
static struct dentry *dbg_dir;

static int dbg_stats_show(struct seq_file *m, void *v)
{
	seq_printf(m, "=== %s v%s ===\n\n", MOD_NAME, MOD_VER);
	seq_puts(m, "Buddy free blocks: (per-order data unavailable on this build)\n");
	seq_printf(m, "Frag index: %u%% (utilization: total-free/total)\n",
		   g_system_info.frag_index);
	seq_printf(m, "Pages in use: %lld\n",
		   atomic64_read(&g_pages_in_use));
	seq_printf(m, "\nSlab bytes: %lld\n",
		   atomic64_read(&g_slab_bytes));
	seq_printf(m, "THP splits: %lld\n",
		   atomic64_read(&g_thp_splits));
	seq_printf(m, "Shrinker: runs=%lld\n",
		   atomic64_read(&g_shrinker_runs));
	return 0;
}

static int dbg_stats_open(struct inode *inode, struct file *file)
{
	return single_open(file, dbg_stats_show, NULL);
}

/* any write resets the event counters */
static ssize_t dbg_stats_write(struct file *file, const char __user *buf,
			       size_t count, loff_t *ppos)
{
	atomic64_set(&g_slab_bytes, 0);
	atomic64_set(&g_pages_in_use, 0);
	atomic64_set(&g_thp_splits, 0);
	atomic64_set(&g_shrinker_runs, 0);
	return count;
}

static const struct file_operations dbg_stats_fops = {
	.owner		= THIS_MODULE,
	.open		= dbg_stats_open,
	.read		= seq_read,
	.write		= dbg_stats_write,
	.llseek		= seq_lseek,
	.release	= single_release,
};
#endif /* CONFIG_DEBUG_FS */

/* ---- Sysfs -------------------------------------------------------------- */

static struct kobject *mthp_kobj;

static ssize_t system_info_show(struct kobject *kobj,
				struct kobj_attribute *attr, char *buf)
{
	struct system_info si;
	unsigned int seq;

	do {
		seq = read_seqbegin(&g_buddy_lock);
		memcpy(&si, &g_system_info, sizeof(si));
	} while (read_seqretry(&g_buddy_lock, seq));

	return sysfs_emit(buf, "total_pages: %lu\nfrag_index: %u\n",
			  si.total_pages, si.frag_index);
}

static struct kobj_attribute system_info_attr =
	__ATTR(system_info, 0444, system_info_show, NULL);

/* ---- Init / Exit -------------------------------------------------------- */

static int __init mthp_monitor_init(void)
{
	pr_info(MOD_NAME ": loading v%s\n", MOD_VER);

	seqlock_init(&g_buddy_lock);

	register_kprobes_all();

	INIT_DELAYED_WORK(&buddy_work, buddy_update_work);
	schedule_delayed_work(&buddy_work, msecs_to_jiffies(buddy_update_ms));

	if (shrinker_setup() < 0)
		pr_warn(MOD_NAME ": shrinker_alloc failed, shrinker disabled\n");

	mthp_kobj = kobject_create_and_add(MOD_NAME, kernel_kobj);
	if (mthp_kobj) {
		if (sysfs_create_file(mthp_kobj, &system_info_attr.attr) < 0)
			pr_warn(MOD_NAME ": sysfs create file failed\n");
	}

#ifdef CONFIG_DEBUG_FS
	dbg_dir = debugfs_create_dir(MOD_NAME, NULL);
	if (dbg_dir)
		debugfs_create_file("stats", 0644, dbg_dir, NULL,
				    &dbg_stats_fops);
#endif

	pr_info(MOD_NAME ": loaded buddy=%ums sysfs=/sys/kernel/%s/system_info\n",
		buddy_update_ms, MOD_NAME);
	return 0;
}

static void __exit mthp_monitor_exit(void)
{
	cancel_delayed_work_sync(&buddy_work);

	if (mthp_kobj) {
		sysfs_remove_file(mthp_kobj, &system_info_attr.attr);
		kobject_put(mthp_kobj);
	}

#ifdef CONFIG_DEBUG_FS
	debugfs_remove(dbg_dir);
#endif

	if (g_shrinker)
		shrinker_free(g_shrinker);

	unregister_kprobes_all();

	pr_info(MOD_NAME ": unloaded\n");
}

module_init(mthp_monitor_init);
module_exit(mthp_monitor_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("mTHP fragmentation monitor: buddy polling, slab shrinker, frag export");
MODULE_VERSION(MOD_VER);
