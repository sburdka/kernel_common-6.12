// SPDX-License-Identifier: GPL-2.0
/*
 * drivers/misc/boot_kpi.c  —  Boot KPI tracer
 *
 * Measures stop_machine() and migration-thread latency during the kernel
 * boot phase and aggregates per-caller statistics so the slowest paths can
 * be identified and targeted for boot-time optimisation.
 *
 * Boot phase window
 * -----------------
 * From module_init() until the first of:
 *   a) kprobe on run_init_process() fires (first userspace binary launched)
 *   b) user writes "1" to debugfs/boot_kpi/boot_done
 *
 * What is tracked during boot
 * ---------------------------
 *   stop_machine() calls — aggregated per call-site (caller):
 *     • invocation count, total/avg/max duration
 *     • per-state breakdown: NONE, PREPARE, DISABLE_IRQ, RUN
 *       (NONE→PREPARE = scheduling latency for this CPU,
 *        PREPARE→DISABLE_IRQ = barrier: waiting for all CPUs,
 *        DISABLE_IRQ→RUN = IRQ-disable synchronisation cost,
 *        RUN→EXIT = actual payload fn() execution time)
 *   migration-thread work items — per work-function:
 *     • queue-to-start latency (scheduling delay = how long the work
 *       waited in the stopper queue before execution began)
 *     • execution time
 *
 * debugfs: /sys/kernel/debug/boot_kpi/
 *   boot_done     rw  0=boot active, 1=done; write "1" to end manually
 *   summary       ro  per-caller stop_machine stats sorted by total duration
 *   timeline      ro  chronological log of every boot-phase stop_machine call
 *   mig_latency   ro  migration-thread work items: q2s + exec latency
 *   clear         wo  reset all data and re-arm boot phase
 *
 * Tracepoints used (all exported by kernel/stop_machine.c):
 *   multi_cpu_stop_begin / multi_cpu_stop_state / multi_cpu_stop_end
 *   cpu_stop_work_begin  / cpu_stop_work_end
 */

#define pr_fmt(fmt) "boot_kpi: " fmt

#include <linux/atomic.h>
#include <linux/debugfs.h>
#include <linux/init.h>
#include <linux/kprobes.h>
#include <linux/ktime.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/percpu.h>
#include <linux/seq_file.h>
#include <linux/sort.h>
#include <linux/spinlock.h>
#include <linux/stop_machine.h>
#include <linux/string.h>
#include <linux/uaccess.h>

#include <trace/events/stopmachine.h>

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

#define BOOT_KPI_CALLERS   64   /* max distinct stop_machine callers tracked */
#define BOOT_KPI_RING      256  /* chronological timeline entries             */
#define BOOT_KPI_MIG_SLOTS 32   /* distinct migration work functions tracked  */

/* ------------------------------------------------------------------ */
/* Boot phase gate                                                      */
/* ------------------------------------------------------------------ */

static atomic_t boot_phase    = ATOMIC_INIT(1);
static u64      boot_start_ns;
static u64      boot_end_ns;

/* ------------------------------------------------------------------ */
/* Per-caller stop_machine aggregation                                  */
/* ------------------------------------------------------------------ */

struct caller_stat {
	unsigned long caller;
	unsigned long fn;
	u32           count;
	u64           total_ns;
	u64           max_ns;
	u64           state_total_ns[MULTI_STOP_MAX_STATE];
};

static struct caller_stat kpi_callers[BOOT_KPI_CALLERS];
static u32                kpi_num_callers;
static u64                kpi_total_stop_ns;
static u32                kpi_total_stop_count;

/* ------------------------------------------------------------------ */
/* Chronological timeline ring                                          */
/* ------------------------------------------------------------------ */

struct timeline_entry {
	unsigned long caller;
	unsigned long fn;
	u64           rel_ns;   /* ns since boot_start_ns */
	u64           dur_ns;
	u64           state_ns[MULTI_STOP_MAX_STATE];
};

static struct timeline_entry kpi_timeline[BOOT_KPI_RING];
static u32                   kpi_tl_idx;
static u32                   kpi_tl_total;

/* ------------------------------------------------------------------ */
/* Migration-thread work item latency                                   */
/* ------------------------------------------------------------------ */

struct mig_slot {
	unsigned long fn;
	u32           count;
	u64           total_q2s_ns;
	u64           max_q2s_ns;
	u64           total_exec_ns;
	u64           max_exec_ns;
};

static struct mig_slot kpi_mig[BOOT_KPI_MIG_SLOTS];
static u32             kpi_mig_num;

struct mig_scratch {
	unsigned long fn;
	u64           q2s_ns;
	u64           start_ns;
	bool          valid;
};
static DEFINE_PER_CPU(struct mig_scratch, mig_scratch);

/* ------------------------------------------------------------------ */
/* Per-CPU scratch: correlate multi_cpu_stop begin/state/end           */
/* ------------------------------------------------------------------ */

struct stop_scratch {
	unsigned long caller;
	unsigned long fn;
	u64           enter_ns;
	u64           state_ns[MULTI_STOP_MAX_STATE];
	bool          is_active;
	bool          valid;
};
static DEFINE_PER_CPU(struct stop_scratch, stop_scratch);

/* single lock for all aggregation tables */
static DEFINE_SPINLOCK(kpi_lock);

/* ------------------------------------------------------------------ */
/* Helpers: find-or-create slots (called under kpi_lock)               */
/* ------------------------------------------------------------------ */

static struct caller_stat *find_or_create_caller(unsigned long caller)
{
	u32 i;

	for (i = 0; i < kpi_num_callers; i++)
		if (kpi_callers[i].caller == caller)
			return &kpi_callers[i];

	if (kpi_num_callers >= BOOT_KPI_CALLERS)
		return NULL;

	kpi_callers[kpi_num_callers].caller = caller;
	return &kpi_callers[kpi_num_callers++];
}

static struct mig_slot *find_or_create_mig(unsigned long fn)
{
	u32 i;

	for (i = 0; i < kpi_mig_num; i++)
		if (kpi_mig[i].fn == fn)
			return &kpi_mig[i];

	if (kpi_mig_num >= BOOT_KPI_MIG_SLOTS)
		return NULL;

	kpi_mig[kpi_mig_num].fn = fn;
	return &kpi_mig[kpi_mig_num++];
}

/* ------------------------------------------------------------------ */
/* Tracepoint handlers — multi_cpu_stop                                */
/* ------------------------------------------------------------------ */

static void kpi_on_stop_begin(void *_unused,
			      unsigned int cpu, unsigned long fn,
			      unsigned long caller, unsigned int num_threads,
			      bool is_active, u64 enter_ns)
{
	struct stop_scratch *sc;

	if (!atomic_read(&boot_phase))
		return;

	sc            = per_cpu_ptr(&stop_scratch, cpu);
	sc->caller    = caller;
	sc->fn        = fn;
	sc->enter_ns  = enter_ns;
	sc->is_active = is_active;
	sc->valid     = true;
	memset(sc->state_ns, 0, sizeof(sc->state_ns));
}

static void kpi_on_stop_state(void *_unused,
			      unsigned int cpu, unsigned long fn,
			      int prev_state, int new_state, u64 dur_ns)
{
	struct stop_scratch *sc;

	if (!atomic_read(&boot_phase))
		return;
	if (prev_state < 0 || prev_state >= MULTI_STOP_MAX_STATE)
		return;

	sc = per_cpu_ptr(&stop_scratch, cpu);
	if (sc->valid && sc->fn == fn)
		sc->state_ns[prev_state] += dur_ns;
}

static void kpi_on_stop_end(void *_unused,
			    unsigned int cpu, unsigned long fn,
			    unsigned long caller,
			    u64 enter_ns, u64 exit_ns, int err)
{
	struct stop_scratch  *sc = per_cpu_ptr(&stop_scratch, cpu);
	struct caller_stat   *cs;
	struct timeline_entry *te;
	unsigned long flags;
	u64 dur = exit_ns - enter_ns;
	u32 i;

	if (!atomic_read(&boot_phase))
		return;
	/* aggregate once per invocation from the active CPU only */
	if (!sc->valid || !sc->is_active ||
	    sc->fn != fn || sc->enter_ns != enter_ns)
		return;

	sc->valid = false;

	spin_lock_irqsave(&kpi_lock, flags);

	cs = find_or_create_caller(caller);
	if (cs) {
		cs->fn        = fn;
		cs->count++;
		cs->total_ns += dur;
		if (dur > cs->max_ns)
			cs->max_ns = dur;
		for (i = 0; i < MULTI_STOP_MAX_STATE; i++)
			cs->state_total_ns[i] += sc->state_ns[i];
	}

	kpi_total_stop_ns    += dur;
	kpi_total_stop_count++;

	te = &kpi_timeline[kpi_tl_idx % BOOT_KPI_RING];
	te->caller = caller;
	te->fn     = fn;
	te->rel_ns = enter_ns - boot_start_ns;
	te->dur_ns = dur;
	memcpy(te->state_ns, sc->state_ns, sizeof(sc->state_ns));
	kpi_tl_idx++;
	kpi_tl_total++;

	spin_unlock_irqrestore(&kpi_lock, flags);
}

/* ------------------------------------------------------------------ */
/* Tracepoint handlers — migration thread work items                   */
/* ------------------------------------------------------------------ */

static void kpi_on_work_begin(void *_unused,
			      unsigned int cpu, unsigned long fn,
			      unsigned long caller, u64 queued_ns, u64 start_ns)
{
	struct mig_scratch *sc;

	if (!atomic_read(&boot_phase))
		return;

	sc           = per_cpu_ptr(&mig_scratch, cpu);
	sc->fn       = fn;
	sc->q2s_ns   = (start_ns >= queued_ns) ? (start_ns - queued_ns) : 0;
	sc->start_ns = start_ns;
	sc->valid    = true;
}

static void kpi_on_work_end(void *_unused,
			    unsigned int cpu, unsigned long fn,
			    unsigned long caller, u64 start_ns, u64 end_ns)
{
	struct mig_scratch *sc = per_cpu_ptr(&mig_scratch, cpu);
	struct mig_slot    *ms;
	unsigned long flags;
	u64 exec_ns;

	if (!atomic_read(&boot_phase))
		return;
	if (!sc->valid || sc->fn != fn)
		return;

	sc->valid = false;
	exec_ns   = end_ns - start_ns;

	spin_lock_irqsave(&kpi_lock, flags);
	ms = find_or_create_mig(fn);
	if (ms) {
		ms->count++;
		ms->total_q2s_ns  += sc->q2s_ns;
		ms->total_exec_ns += exec_ns;
		if (sc->q2s_ns > ms->max_q2s_ns)
			ms->max_q2s_ns = sc->q2s_ns;
		if (exec_ns > ms->max_exec_ns)
			ms->max_exec_ns = exec_ns;
	}
	spin_unlock_irqrestore(&kpi_lock, flags);
}

/* ------------------------------------------------------------------ */
/* Boot-done: kprobe on run_init_process                               */
/* ------------------------------------------------------------------ */

static void mark_boot_done(const char *trigger)
{
	if (atomic_cmpxchg(&boot_phase, 1, 0) != 1)
		return;

	boot_end_ns = ktime_get_ns();
	pr_info("boot phase ended (%s) at %llu ms: %u stop_machine calls, "
		"%llu ms cumulative stop latency\n",
		trigger,
		(boot_end_ns - boot_start_ns) / 1000000,
		kpi_total_stop_count,
		kpi_total_stop_ns    / 1000000);
}

static int run_init_pre(struct kprobe *p, struct pt_regs *regs)
{
	mark_boot_done("run_init_process");
	return 0;
}

static struct kprobe init_kp = {
	.symbol_name = "run_init_process",
	.pre_handler = run_init_pre,
};

/* ------------------------------------------------------------------ */
/* Duration formatter                                                   */
/* ------------------------------------------------------------------ */

static int fmt_ns(char *buf, size_t sz, u64 ns)
{
	if (!ns)
		return scnprintf(buf, sz, "0 ns");
	if (ns < 1000ULL)
		return scnprintf(buf, sz, "%llu ns", ns);
	if (ns < 1000000ULL)
		return scnprintf(buf, sz, "%llu.%02llu us",
				 ns / 1000, (ns % 1000) / 10);
	if (ns < 1000000000ULL)
		return scnprintf(buf, sz, "%llu.%02llu ms",
				 ns / 1000000, (ns % 1000000) / 10000);
	return scnprintf(buf, sz, "%llu.%02llu s",
			 ns / 1000000000ULL,
			 (ns % 1000000000ULL) / 10000000ULL);
}

/* ------------------------------------------------------------------ */
/* Sorting comparators                                                  */
/* ------------------------------------------------------------------ */

static int cmp_caller_by_total(const void *a, const void *b)
{
	const struct caller_stat *ca = a, *cb = b;

	if (cb->total_ns > ca->total_ns) return  1;
	if (cb->total_ns < ca->total_ns) return -1;
	return 0;
}

static int cmp_mig_by_exec(const void *a, const void *b)
{
	const struct mig_slot *ma = a, *mb = b;

	if (mb->total_exec_ns > ma->total_exec_ns) return  1;
	if (mb->total_exec_ns < ma->total_exec_ns) return -1;
	return 0;
}

/* ------------------------------------------------------------------ */
/* debugfs — boot_done                                                  */
/* ------------------------------------------------------------------ */

static ssize_t boot_done_read(struct file *f, char __user *buf,
			      size_t len, loff_t *ppos)
{
	char tmp[4];
	int  n = scnprintf(tmp, sizeof(tmp), "%d\n",
			   atomic_read(&boot_phase) ? 0 : 1);

	return simple_read_from_buffer(buf, len, ppos, tmp, n);
}

static ssize_t boot_done_write(struct file *f, const char __user *buf,
			       size_t len, loff_t *ppos)
{
	char         tmp[8] = {};
	unsigned int val;

	if (copy_from_user(tmp, buf, min(len, sizeof(tmp) - 1)))
		return -EFAULT;
	if (kstrtouint(strim(tmp), 0, &val))
		return -EINVAL;
	if (val)
		mark_boot_done("debugfs");
	return len;
}

static const struct file_operations boot_done_fops = {
	.open  = simple_open,
	.read  = boot_done_read,
	.write = boot_done_write,
};

/* ------------------------------------------------------------------ */
/* debugfs — summary                                                    */
/* ------------------------------------------------------------------ */

static const char * const state_label[] = {
	[MULTI_STOP_NONE]        = "NONE",
	[MULTI_STOP_PREPARE]     = "PREPARE",
	[MULTI_STOP_DISABLE_IRQ] = "DISABLE_IRQ",
	[MULTI_STOP_RUN]         = "RUN",
	[MULTI_STOP_EXIT]        = "EXIT",
};

static int summary_show(struct seq_file *m, void *v)
{
	struct caller_stat sorted[BOOT_KPI_CALLERS];
	unsigned long flags;
	u32 n, i, s;
	u64 boot_dur;
	char dur[32], avg[32], max[32], sbuf[32];

	spin_lock_irqsave(&kpi_lock, flags);
	n        = kpi_num_callers;
	boot_dur = boot_end_ns ? (boot_end_ns - boot_start_ns)
			       : (ktime_get_ns() - boot_start_ns);
	memcpy(sorted, kpi_callers, n * sizeof(sorted[0]));
	spin_unlock_irqrestore(&kpi_lock, flags);

	sort(sorted, n, sizeof(sorted[0]), cmp_caller_by_total, NULL);

	fmt_ns(dur, sizeof(dur), boot_dur);
	seq_printf(m,
		   "Boot KPI — stop_machine() summary\n"
		   "  Phase      : %s\n"
		   "  Boot window: %s\n",
		   atomic_read(&boot_phase) ? "ACTIVE (boot not yet done)" : "DONE",
		   dur);

	fmt_ns(dur, sizeof(dur), kpi_total_stop_ns);
	seq_printf(m,
		   "  Calls      : %u total stop_machine() invocations\n"
		   "  Total stop : %s cumulative latency across all CPUs\n\n",
		   kpi_total_stop_count, dur);

	if (!n) {
		seq_puts(m, "  (no data yet)\n");
		return 0;
	}

	seq_printf(m, "  %-4s  %-5s  %-14s %-14s %-14s  caller\n"
		      "                                              -> fn\n"
		      "  %-4s  %-5s  %-14s %-14s %-14s  ------\n",
		   "Rank", "Count", "total", "avg", "max",
		   "----", "-----", "-----", "---", "---");

	for (i = 0; i < n; i++) {
		struct caller_stat *c = &sorted[i];
		u64 avg_ns = c->count ? div64_u64(c->total_ns, c->count) : 0;

		fmt_ns(dur, sizeof(dur), c->total_ns);
		fmt_ns(avg, sizeof(avg), avg_ns);
		fmt_ns(max, sizeof(max), c->max_ns);

		seq_printf(m,
			   "  #%-3u  %-5u  %-14s %-14s %-14s  %pS\n"
			   "                                              -> %pS\n",
			   i + 1, c->count, dur, avg, max,
			   (void *)c->caller, (void *)c->fn);

		seq_puts(m, "         state avg:");
		for (s = 0; s < MULTI_STOP_MAX_STATE - 1; s++) {
			u64 savg = c->count ?
				div64_u64(c->state_total_ns[s], c->count) : 0;
			fmt_ns(sbuf, sizeof(sbuf), savg);
			seq_printf(m, "  %s=%s", state_label[s], sbuf);
		}
		seq_putc(m, '\n');
	}

	seq_puts(m,
		 "\nOptimisation hints:\n"
		 "  High PREPARE      -> CPUs slow to schedule; look for early-boot IRQ load\n"
		 "  High DISABLE_IRQ  -> IRQ-disable barrier overhead; reduce online CPU count\n"
		 "  High RUN          -> payload fn() itself is slow; batch or defer it\n"
		 "  High call count   -> repeated small stops; batch into one stop_machine()\n");

	return 0;
}

static int summary_open(struct inode *inode, struct file *file)
{
	return single_open(file, summary_show, NULL);
}

static const struct file_operations summary_fops = {
	.open    = summary_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/* ------------------------------------------------------------------ */
/* debugfs — timeline                                                   */
/* ------------------------------------------------------------------ */

static int timeline_show(struct seq_file *m, void *v)
{
	unsigned long flags;
	u32 total, start_i, i;
	char rel[32], dur[32], s0[32], s1[32], s2[32], s3[32];

	spin_lock_irqsave(&kpi_lock, flags);
	total   = kpi_tl_total;
	start_i = (total > BOOT_KPI_RING) ? total - BOOT_KPI_RING : 0;
	spin_unlock_irqrestore(&kpi_lock, flags);

	seq_printf(m,
		   "Boot KPI — stop_machine() timeline  (%u total, last %u shown)\n"
		   "  %-13s  %-12s  %-12s %-12s %-12s %-12s  caller -> fn\n"
		   "  %s  %s  %s %s %s %s  ----------\n",
		   total, min(total, (u32)BOOT_KPI_RING),
		   "rel(boot)", "total",
		   state_label[MULTI_STOP_NONE],
		   state_label[MULTI_STOP_PREPARE],
		   state_label[MULTI_STOP_DISABLE_IRQ],
		   state_label[MULTI_STOP_RUN],
		   "-------------", "------------",
		   "------------", "------------",
		   "------------", "------------");

	for (i = start_i; i < total; i++) {
		struct timeline_entry *te = &kpi_timeline[i % BOOT_KPI_RING];

		fmt_ns(rel, sizeof(rel), te->rel_ns);
		fmt_ns(dur, sizeof(dur), te->dur_ns);
		fmt_ns(s0,  sizeof(s0),  te->state_ns[MULTI_STOP_NONE]);
		fmt_ns(s1,  sizeof(s1),  te->state_ns[MULTI_STOP_PREPARE]);
		fmt_ns(s2,  sizeof(s2),  te->state_ns[MULTI_STOP_DISABLE_IRQ]);
		fmt_ns(s3,  sizeof(s3),  te->state_ns[MULTI_STOP_RUN]);

		seq_printf(m,
			   "  %-13s  %-12s  %-12s %-12s %-12s %-12s  %pS -> %pS\n",
			   rel, dur, s0, s1, s2, s3,
			   (void *)te->caller, (void *)te->fn);
	}

	if (!total)
		seq_puts(m, "  (no entries yet)\n");

	return 0;
}

static int timeline_open(struct inode *inode, struct file *file)
{
	return single_open(file, timeline_show, NULL);
}

static const struct file_operations timeline_fops = {
	.open    = timeline_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/* ------------------------------------------------------------------ */
/* debugfs — mig_latency                                               */
/* ------------------------------------------------------------------ */

static int mig_latency_show(struct seq_file *m, void *v)
{
	struct mig_slot sorted[BOOT_KPI_MIG_SLOTS];
	unsigned long   flags;
	u32 n, i;
	char qa[32], qm[32], ea[32], em[32];

	spin_lock_irqsave(&kpi_lock, flags);
	n = kpi_mig_num;
	memcpy(sorted, kpi_mig, n * sizeof(sorted[0]));
	spin_unlock_irqrestore(&kpi_lock, flags);

	sort(sorted, n, sizeof(sorted[0]), cmp_mig_by_exec, NULL);

	seq_puts(m,
		 "Boot KPI — migration thread work items  (sorted by total exec time)\n"
		 "  q2s = queue-to-start latency (scheduling delay before execution)\n\n"
		 "  Count  q2s_avg       q2s_max       exec_avg      exec_max      fn\n"
		 "  -----  ------------- ------------- ------------- ------------- --\n");

	for (i = 0; i < n; i++) {
		struct mig_slot *ms = &sorted[i];
		u64 q_avg = ms->count ? div64_u64(ms->total_q2s_ns,  ms->count) : 0;
		u64 e_avg = ms->count ? div64_u64(ms->total_exec_ns, ms->count) : 0;

		fmt_ns(qa, sizeof(qa), q_avg);
		fmt_ns(qm, sizeof(qm), ms->max_q2s_ns);
		fmt_ns(ea, sizeof(ea), e_avg);
		fmt_ns(em, sizeof(em), ms->max_exec_ns);

		seq_printf(m, "  %-5u  %-13s %-13s %-13s %-13s %pS\n",
			   ms->count, qa, qm, ea, em, (void *)ms->fn);
	}

	if (!n)
		seq_puts(m, "  (no entries yet)\n");

	seq_puts(m,
		 "\nHigh q2s_max: stopper thread was delayed reaching the CPU.\n"
		 "  -> look for early-boot IRQ floods or preempt-disabled sections\n"
		 "     that prevent the migration/%d thread from running promptly.\n");

	return 0;
}

static int mig_latency_open(struct inode *inode, struct file *file)
{
	return single_open(file, mig_latency_show, NULL);
}

static const struct file_operations mig_latency_fops = {
	.open    = mig_latency_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/* ------------------------------------------------------------------ */
/* debugfs — clear                                                      */
/* ------------------------------------------------------------------ */

static ssize_t clear_write(struct file *f, const char __user *buf,
			   size_t len, loff_t *ppos)
{
	unsigned long flags;

	spin_lock_irqsave(&kpi_lock, flags);
	memset(kpi_callers,  0, sizeof(kpi_callers));
	memset(kpi_timeline, 0, sizeof(kpi_timeline));
	memset(kpi_mig,      0, sizeof(kpi_mig));
	kpi_num_callers      = 0;
	kpi_total_stop_ns    = 0;
	kpi_total_stop_count = 0;
	kpi_tl_idx           = 0;
	kpi_tl_total         = 0;
	kpi_mig_num          = 0;
	spin_unlock_irqrestore(&kpi_lock, flags);

	boot_start_ns = ktime_get_ns();
	boot_end_ns   = 0;
	atomic_set(&boot_phase, 1);

	pr_info("data cleared, boot phase re-armed\n");
	return len;
}

static const struct file_operations clear_fops = {
	.open  = simple_open,
	.write = clear_write,
};

/* ------------------------------------------------------------------ */
/* debugfs setup / teardown                                            */
/* ------------------------------------------------------------------ */

static struct dentry *kpi_dentry;

static int debugfs_setup(void)
{
	kpi_dentry = debugfs_create_dir("boot_kpi", NULL);
	if (IS_ERR(kpi_dentry))
		return PTR_ERR(kpi_dentry);

	debugfs_create_file("boot_done",   0644, kpi_dentry, NULL, &boot_done_fops);
	debugfs_create_file("summary",     0444, kpi_dentry, NULL, &summary_fops);
	debugfs_create_file("timeline",    0444, kpi_dentry, NULL, &timeline_fops);
	debugfs_create_file("mig_latency", 0444, kpi_dentry, NULL, &mig_latency_fops);
	debugfs_create_file("clear",       0200, kpi_dentry, NULL, &clear_fops);

	pr_info("debugfs: /sys/kernel/debug/boot_kpi/\n");
	return 0;
}

static void debugfs_teardown(void)
{
	debugfs_remove_recursive(kpi_dentry);
	kpi_dentry = NULL;
}

/* ------------------------------------------------------------------ */
/* Module lifecycle                                                     */
/* ------------------------------------------------------------------ */

static int __init boot_kpi_init(void)
{
	int ret;

	boot_start_ns = ktime_get_ns();

	ret = debugfs_setup();
	if (ret) {
		pr_err("debugfs setup failed: %d\n", ret);
		return ret;
	}

	ret = register_trace_multi_cpu_stop_begin(kpi_on_stop_begin, NULL);
	if (ret) {
		pr_err("register multi_cpu_stop_begin failed: %d\n", ret);
		goto err_debugfs;
	}

	ret = register_trace_multi_cpu_stop_state(kpi_on_stop_state, NULL);
	if (ret) {
		pr_err("register multi_cpu_stop_state failed: %d\n", ret);
		goto err_unreg_stop_begin;
	}

	ret = register_trace_multi_cpu_stop_end(kpi_on_stop_end, NULL);
	if (ret) {
		pr_err("register multi_cpu_stop_end failed: %d\n", ret);
		goto err_unreg_stop_state;
	}

	ret = register_trace_cpu_stop_work_begin(kpi_on_work_begin, NULL);
	if (ret) {
		pr_err("register cpu_stop_work_begin failed: %d\n", ret);
		goto err_unreg_stop_end;
	}

	ret = register_trace_cpu_stop_work_end(kpi_on_work_end, NULL);
	if (ret) {
		pr_err("register cpu_stop_work_end failed: %d\n", ret);
		goto err_unreg_work_begin;
	}

	/*
	 * Auto-detect end of boot by probing run_init_process().
	 * Non-fatal: user can write "1" to boot_done manually if unavailable.
	 */
	ret = register_kprobe(&init_kp);
	if (ret)
		pr_warn("kprobe on run_init_process unavailable (%d) — "
			"write 1 to debugfs/boot_kpi/boot_done when userspace starts\n",
			ret);

	pr_info("loaded: tracking stop_machine() + migration threads during boot\n");
	return 0;

err_unreg_work_begin:
	unregister_trace_cpu_stop_work_begin(kpi_on_work_begin, NULL);
err_unreg_stop_end:
	unregister_trace_multi_cpu_stop_end(kpi_on_stop_end, NULL);
err_unreg_stop_state:
	unregister_trace_multi_cpu_stop_state(kpi_on_stop_state, NULL);
err_unreg_stop_begin:
	unregister_trace_multi_cpu_stop_begin(kpi_on_stop_begin, NULL);
err_debugfs:
	debugfs_teardown();
	return ret;
}
module_init(boot_kpi_init);

static void __exit boot_kpi_exit(void)
{
	unregister_kprobe(&init_kp);
	unregister_trace_cpu_stop_work_end(kpi_on_work_end, NULL);
	unregister_trace_cpu_stop_work_begin(kpi_on_work_begin, NULL);
	unregister_trace_multi_cpu_stop_end(kpi_on_stop_end, NULL);
	unregister_trace_multi_cpu_stop_state(kpi_on_stop_state, NULL);
	unregister_trace_multi_cpu_stop_begin(kpi_on_stop_begin, NULL);
	tracepoint_synchronize_unregister();
	debugfs_teardown();
	pr_info("unloaded\n");
}
module_exit(boot_kpi_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Boot KPI tracer: stop_machine and migration thread latency during boot");
