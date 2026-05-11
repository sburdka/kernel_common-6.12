// SPDX-License-Identifier: GPL-2.0
/*
 * drivers/misc/migration_tracer.c
 *
 * Records the call history of every per-CPU migration/stopper thread work
 * item and exposes it via debugfs.  Automatically dumps history to the
 * kernel log when the QCOM watchdog fires.
 *
 * Tracepoints
 * -----------
 * Subscribes to two events from <trace/events/stopmachine.h>:
 *   cpu_stop_work_begin  fired just before fn(arg) is called
 *   cpu_stop_work_end    fired just after  fn(arg) returns
 *
 * Per-CPU ring buffer (MIGRATION_HIST_SIZE entries)
 * -------------------------------------------------
 * Each slot records:
 *   caller    _RET_IP_ captured at enqueue time (who requested the work)
 *   fn        the stop callback address
 *   queued_ns ktime_get_ns() at enqueue
 *   start_ns  ktime_get_ns() when execution began
 *   end_ns    ktime_get_ns() when done; 0 means still running
 *
 * Per-CPU stats (updated in on_work_end, reset via debugfs/clear)
 * ---------------------------------------------------------------
 *   total_dispatched  total work items seen on this CPU
 *   total_exec_ns     sum of all execution durations
 *   max_exec_ns       longest single execution (fn + caller stored)
 *   min_exec_ns       shortest single execution (0 = no data yet)
 *
 * debugfs interface  (/sys/kernel/debug/migration_tracer/)
 * --------------------------------------------------------
 *   enable   rw  write "1" to enable, "0" to disable tracing
 *   history  ro  chronological dump of all CPUs' ring buffers
 *   current  ro  only in-flight (end_ns == 0) work items
 *   stats    ro  per-CPU aggregate statistics
 *   clear    wo  write anything to reset all ring buffers and stats
 *
 * QCOM watchdog integration
 * -------------------------
 * QCOM hardware watchdog fires a bark (pretimeout) interrupt before the
 * final bite (hardware reset).  The bark ISR calls watchdog_notify_pretimeout()
 * which triggers the pretimeout governor (typically pretimeout_panic -> panic).
 * The bite itself is a direct hardware reset with no software notification.
 *
 *   Bark hook:  kprobe on watchdog_notify_pretimeout()
 *               Fires from the QCOM bark IRQ handler, before the governor
 *               (and therefore before any panic).  Dumps the migration history
 *               for every CPU immediately.
 *
 *   Panic hook: atomic panic notifier (priority INT_MAX, runs first)
 *               Backup path: fires inside panic() regardless of the trigger
 *               cause.  Only dumps CPUs that have a migration work item
 *               currently stuck (end_ns == 0), so the output stays concise
 *               when the panic has an unrelated cause.
 *
 * NMI safety
 * ----------
 * dump_migration_history() uses only READ_ONCE() and ktime_get_ns(),
 * both safe from any context including NMI.  Ring-buffer writes use
 * smp_wmb() + WRITE_ONCE() for the same reason.
 */

#define pr_fmt(fmt) "migration_tracer: " fmt

#include <linux/atomic.h>
#include <linux/debugfs.h>
#include <linux/kprobes.h>
#include <linux/ktime.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/nmi.h>
#include <linux/notifier.h>
#include <linux/panic_notifier.h>
#include <linux/percpu.h>
#include <linux/sched.h>
#include <linux/seq_file.h>
#include <linux/smp.h>
#include <linux/stop_machine.h>
#include <linux/string.h>
#include <linux/uaccess.h>
#include <linux/watchdog.h>

#include <trace/events/stopmachine.h>

/* ------------------------------------------------------------------ */
/* Compile-time constants                                               */
/* ------------------------------------------------------------------ */

#define MIGRATION_HIST_SIZE	32

/* ------------------------------------------------------------------ */
/* Global on/off gate                                                   */
/* ------------------------------------------------------------------ */

static atomic_t tracer_enabled = ATOMIC_INIT(1);

/* ------------------------------------------------------------------ */
/* multi_cpu_stop() per-CPU ring buffer                                 */
/* ------------------------------------------------------------------ */

#define MSTOP_HIST_SIZE  16

/*
 * Per-state timing record.  Filled only when mstop_state_tracing is 1.
 * Indexed by enum multi_stop_state value (prev_state from the tracepoint).
 */
struct mstop_state_rec {
	u64	dur_ns;		/* time this CPU spent in this state */
};

struct mstop_entry {
	unsigned long		fn;		/* payload function            */
	unsigned long		caller;		/* stop_machine() call site    */
	int			cpu;
	bool			is_active;	/* ran fn() on this CPU?       */
	unsigned int		num_threads;
	u64			enter_ns;
	u64			exit_ns;	/* 0 = still in progress       */
	int			err;
	bool			states_valid;	/* per-state data was recorded */
	/* states[0..MULTI_STOP_RUN] hold durations; EXIT has none */
	struct mstop_state_rec	states[MULTI_STOP_MAX_STATE];
};

struct mstop_hist {
	struct mstop_entry	ring[MSTOP_HIST_SIZE];
	unsigned int		write_idx;
	unsigned int		curr_slot;
};

static DEFINE_PER_CPU(struct mstop_hist, cpu_mstop_hist);

/*
 * Per-state timing toggle.  Write "1" to multi_stop/state_enable to
 * enable; "0" to disable.  Off by default -- zero overhead when off.
 */
static atomic_t mstop_state_tracing = ATOMIC_INIT(0);

/* ------------------------------------------------------------------ */
/* Per-CPU data                                                         */
/* ------------------------------------------------------------------ */

struct mig_entry {
	unsigned long	caller;		/* _RET_IP_ at enqueue time */
	unsigned long	fn;		/* stop callback address */
	u64		queued_ns;	/* ktime_get_ns() at enqueue */
	u64		start_ns;	/* ktime_get_ns() when execution began */
	u64		end_ns;		/* ktime_get_ns() when done; 0 = running */
};

struct mig_cpu_stats {
	u64		total_dispatched;
	u64		total_exec_ns;
	u64		max_exec_ns;
	unsigned long	max_fn;		/* fn of the slowest item seen */
	unsigned long	max_caller;	/* caller of the slowest item seen */
	u64		min_exec_ns;	/* 0 = no data yet */
};

struct mig_hist {
	struct mig_entry	ring[MIGRATION_HIST_SIZE];
	unsigned int		write_idx;  /* monotonically increasing */
	unsigned int		curr_slot;  /* slot of the currently-running entry */
	struct mig_cpu_stats	stats;
};

static DEFINE_PER_CPU(struct mig_hist, cpu_mig_hist);

/* ------------------------------------------------------------------ */
/* Tracepoint handlers                                                  */
/* ------------------------------------------------------------------ */

static void on_work_begin(void *_unused,
			  unsigned int cpu, unsigned long fn,
			  unsigned long caller, u64 queued_ns, u64 start_ns)
{
	struct mig_hist  *h;
	unsigned int      slot;
	struct mig_entry *e;

	if (!atomic_read(&tracer_enabled))
		return;

	h    = per_cpu_ptr(&cpu_mig_hist, cpu);
	slot = h->write_idx % MIGRATION_HIST_SIZE;
	e    = &h->ring[slot];

	e->fn        = fn;
	e->caller    = caller;
	e->queued_ns = queued_ns;
	e->start_ns  = start_ns;
	e->end_ns    = 0;		/* mark in-progress */

	/*
	 * All entry fields must be visible to NMI readers before write_idx
	 * is incremented (which makes this slot "published").
	 */
	smp_wmb();
	h->curr_slot = slot;
	h->write_idx++;
	h->stats.total_dispatched++;
}

static void on_work_end(void *_unused,
			unsigned int cpu, unsigned long fn,
			unsigned long caller, u64 start_ns, u64 end_ns)
{
	struct mig_hist      *h;
	struct mig_cpu_stats *s;
	u64                   dur;

	if (!atomic_read(&tracer_enabled))
		return;

	h   = per_cpu_ptr(&cpu_mig_hist, cpu);
	s   = &h->stats;
	dur = end_ns - start_ns;

	/* Atomically transition the slot from "running" to "done". */
	WRITE_ONCE(h->ring[h->curr_slot].end_ns, end_ns);

	s->total_exec_ns += dur;

	if (dur > s->max_exec_ns) {
		s->max_exec_ns  = dur;
		s->max_fn       = fn;
		s->max_caller   = caller;
	}
	if (!s->min_exec_ns || dur < s->min_exec_ns)
		s->min_exec_ns = dur;
}

/* ------------------------------------------------------------------ */
/* multi_cpu_stop() tracepoint handlers                                 */
/* ------------------------------------------------------------------ */

static void on_multi_cpu_stop_begin(void *_unused,
				    unsigned int cpu, unsigned long fn,
				    unsigned long caller,
				    unsigned int num_threads,
				    bool is_active, u64 enter_ns)
{
	struct mstop_hist  *h;
	struct mstop_entry *e;
	unsigned int        slot;

	if (!atomic_read(&tracer_enabled))
		return;

	h    = per_cpu_ptr(&cpu_mstop_hist, cpu);
	slot = h->write_idx % MSTOP_HIST_SIZE;
	e    = &h->ring[slot];

	memset(e, 0, sizeof(*e));
	e->fn           = fn;
	e->caller       = caller;
	e->cpu          = cpu;
	e->is_active    = is_active;
	e->num_threads  = num_threads;
	e->enter_ns     = enter_ns;
	e->exit_ns      = 0;
	e->states_valid = !!atomic_read(&mstop_state_tracing);

	smp_wmb();
	h->curr_slot = slot;
	h->write_idx++;
}

static void on_multi_cpu_stop_state(void *_unused,
				    unsigned int cpu, unsigned long fn,
				    int prev_state, int new_state, u64 dur_ns)
{
	struct mstop_hist  *h;
	struct mstop_entry *e;

	if (!atomic_read(&tracer_enabled))
		return;
	if (!atomic_read(&mstop_state_tracing))
		return;
	if (prev_state < 0 || prev_state >= MULTI_STOP_MAX_STATE)
		return;

	h = per_cpu_ptr(&cpu_mstop_hist, cpu);
	e = &h->ring[h->curr_slot];

	if (e->fn == fn)	/* sanity: same invocation */
		e->states[prev_state].dur_ns = dur_ns;
}

static void on_multi_cpu_stop_end(void *_unused,
				  unsigned int cpu, unsigned long fn,
				  unsigned long caller,
				  u64 enter_ns, u64 exit_ns, int err)
{
	struct mstop_hist  *h;
	struct mstop_entry *e;

	if (!atomic_read(&tracer_enabled))
		return;

	h = per_cpu_ptr(&cpu_mstop_hist, cpu);
	e = &h->ring[h->curr_slot];

	if (e->fn == fn && e->enter_ns == enter_ns) {
		e->err = err;
		WRITE_ONCE(e->exit_ns, exit_ns);
	}
}

/* ------------------------------------------------------------------ */
/* Duration formatter                                                   */
/* ------------------------------------------------------------------ */

/*
 * Format @ns nanoseconds into @buf in the most readable unit.
 * Returns the number of bytes written (like scnprintf).
 */
static int fmt_ns(char *buf, size_t sz, u64 ns)
{
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
/* Core dump function (safe from any context including NMI)             */
/* ------------------------------------------------------------------ */

static void dump_migration_history(int cpu)
{
	struct mig_hist *h         = per_cpu_ptr(&cpu_mig_hist, cpu);
	unsigned int     write_idx = READ_ONCE(h->write_idx);
	unsigned int     count     = min(write_idx,
					 (unsigned int)MIGRATION_HIST_SIZE);
	unsigned int     start_i   = write_idx - count;
	u64              now       = ktime_get_ns();
	char             dur_buf[32];
	unsigned int     i;

	pr_emerg("migration/%d: %u total dispatched, dumping last %u:\n",
		 cpu, write_idx, count);

	for (i = start_i; i < write_idx; i++) {
		struct mig_entry *e      = &h->ring[i % MIGRATION_HIST_SIZE];
		unsigned long     fn     = READ_ONCE(e->fn);
		unsigned long     caller = READ_ONCE(e->caller);
		u64               qns    = READ_ONCE(e->queued_ns);
		u64               sns    = READ_ONCE(e->start_ns);
		u64               ens    = READ_ONCE(e->end_ns);

		if (!fn)
			continue;

		if (ens) {
			fmt_ns(dur_buf, sizeof(dur_buf), ens - sns);
			pr_emerg("  [%4u] fn=%pS caller=%pS queued=%llu start=%llu end=%llu dur=%s\n",
				 i, (void *)fn, (void *)caller,
				 qns, sns, ens, dur_buf);
		} else {
			fmt_ns(dur_buf, sizeof(dur_buf), sns ? (now - sns) : 0);
			pr_emerg("  [%4u] fn=%pS caller=%pS queued=%llu start=%llu [RUNNING %s]\n",
				 i, (void *)fn, (void *)caller,
				 qns, sns, dur_buf);
		}
	}
}

/* ------------------------------------------------------------------ */
/* debugfs -- helper: emit one history entry as a single grep-able line  */
/* ------------------------------------------------------------------ */

static void seq_show_entry(struct seq_file *m, int cpu, unsigned int idx,
			   struct mig_entry *e, u64 now)
{
	unsigned long fn     = READ_ONCE(e->fn);
	unsigned long caller = READ_ONCE(e->caller);
	u64           qns    = READ_ONCE(e->queued_ns);
	u64           sns    = READ_ONCE(e->start_ns);
	u64           ens    = READ_ONCE(e->end_ns);
	u64           q2s    = (sns >= qns) ? (sns - qns) : 0;
	char          dur_buf[32], q2s_buf[32];

	if (!fn)
		return;

	fmt_ns(q2s_buf, sizeof(q2s_buf), q2s);

	if (ens) {
		fmt_ns(dur_buf, sizeof(dur_buf), ens - sns);
		seq_printf(m,
			   "cpu=%-3d [%4u] fn=%-40pS caller=%-40pS q2s=%-12s dur=%s\n",
			   cpu, idx,
			   (void *)fn, (void *)caller,
			   q2s_buf, dur_buf);
	} else {
		fmt_ns(dur_buf, sizeof(dur_buf), sns ? (now - sns) : 0);
		seq_printf(m,
			   "cpu=%-3d [%4u] fn=%-40pS caller=%-40pS q2s=%-12s [RUNNING %s]\n",
			   cpu, idx,
			   (void *)fn, (void *)caller,
			   q2s_buf, dur_buf);
	}
}

/* ------------------------------------------------------------------ */
/* debugfs -- history: all CPUs, all ring buffer entries                 */
/* ------------------------------------------------------------------ */

static int history_show(struct seq_file *m, void *v)
{
	u64 now = ktime_get_ns();
	int cpu;

	seq_printf(m,
		   "migration_tracer history  [tracing: %s]  [buf_size: %d entries/cpu]\n"
		   "Fields: cpu  idx  fn  caller  q2s(queue-to-start)  dur(duration or RUNNING)\n"
		   "--------------------------------------------------------------------------------\n",
		   atomic_read(&tracer_enabled) ? "ON" : "OFF",
		   MIGRATION_HIST_SIZE);

	for_each_possible_cpu(cpu) {
		struct mig_hist *h         = per_cpu_ptr(&cpu_mig_hist, cpu);
		unsigned int     write_idx = READ_ONCE(h->write_idx);
		unsigned int     count     = min(write_idx,
						 (unsigned int)MIGRATION_HIST_SIZE);
		unsigned int     start_i   = write_idx - count;
		unsigned int     i;

		seq_printf(m, "--- migration/%d  (%u total dispatched) ---\n",
			   cpu, write_idx);

		if (!count) {
			seq_puts(m, "  (no entries)\n");
			continue;
		}

		for (i = start_i; i < write_idx; i++) {
			struct mig_entry *e = &h->ring[i % MIGRATION_HIST_SIZE];

			seq_show_entry(m, cpu, i, e, now);
		}
	}

	return 0;
}

static int history_open(struct inode *inode, struct file *file)
{
	return single_open(file, history_show, NULL);
}

static const struct file_operations history_fops = {
	.open    = history_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/* ------------------------------------------------------------------ */
/* debugfs -- current: in-flight work items only                         */
/* ------------------------------------------------------------------ */

static int current_inflight_show(struct seq_file *m, void *v)
{
	u64 now   = ktime_get_ns();
	int cpu, found = 0;

	seq_puts(m,
		 "In-flight migration work items (end_ns == 0):\n"
		 "cpu  fn                                          caller"
		 "                                       running\n"
		 "---- ------------------------------------------ ------"
		 "--------------------------------------- ----------\n");

	for_each_possible_cpu(cpu) {
		struct mig_hist  *h    = per_cpu_ptr(&cpu_mig_hist, cpu);
		unsigned int      slot = READ_ONCE(h->curr_slot);
		unsigned long     fn   = READ_ONCE(h->ring[slot].fn);
		unsigned long  caller  = READ_ONCE(h->ring[slot].caller);
		u64             sns    = READ_ONCE(h->ring[slot].start_ns);
		u64             ens    = READ_ONCE(h->ring[slot].end_ns);
		char            dur_buf[32];

		if (!fn || ens)		/* no entry or already finished */
			continue;

		fmt_ns(dur_buf, sizeof(dur_buf), sns ? (now - sns) : 0);
		seq_printf(m, "%-4d %-42pS %-42pS %s\n",
			   cpu, (void *)fn, (void *)caller, dur_buf);
		found++;
	}

	if (!found)
		seq_puts(m, "(none)\n");

	return 0;
}

static int current_inflight_open(struct inode *inode, struct file *file)
{
	return single_open(file, current_inflight_show, NULL);
}

static const struct file_operations current_fops = {
	.open    = current_inflight_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/* ------------------------------------------------------------------ */
/* debugfs -- stats: per-CPU aggregate statistics                        */
/* ------------------------------------------------------------------ */

static int stats_show(struct seq_file *m, void *v)
{
	int cpu;

	seq_printf(m,
		   "migration_tracer per-CPU statistics  [tracing: %s]\n\n"
		   "  cpu  dispatched  avg             max             min\n"
		   "  ---- ----------  --------------- --------------- ---------------\n",
		   atomic_read(&tracer_enabled) ? "ON" : "OFF");

	for_each_possible_cpu(cpu) {
		struct mig_hist      *h     = per_cpu_ptr(&cpu_mig_hist, cpu);
		struct mig_cpu_stats *s     = &h->stats;
		u64  total    = READ_ONCE(s->total_dispatched);
		u64  total_ns = READ_ONCE(s->total_exec_ns);
		u64  max_ns   = READ_ONCE(s->max_exec_ns);
		u64  min_ns   = READ_ONCE(s->min_exec_ns);
		u64  avg_ns   = total ? div64_u64(total_ns, total) : 0;
		unsigned long max_fn     = READ_ONCE(s->max_fn);
		unsigned long max_caller = READ_ONCE(s->max_caller);
		char avg_buf[32], max_buf[32], min_buf[32];

		fmt_ns(avg_buf, sizeof(avg_buf), avg_ns);
		fmt_ns(max_buf, sizeof(max_buf), max_ns);
		fmt_ns(min_buf, sizeof(min_buf), min_ns);

		seq_printf(m, "  %-4d %10llu  %-15s %-15s %s\n",
			   cpu, total, avg_buf, max_buf,
			   min_ns ? min_buf : "n/a");

		if (max_fn)
			seq_printf(m,
				   "       slowest fn:     %pS\n"
				   "       slowest caller: %pS\n",
				   (void *)max_fn, (void *)max_caller);
	}

	return 0;
}

static int stats_open(struct inode *inode, struct file *file)
{
	return single_open(file, stats_show, NULL);
}

static const struct file_operations stats_fops = {
	.open    = stats_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/* ------------------------------------------------------------------ */
/* debugfs -- enable: read/write tracing gate                            */
/* ------------------------------------------------------------------ */

static ssize_t enable_read(struct file *f, char __user *buf,
			   size_t len, loff_t *ppos)
{
	char tmp[4];
	int  n = scnprintf(tmp, sizeof(tmp), "%d\n",
			   atomic_read(&tracer_enabled));

	return simple_read_from_buffer(buf, len, ppos, tmp, n);
}

static ssize_t enable_write(struct file *f, const char __user *buf,
			    size_t len, loff_t *ppos)
{
	char         tmp[8] = {};
	unsigned int val;

	if (copy_from_user(tmp, buf, min(len, sizeof(tmp) - 1)))
		return -EFAULT;
	if (kstrtouint(strim(tmp), 0, &val))
		return -EINVAL;

	atomic_set(&tracer_enabled, val ? 1 : 0);
	pr_info("tracing %s\n", val ? "enabled" : "disabled");
	return len;
}

static const struct file_operations enable_fops = {
	.open  = simple_open,
	.read  = enable_read,
	.write = enable_write,
};

/* ------------------------------------------------------------------ */
/* debugfs -- clear: reset ring buffers and stats on all CPUs            */
/* ------------------------------------------------------------------ */

static ssize_t clear_write(struct file *f, const char __user *buf,
			   size_t len, loff_t *ppos)
{
	int cpu;

	for_each_possible_cpu(cpu) {
		struct mig_hist *h = per_cpu_ptr(&cpu_mig_hist, cpu);

		/*
		 * Zero entries and stats first, then reset the indices.
		 * The ordering ensures a concurrent NMI reader never sees
		 * a zero write_idx with stale entries still referenced.
		 */
		memset(h->ring,   0, sizeof(h->ring));
		memset(&h->stats, 0, sizeof(h->stats));
		smp_wmb();
		WRITE_ONCE(h->curr_slot, 0);
		WRITE_ONCE(h->write_idx, 0);
	}

	pr_info("ring buffers and stats cleared on all CPUs\n");
	return len;
}

static const struct file_operations clear_fops = {
	.open  = simple_open,
	.write = clear_write,
};

/* ------------------------------------------------------------------ */
/* debugfs -- multi_stop: helpers                                        */
/* ------------------------------------------------------------------ */

static const char * const mstop_state_name[MULTI_STOP_MAX_STATE] = {
	[MULTI_STOP_NONE]        = "NONE",
	[MULTI_STOP_PREPARE]     = "PREPARE",
	[MULTI_STOP_DISABLE_IRQ] = "DISABLE_IRQ",
	[MULTI_STOP_RUN]         = "RUN",
	[MULTI_STOP_EXIT]        = "EXIT",
};

static void seq_show_mstop_entry(struct seq_file *m, struct mstop_entry *e,
				 u64 now)
{
	char total_buf[32];

	if (!e->fn)
		return;

	if (e->exit_ns) {
		fmt_ns(total_buf, sizeof(total_buf), e->exit_ns - e->enter_ns);
		seq_printf(m, "  cpu=%-3d fn=%-40pS caller=%-40pS active=%-5s threads=%-3u dur=%s err=%d\n",
			   e->cpu, (void *)e->fn, (void *)e->caller,
			   e->is_active ? "yes" : "no",
			   e->num_threads, total_buf, e->err);
	} else {
		fmt_ns(total_buf, sizeof(total_buf), now - e->enter_ns);
		seq_printf(m, "  cpu=%-3d fn=%-40pS caller=%-40pS active=%-5s threads=%-3u [RUNNING %s]\n",
			   e->cpu, (void *)e->fn, (void *)e->caller,
			   e->is_active ? "yes" : "no",
			   e->num_threads, total_buf);
	}

	if (e->states_valid) {
		int s;

		seq_puts(m, "         state breakdown:");
		for (s = 0; s < MULTI_STOP_MAX_STATE - 1; s++) {
			char sbuf[32];

			fmt_ns(sbuf, sizeof(sbuf), e->states[s].dur_ns);
			seq_printf(m, "  %s=%s", mstop_state_name[s], sbuf);
		}
		seq_putc(m, '\n');
	}
}

/* ------------------------------------------------------------------ */
/* debugfs -- multi_stop/history                                         */
/* ------------------------------------------------------------------ */

static int mstop_history_show(struct seq_file *m, void *v)
{
	u64 now = ktime_get_ns();
	int cpu;

	seq_printf(m,
		   "multi_cpu_stop history  [tracing: %s]  [state_tracing: %s]  [buf: %d/cpu]\n"
		   "Fields: cpu  fn  caller  active  threads  duration  [state breakdown]\n"
		   "------------------------------------------------------------------------\n",
		   atomic_read(&tracer_enabled)      ? "ON" : "OFF",
		   atomic_read(&mstop_state_tracing) ? "ON" : "OFF",
		   MSTOP_HIST_SIZE);

	for_each_possible_cpu(cpu) {
		struct mstop_hist *h         = per_cpu_ptr(&cpu_mstop_hist, cpu);
		unsigned int       write_idx = READ_ONCE(h->write_idx);
		unsigned int       count     = min(write_idx, (unsigned int)MSTOP_HIST_SIZE);
		unsigned int       start_i   = write_idx - count;
		unsigned int       i;

		seq_printf(m, "--- CPU %d  (%u total) ---\n", cpu, write_idx);
		if (!count) {
			seq_puts(m, "  (no entries)\n");
			continue;
		}
		for (i = start_i; i < write_idx; i++)
			seq_show_mstop_entry(m, &h->ring[i % MSTOP_HIST_SIZE], now);
	}
	return 0;
}

static int mstop_history_open(struct inode *inode, struct file *file)
{
	return single_open(file, mstop_history_show, NULL);
}

static const struct file_operations mstop_history_fops = {
	.open    = mstop_history_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/* ------------------------------------------------------------------ */
/* debugfs -- multi_stop/current                                         */
/* ------------------------------------------------------------------ */

static int mstop_current_show(struct seq_file *m, void *v)
{
	u64 now   = ktime_get_ns();
	int cpu, found = 0;

	seq_puts(m, "In-flight multi_cpu_stop invocations:\n");

	for_each_possible_cpu(cpu) {
		struct mstop_hist  *h    = per_cpu_ptr(&cpu_mstop_hist, cpu);
		unsigned int        slot = READ_ONCE(h->curr_slot);
		struct mstop_entry *e    = &h->ring[slot];

		if (!e->fn || READ_ONCE(e->exit_ns))
			continue;

		seq_show_mstop_entry(m, e, now);
		found++;
	}

	if (!found)
		seq_puts(m, "  (none)\n");
	return 0;
}

static int mstop_current_open(struct inode *inode, struct file *file)
{
	return single_open(file, mstop_current_show, NULL);
}

static const struct file_operations mstop_current_fops = {
	.open    = mstop_current_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/* ------------------------------------------------------------------ */
/* debugfs -- multi_stop/state_enable                                    */
/* ------------------------------------------------------------------ */

static ssize_t mstop_state_enable_read(struct file *f, char __user *buf,
				       size_t len, loff_t *ppos)
{
	char tmp[4];
	int  n = scnprintf(tmp, sizeof(tmp), "%d\n",
			   atomic_read(&mstop_state_tracing));

	return simple_read_from_buffer(buf, len, ppos, tmp, n);
}

static ssize_t mstop_state_enable_write(struct file *f, const char __user *buf,
					size_t len, loff_t *ppos)
{
	char         tmp[8] = {};
	unsigned int val;

	if (copy_from_user(tmp, buf, min(len, sizeof(tmp) - 1)))
		return -EFAULT;
	if (kstrtouint(strim(tmp), 0, &val))
		return -EINVAL;

	atomic_set(&mstop_state_tracing, val ? 1 : 0);
	pr_info("multi_cpu_stop state tracing %s\n", val ? "enabled" : "disabled");
	return len;
}

static const struct file_operations mstop_state_enable_fops = {
	.open  = simple_open,
	.read  = mstop_state_enable_read,
	.write = mstop_state_enable_write,
};

/* ------------------------------------------------------------------ */
/* debugfs -- multi_stop/states (per-state breakdown table)              */
/* ------------------------------------------------------------------ */

static int mstop_states_show(struct seq_file *m, void *v)
{
	int cpu;

	if (!atomic_read(&mstop_state_tracing)) {
		seq_puts(m,
			 "# State tracing is OFF.\n"
			 "# Enable it: echo 1 > multi_stop/state_enable\n"
			 "# (New records will have per-state data.)\n");
	}

	seq_printf(m,
		   "# %-4s  %-4s  %-8s  %-14s  %-14s  %-14s  %-14s  fn\n",
		   "CPU", "idx",
		   "total",
		   mstop_state_name[MULTI_STOP_NONE],
		   mstop_state_name[MULTI_STOP_PREPARE],
		   mstop_state_name[MULTI_STOP_DISABLE_IRQ],
		   mstop_state_name[MULTI_STOP_RUN]);

	for_each_possible_cpu(cpu) {
		struct mstop_hist *h         = per_cpu_ptr(&cpu_mstop_hist, cpu);
		unsigned int       write_idx = READ_ONCE(h->write_idx);
		unsigned int       count     = min(write_idx, (unsigned int)MSTOP_HIST_SIZE);
		unsigned int       start_i   = write_idx - count;
		unsigned int       i;

		for (i = start_i; i < write_idx; i++) {
			struct mstop_entry *e = &h->ring[i % MSTOP_HIST_SIZE];
			char tot[32], s0[32], s1[32], s2[32], s3[32];

			if (!e->fn)
				continue;

			if (!e->states_valid) {
				seq_printf(m, "  %-4d  %-4u  (no state data)\n",
					   cpu, i);
				continue;
			}

			fmt_ns(tot, sizeof(tot),
			       e->exit_ns ? e->exit_ns - e->enter_ns : 0);
			fmt_ns(s0,  sizeof(s0),  e->states[MULTI_STOP_NONE].dur_ns);
			fmt_ns(s1,  sizeof(s1),  e->states[MULTI_STOP_PREPARE].dur_ns);
			fmt_ns(s2,  sizeof(s2),  e->states[MULTI_STOP_DISABLE_IRQ].dur_ns);
			fmt_ns(s3,  sizeof(s3),  e->states[MULTI_STOP_RUN].dur_ns);

			seq_printf(m, "  %-4d  %-4u  %-8s  %-14s  %-14s  %-14s  %-14s  %pS\n",
				   cpu, i, tot, s0, s1, s2, s3,
				   (void *)e->fn);
		}
	}
	return 0;
}

static int mstop_states_open(struct inode *inode, struct file *file)
{
	return single_open(file, mstop_states_show, NULL);
}

static const struct file_operations mstop_states_fops = {
	.open    = mstop_states_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/* ------------------------------------------------------------------ */
/* debugfs -- multi_stop/clear                                           */
/* ------------------------------------------------------------------ */

static ssize_t mstop_clear_write(struct file *f, const char __user *buf,
				 size_t len, loff_t *ppos)
{
	int cpu;

	for_each_possible_cpu(cpu) {
		struct mstop_hist *h = per_cpu_ptr(&cpu_mstop_hist, cpu);

		memset(h->ring, 0, sizeof(h->ring));
		smp_wmb();
		WRITE_ONCE(h->curr_slot, 0);
		WRITE_ONCE(h->write_idx, 0);
	}
	pr_info("multi_cpu_stop ring buffers cleared on all CPUs\n");
	return len;
}

static const struct file_operations mstop_clear_fops = {
	.open  = simple_open,
	.write = mstop_clear_write,
};

/* ------------------------------------------------------------------ */
/* debugfs setup / teardown                                             */
/* ------------------------------------------------------------------ */

static struct dentry *tracer_dentry;

static int debugfs_setup(void)
{
	struct dentry *mstop_dir;

	tracer_dentry = debugfs_create_dir("migration_tracer", NULL);
	if (IS_ERR(tracer_dentry))
		return PTR_ERR(tracer_dentry);

	debugfs_create_file("enable",  0644, tracer_dentry, NULL, &enable_fops);
	debugfs_create_file("history", 0444, tracer_dentry, NULL, &history_fops);
	debugfs_create_file("current", 0444, tracer_dentry, NULL, &current_fops);
	debugfs_create_file("stats",   0444, tracer_dentry, NULL, &stats_fops);
	debugfs_create_file("clear",   0200, tracer_dentry, NULL, &clear_fops);

	/* multi_cpu_stop sub-directory */
	mstop_dir = debugfs_create_dir("multi_stop", tracer_dentry);
	debugfs_create_file("history",      0444, mstop_dir, NULL, &mstop_history_fops);
	debugfs_create_file("current",      0444, mstop_dir, NULL, &mstop_current_fops);
	debugfs_create_file("state_enable", 0644, mstop_dir, NULL, &mstop_state_enable_fops);
	debugfs_create_file("states",       0444, mstop_dir, NULL, &mstop_states_fops);
	debugfs_create_file("clear",        0200, mstop_dir, NULL, &mstop_clear_fops);

	pr_info("debugfs: /sys/kernel/debug/migration_tracer/\n");
	return 0;
}

static void debugfs_teardown(void)
{
	debugfs_remove_recursive(tracer_dentry);
	tracer_dentry = NULL;
}

/* ------------------------------------------------------------------ */
/* QCOM watchdog bark: kprobe on watchdog_notify_pretimeout()           */
/* ------------------------------------------------------------------ */

/*
 * Prototype: void watchdog_notify_pretimeout(struct watchdog_device *wdd)
 *
 * Called from qcom_wdt_isr() / pm8916_wdt_isr() when the QCOM hardware
 * bark fires.  We dump the migration history for every CPU here -- before
 * the pretimeout governor has a chance to call panic() -- so the full
 * history is captured in the kernel log while the system is still alive.
 *
 * Context: interrupt (IRQ) context, not NMI.
 */
static int qcom_bark_pre(struct kprobe *p, struct pt_regs *regs)
{
	int cpu;

	pr_emerg("QCOM watchdog bark -- dumping migration thread history for all CPUs:\n");

	for_each_possible_cpu(cpu)
		dump_migration_history(cpu);

	return 0;
}

static struct kprobe bark_kp = {
	.symbol_name = "watchdog_notify_pretimeout",
	.pre_handler = qcom_bark_pre,
};

/* ------------------------------------------------------------------ */
/* Panic notifier -- backup path                                         */
/* ------------------------------------------------------------------ */

/*
 * Fires early inside panic() regardless of what triggered it.  We only
 * print CPUs that have a migration work item currently stuck (end_ns == 0)
 * so the output is focused when the panic has an unrelated cause.
 *
 * Priority INT_MAX ensures this runs before any other panic notifier,
 * maximising the chance the history is captured before buffers flush.
 */
static int migration_panic_notifier(struct notifier_block *nb,
				    unsigned long action, void *data)
{
	int  cpu, found = 0;
	u64  now = ktime_get_ns();
	char dur_buf[32];

	for_each_possible_cpu(cpu) {
		struct mig_hist  *h    = per_cpu_ptr(&cpu_mig_hist, cpu);
		unsigned int      slot = READ_ONCE(h->curr_slot);
		unsigned long     fn   = READ_ONCE(h->ring[slot].fn);
		unsigned long  caller  = READ_ONCE(h->ring[slot].caller);
		u64             sns    = READ_ONCE(h->ring[slot].start_ns);
		u64             ens    = READ_ONCE(h->ring[slot].end_ns);

		if (!fn || ens)		/* nothing running on this CPU */
			continue;

		if (!found)
			pr_emerg("migration_tracer: stuck work items at panic time:\n");

		fmt_ns(dur_buf, sizeof(dur_buf), sns ? (now - sns) : 0);
		pr_emerg("  cpu=%-3d fn=%pS caller=%pS [RUNNING %s]\n",
			 cpu, (void *)fn, (void *)caller, dur_buf);

		/* Also dump the full ring for this CPU. */
		dump_migration_history(cpu);
		found++;
	}

	return NOTIFY_OK;
}

static struct notifier_block panic_nb = {
	.notifier_call = migration_panic_notifier,
	.priority      = INT_MAX,
};

/* ------------------------------------------------------------------ */
/* Module lifecycle                                                     */
/* ------------------------------------------------------------------ */

static int __init migration_tracer_init(void)
{
	int ret;

	ret = debugfs_setup();
	if (ret) {
		pr_err("debugfs setup failed: %d\n", ret);
		return ret;
	}

	ret = register_trace_cpu_stop_work_begin(on_work_begin, NULL);
	if (ret) {
		pr_err("register cpu_stop_work_begin failed: %d\n", ret);
		goto err_debugfs;
	}

	ret = register_trace_cpu_stop_work_end(on_work_end, NULL);
	if (ret) {
		pr_err("register cpu_stop_work_end failed: %d\n", ret);
		goto err_unreg_begin;
	}

	ret = register_trace_multi_cpu_stop_begin(on_multi_cpu_stop_begin, NULL);
	if (ret) {
		pr_err("register multi_cpu_stop_begin failed: %d\n", ret);
		goto err_unreg_work_end;
	}

	ret = register_trace_multi_cpu_stop_state(on_multi_cpu_stop_state, NULL);
	if (ret) {
		pr_err("register multi_cpu_stop_state failed: %d\n", ret);
		goto err_unreg_mstop_begin;
	}

	ret = register_trace_multi_cpu_stop_end(on_multi_cpu_stop_end, NULL);
	if (ret) {
		pr_err("register multi_cpu_stop_end failed: %d\n", ret);
		goto err_unreg_mstop_state;
	}

	/*
	 * Kprobe on watchdog_notify_pretimeout() catches the QCOM bark event
	 * before the governor runs.  Non-fatal: if unavailable the panic
	 * notifier still covers the watchdog -> panic path.
	 */
	ret = register_kprobe(&bark_kp);
	if (ret)
		pr_warn("kprobe on watchdog_notify_pretimeout unavailable (%d) "
			"-- QCOM bark dump disabled, panic path still active\n",
			ret);

	atomic_notifier_chain_register(&panic_notifier_list, &panic_nb);

	pr_info("loaded: %d-entry ring buffer/CPU | QCOM bark kprobe %s | panic notifier active\n",
		MIGRATION_HIST_SIZE,
		bark_kp.addr ? "active" : "inactive");
	return 0;

err_unreg_mstop_state:
	unregister_trace_multi_cpu_stop_state(on_multi_cpu_stop_state, NULL);
err_unreg_mstop_begin:
	unregister_trace_multi_cpu_stop_begin(on_multi_cpu_stop_begin, NULL);
err_unreg_work_end:
	unregister_trace_cpu_stop_work_end(on_work_end, NULL);
err_unreg_begin:
	unregister_trace_cpu_stop_work_begin(on_work_begin, NULL);
err_debugfs:
	debugfs_teardown();
	return ret;
}
module_init(migration_tracer_init);

static void __exit migration_tracer_exit(void)
{
	atomic_notifier_chain_unregister(&panic_notifier_list, &panic_nb);
	unregister_kprobe(&bark_kp);
	unregister_trace_multi_cpu_stop_end(on_multi_cpu_stop_end, NULL);
	unregister_trace_multi_cpu_stop_state(on_multi_cpu_stop_state, NULL);
	unregister_trace_multi_cpu_stop_begin(on_multi_cpu_stop_begin, NULL);
	unregister_trace_cpu_stop_work_end(on_work_end, NULL);
	unregister_trace_cpu_stop_work_begin(on_work_begin, NULL);
	/*
	 * Ensure all in-flight trace callbacks complete before the module
	 * text is unmapped.  Must happen before debugfs_teardown() so no
	 * new callbacks run after the dentry tree is removed.
	 */
	tracepoint_synchronize_unregister();
	debugfs_teardown();
	pr_info("unloaded\n");
}
module_exit(migration_tracer_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Migration/stopper thread tracer for QCOM watchdog platforms");
