/* SPDX-License-Identifier: GPL-2.0 */
#undef TRACE_SYSTEM
#define TRACE_SYSTEM stopmachine

#if !defined(_TRACE_STOPMACHINE_H) || defined(TRACE_HEADER_MULTI_READ)
#define _TRACE_STOPMACHINE_H

#include <linux/tracepoint.h>

/**
 * cpu_stop_work_begin - fired just before cpu_stopper_thread() calls fn(arg)
 *
 * @cpu:       CPU number the migration/stopper thread runs on
 * @fn:        address of the stop callback about to execute
 * @caller:    _RET_IP_ captured when the work was enqueued
 * @queued_ns: ktime_get_ns() at enqueue time
 * @start_ns:  ktime_get_ns() at the start of this execution
 */
TRACE_EVENT(cpu_stop_work_begin,

	TP_PROTO(unsigned int cpu, unsigned long fn,
		 unsigned long caller, u64 queued_ns, u64 start_ns),

	TP_ARGS(cpu, fn, caller, queued_ns, start_ns),

	TP_STRUCT__entry(
		__field(unsigned int,  cpu)
		__field(unsigned long, fn)
		__field(unsigned long, caller)
		__field(u64,           queued_ns)
		__field(u64,           start_ns)
	),

	TP_fast_assign(
		__entry->cpu       = cpu;
		__entry->fn        = fn;
		__entry->caller    = caller;
		__entry->queued_ns = queued_ns;
		__entry->start_ns  = start_ns;
	),

	TP_printk("cpu=%u fn=%pS caller=%pS queued_ns=%llu start_ns=%llu",
		  __entry->cpu,
		  (void *)__entry->fn, (void *)__entry->caller,
		  __entry->queued_ns, __entry->start_ns)
);

/**
 * cpu_stop_work_end - fired just after fn(arg) returns
 *
 * @cpu:      CPU number the migration/stopper thread runs on
 * @fn:       address of the stop callback that just returned
 * @caller:   _RET_IP_ captured when the work was enqueued
 * @start_ns: must match the start_ns from the paired cpu_stop_work_begin event
 * @end_ns:   ktime_get_ns() after fn returns
 */
TRACE_EVENT(cpu_stop_work_end,

	TP_PROTO(unsigned int cpu, unsigned long fn,
		 unsigned long caller, u64 start_ns, u64 end_ns),

	TP_ARGS(cpu, fn, caller, start_ns, end_ns),

	TP_STRUCT__entry(
		__field(unsigned int,  cpu)
		__field(unsigned long, fn)
		__field(unsigned long, caller)
		__field(u64,           start_ns)
		__field(u64,           end_ns)
	),

	TP_fast_assign(
		__entry->cpu      = cpu;
		__entry->fn       = fn;
		__entry->caller   = caller;
		__entry->start_ns = start_ns;
		__entry->end_ns   = end_ns;
	),

	TP_printk("cpu=%u fn=%pS caller=%pS start_ns=%llu end_ns=%llu dur_ns=%llu",
		  __entry->cpu,
		  (void *)__entry->fn, (void *)__entry->caller,
		  __entry->start_ns, __entry->end_ns,
		  __entry->end_ns - __entry->start_ns)
);

/**
 * multi_cpu_stop_begin - fired at the very start of multi_cpu_stop()
 *
 * @cpu:         CPU running this instance
 * @fn:          address of the payload function (msdata->fn)
 * @caller:      _RET_IP_ captured at the stop_machine()/stop_two_cpus() call site
 * @num_threads: number of CPUs participating in this stop
 * @is_active:   true if this CPU will execute fn()
 * @enter_ns:    ktime_get_ns() at function entry
 */
TRACE_EVENT(multi_cpu_stop_begin,

	TP_PROTO(unsigned int cpu, unsigned long fn, unsigned long caller,
		 unsigned int num_threads, bool is_active, u64 enter_ns),

	TP_ARGS(cpu, fn, caller, num_threads, is_active, enter_ns),

	TP_STRUCT__entry(
		__field(unsigned int,  cpu)
		__field(unsigned long, fn)
		__field(unsigned long, caller)
		__field(unsigned int,  num_threads)
		__field(bool,          is_active)
		__field(u64,           enter_ns)
	),

	TP_fast_assign(
		__entry->cpu         = cpu;
		__entry->fn          = fn;
		__entry->caller      = caller;
		__entry->num_threads = num_threads;
		__entry->is_active   = is_active;
		__entry->enter_ns    = enter_ns;
	),

	TP_printk("cpu=%u fn=%pS caller=%pS threads=%u active=%d enter_ns=%llu",
		  __entry->cpu,
		  (void *)__entry->fn, (void *)__entry->caller,
		  __entry->num_threads, __entry->is_active,
		  __entry->enter_ns)
);

/**
 * multi_cpu_stop_state - fired at each state transition inside multi_cpu_stop()
 *
 * @cpu:        CPU observing the transition
 * @fn:         payload function address
 * @prev_state: state being left (enum multi_stop_state value)
 * @new_state:  state being entered
 * @dur_ns:     nanoseconds this CPU spent in prev_state
 */
TRACE_EVENT(multi_cpu_stop_state,

	TP_PROTO(unsigned int cpu, unsigned long fn,
		 int prev_state, int new_state, u64 dur_ns),

	TP_ARGS(cpu, fn, prev_state, new_state, dur_ns),

	TP_STRUCT__entry(
		__field(unsigned int,  cpu)
		__field(unsigned long, fn)
		__field(int,           prev_state)
		__field(int,           new_state)
		__field(u64,           dur_ns)
	),

	TP_fast_assign(
		__entry->cpu        = cpu;
		__entry->fn         = fn;
		__entry->prev_state = prev_state;
		__entry->new_state  = new_state;
		__entry->dur_ns     = dur_ns;
	),

	TP_printk("cpu=%u fn=%pS %d->%d dur_ns=%llu",
		  __entry->cpu, (void *)__entry->fn,
		  __entry->prev_state, __entry->new_state,
		  __entry->dur_ns)
);

/**
 * multi_cpu_stop_end - fired just before multi_cpu_stop() returns
 *
 * @cpu:      CPU that ran this instance
 * @fn:       payload function address
 * @caller:   same caller as in the paired multi_cpu_stop_begin event
 * @enter_ns: same enter_ns as in the paired multi_cpu_stop_begin event
 * @exit_ns:  ktime_get_ns() after the state machine exits
 * @err:      return value from fn() on the active CPU, 0 on all others
 */
TRACE_EVENT(multi_cpu_stop_end,

	TP_PROTO(unsigned int cpu, unsigned long fn, unsigned long caller,
		 u64 enter_ns, u64 exit_ns, int err),

	TP_ARGS(cpu, fn, caller, enter_ns, exit_ns, err),

	TP_STRUCT__entry(
		__field(unsigned int,  cpu)
		__field(unsigned long, fn)
		__field(unsigned long, caller)
		__field(u64,           enter_ns)
		__field(u64,           exit_ns)
		__field(int,           err)
	),

	TP_fast_assign(
		__entry->cpu      = cpu;
		__entry->fn       = fn;
		__entry->caller   = caller;
		__entry->enter_ns = enter_ns;
		__entry->exit_ns  = exit_ns;
		__entry->err      = err;
	),

	TP_printk("cpu=%u fn=%pS caller=%pS enter_ns=%llu exit_ns=%llu dur_ns=%llu err=%d",
		  __entry->cpu,
		  (void *)__entry->fn, (void *)__entry->caller,
		  __entry->enter_ns, __entry->exit_ns,
		  __entry->exit_ns - __entry->enter_ns,
		  __entry->err)
);

#endif /* _TRACE_STOPMACHINE_H */

/* This part must be outside the header guard */
#include <trace/define_trace.h>
