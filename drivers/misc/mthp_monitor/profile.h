/* SPDX-License-Identifier: GPL-2.0 */
/*
 * profile.h - shared monitor <-> mthp_bestfit interface
 *
 * NOTE: this header was not in the OCR dump; it is reconstructed from
 * usage in mthp_monitor.c. The struct layout is ABI between the two
 * modules — it must match mthp_bestfit's copy field-for-field.
 */
#ifndef _MTHP_PROFILE_H
#define _MTHP_PROFILE_H

#include <linux/types.h>

/* Buddy orders 0..MAX_PAGE_ORDER (10) tracked */
#define MAX_TRACKED_ORDER	11

struct system_info {
	/*
	 * Free blocks at each order. INT_MAX is the "data unavailable"
	 * sentinel: it keeps mthp_bestfit's buddy_guard permanently
	 * satisfied (free_at >= guard_min) when per-order data cannot
	 * be read on this kernel.
	 */
	u64		buddy_free[MAX_TRACKED_ORDER];
	unsigned long	total_pages;
	unsigned long	mem_available;	/* si_mem_available() snapshot */
	unsigned int	frag_index;	/* (total - free) * 100 / total */
};

int mthp_monitor_get_sysinfo(struct system_info *out);

#endif /* _MTHP_PROFILE_H */
