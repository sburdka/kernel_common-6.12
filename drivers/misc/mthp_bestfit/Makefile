# SPDX-License-Identifier: GPL-2.0
# In-tree build: parent Makefile descends here when CONFIG_MTHP_BESTFIT=m
# Standalone external module build:
#   make KDIR=/path/to/gki-6.12/out ARCH=arm64 LLVM=1
#
# No KBUILD_EXTRA_SYMBOLS needed: mthp_monitor_get_sysinfo is resolved at
# runtime via symbol_get(), so mthp_bestfit builds independently.

obj-$(CONFIG_MTHP_BESTFIT) += mthp_bestfit.o

# Standalone targets (active only when invoked directly, not via Kbuild)
KDIR ?= /lib/modules/$(shell uname -r)/build
ARCH ?= arm64
LLVM ?= 1

standalone:
	$(MAKE) -C $(KDIR) M=$(PWD) ARCH=$(ARCH) LLVM=$(LLVM) \
		CONFIG_MTHP_BESTFIT=m modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) ARCH=$(ARCH) LLVM=$(LLVM) clean
