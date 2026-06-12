# SPDX-License-Identifier: GPL-2.0
# In-tree build: parent Makefile descends here when CONFIG_MTHP_MONITOR=m
# Standalone external module build:
#   make KDIR=/path/to/gki-6.12/out ARCH=arm64 LLVM=1

obj-$(CONFIG_MTHP_MONITOR) += mthp_monitor.o

# Standalone targets (active only when invoked directly, not via Kbuild)
KDIR ?= /lib/modules/$(shell uname -r)/build
ARCH ?= arm64
LLVM ?= 1

standalone:
	$(MAKE) -C $(KDIR) M=$(PWD) ARCH=$(ARCH) LLVM=$(LLVM) \
		CONFIG_MTHP_MONITOR=m modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) ARCH=$(ARCH) LLVM=$(LLVM) clean
