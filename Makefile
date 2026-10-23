# Cargo remains the build system; this frontend supplies saved build options.
.DEFAULT_GOAL := build

CARGO ?= cargo
RUSTC ?= rustc
CONFIG ?= .config
CONFIG_TOOL := target/baldes-build
quote = '$(subst ','"'"',$(1))'
RUN_CONFIG = $(CONFIG_TOOL) --config $(call quote,$(CONFIG)) --cargo $(call quote,$(CARGO))

.PHONY: all build config menuconfig defconfig showconfig print-command check test clippy fmt clean test-config help

all: build

$(CONFIG_TOOL): scripts/build/build_config.rs scripts/build/build_config_ui.rs
	@mkdir -p target
	$(call quote,$(RUSTC)) --edition=2021 -O $< -o $@

config menuconfig: $(CONFIG_TOOL)
	@$(RUN_CONFIG) menu

defconfig: $(CONFIG_TOOL)
	@$(RUN_CONFIG) defaults

showconfig: $(CONFIG_TOOL)
	@$(RUN_CONFIG) show

print-command: $(CONFIG_TOOL)
	@$(RUN_CONFIG) command

build check test clippy clean: $(CONFIG_TOOL)
	@$(RUN_CONFIG) $@

fmt:
	$(call quote,$(CARGO)) fmt -p baldes --check

test-config: $(CONFIG_TOOL)
	$(call quote,$(RUSTC)) --edition=2021 --test scripts/build/build_config.rs -o target/baldes-build-tests
	@target/baldes-build-tests

help:
	@printf '%s\n' \
	  'make                  Build the configured library and optional CLI' \
	  'make config           Open the terminal configuration window' \
	  'make menuconfig       Alias for make config' \
	  'make defconfig        Save defaults (replaces the selected config)' \
	  'make showconfig       Show effective settings' \
	  'make print-command    Preview the configured Cargo build command' \
	  'make check/test       Check or test with the selected build settings' \
	  'make clippy/fmt       Lint selected targets / check Rust formatting' \
	  'make clean            Cargo clean for the baldes package' \
	  'make test-config      Test configuration parsing and Cargo translation' \
	  'CONFIG=path           Use a configuration file other than .config' \
	  'CARGO=path RUSTC=path Override tool executables'
