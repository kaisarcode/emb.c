## Makefile
## Summary: Cross-compilation builder for emb.
##
## Author:  KaisarCode
## Website: https://kaisarcode.com
## License: https://www.gnu.org/licenses/gpl-3.0.html

ANDROID_HOME  ?= $(HOME)/.local/share/android-sdk
NDK_VERSION   ?= 27.2.12479018
NDK_DIR       := $(ANDROID_HOME)/ndk/$(NDK_VERSION)
NDK_TOOLCHAIN := $(NDK_DIR)/build/cmake/android.toolchain.cmake

BUILD_DIR := .build
BIN_DIR   := bin

.PHONY: all test clean \
	x86_64/linux x86_64/windows \
	i686/linux i686/windows \
	aarch64/linux aarch64/android \
	armv7/linux armv7/android \
	armv7hf/linux \
	riscv64/linux \
	powerpc64le/linux \
	mips/linux mipsel/linux mips64el/linux \
	s390x/linux \
	loongarch64/linux

all: \
	x86_64/linux x86_64/windows \
	i686/linux i686/windows \
	aarch64/linux aarch64/android \
	armv7/linux armv7/android \
	armv7hf/linux \
	riscv64/linux \
	powerpc64le/linux \
	mips/linux mipsel/linux mips64el/linux \
	s390x/linux \
	loongarch64/linux

## ── Linux ────────────────────────────────────────────────────────────────────
## $(1)=target  $(2)=processor  $(3)=CC  $(4)=CXX

define linux_target
	@mkdir -p $(BIN_DIR)/$(1)/linux
	@cmake -S . -B $(BUILD_DIR)/$(subst /,-,$(1))-linux \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_SYSTEM_NAME=Linux \
		-DCMAKE_SYSTEM_PROCESSOR=$(2) \
		-DCMAKE_C_COMPILER=$(3) \
		-DCMAKE_CXX_COMPILER=$(4) \
		-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(CURDIR)/$(BUILD_DIR)/$(subst /,-,$(1))-linux/out \
		-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/linux \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/linux \
		-G Ninja -Wno-dev > /dev/null
	@cmake --build $(BUILD_DIR)/$(subst /,-,$(1))-linux
	@cp $(BUILD_DIR)/$(subst /,-,$(1))-linux/out/emb $(BIN_DIR)/$(1)/linux/emb
	@echo "OK $(1)/linux"
endef

x86_64/linux:
	$(call linux_target,x86_64,x86_64,gcc,g++)

i686/linux:
	$(call linux_target,i686,i686,i686-linux-gnu-gcc,i686-linux-gnu-g++)

aarch64/linux:
	$(call linux_target,aarch64,aarch64,aarch64-linux-gnu-gcc,aarch64-linux-gnu-g++)

armv7hf/linux:
	$(call linux_target,armv7hf,armv7,arm-linux-gnueabihf-gcc,arm-linux-gnueabihf-g++)

armv7/linux:
	$(call linux_target,armv7,armv7,arm-linux-gnueabi-gcc,arm-linux-gnueabi-g++)

riscv64/linux:
	$(call linux_target,riscv64,riscv64,riscv64-linux-gnu-gcc,riscv64-linux-gnu-g++)

powerpc64le/linux:
	$(call linux_target,powerpc64le,powerpc64le,powerpc64le-linux-gnu-gcc,powerpc64le-linux-gnu-g++)

mips/linux:
	$(call linux_target,mips,mips,mips-linux-gnu-gcc,mips-linux-gnu-g++)

mipsel/linux:
	$(call linux_target,mipsel,mipsel,mipsel-linux-gnu-gcc,mipsel-linux-gnu-g++)

mips64el/linux:
	$(call linux_target,mips64el,mips64el,mips64el-linux-gnuabi64-gcc,mips64el-linux-gnuabi64-g++)

s390x/linux:
	$(call linux_target,s390x,s390x,s390x-linux-gnu-gcc,s390x-linux-gnu-g++)

loongarch64/linux:
	$(call linux_target,loongarch64,loongarch64,loongarch64-linux-gnu-gcc,loongarch64-linux-gnu-g++)

## ── Windows ──────────────────────────────────────────────────────────────────
## $(1)=target  $(2)=processor  $(3)=CC  $(4)=CXX

define windows_target
	@mkdir -p $(BIN_DIR)/$(1)/windows
	@cmake -S . -B $(BUILD_DIR)/$(1)-windows \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_SYSTEM_NAME=Windows \
		-DCMAKE_SYSTEM_PROCESSOR=$(2) \
		-DCMAKE_C_COMPILER=$(3) \
		-DCMAKE_CXX_COMPILER=$(4) \
		-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(CURDIR)/$(BUILD_DIR)/$(1)-windows/out \
		-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/windows \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/windows \
		-G Ninja -Wno-dev > /dev/null
	@cmake --build $(BUILD_DIR)/$(1)-windows
	@cp $(BUILD_DIR)/$(1)-windows/out/emb.exe $(BIN_DIR)/$(1)/windows/emb.exe
	@cp $(BUILD_DIR)/$(1)-windows/out/libemb.dll $(BIN_DIR)/$(1)/windows/libemb.dll
	@echo "OK $(1)/windows"
endef

x86_64/windows:
	$(call windows_target,x86_64,x86_64,x86_64-w64-mingw32-gcc,x86_64-w64-mingw32-g++)

i686/windows:
	$(call windows_target,i686,i686,i686-w64-mingw32-gcc,i686-w64-mingw32-g++)

## ── Android ──────────────────────────────────────────────────────────────────

define android_target
	@mkdir -p $(BIN_DIR)/$(1)/android
	@cmake -S . -B $(BUILD_DIR)/$(1)-android \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_TOOLCHAIN_FILE=$(NDK_TOOLCHAIN) \
		-DANDROID_ABI=$(2) \
		-DANDROID_PLATFORM=android-21 \
		-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(CURDIR)/$(BUILD_DIR)/$(1)-android/out \
		-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/android \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/android \
		-G Ninja -Wno-dev > /dev/null
	@cmake --build $(BUILD_DIR)/$(1)-android
	@cp $(BUILD_DIR)/$(1)-android/out/emb $(BIN_DIR)/$(1)/android/emb
	@echo "OK $(1)/android"
endef

aarch64/android:
	$(call android_target,aarch64,arm64-v8a)

armv7/android:
	$(call android_target,armv7,armeabi-v7a)

## ── Utility ──────────────────────────────────────────────────────────────────

test:
	@sh test.sh

clean:
	@rm -rf $(BUILD_DIR)
	@echo "OK clean"
