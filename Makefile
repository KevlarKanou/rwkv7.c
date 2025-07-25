BIN = rwkv7
DEPS = $(shell find ./src -name '*.c' -or -name '*.h')
SRC = ./src/rwkv7.c
CC = clang
CFLAGS = -Wall -Wextra -Werror -pedantic -O3 -finput-charset=UTF-8 -lm

default: $(BIN)

$(BIN): $(DEPS)
	@$(CC) -o $(BIN) $(SRC) $(CFLAGS)

avx: $(DEPS)
	@$(CC) -o $(BIN) $(SRC) -mavx -mfma -DAVX $(CFLAGS)

neon: $(DEPS)
	@$(CC) -o $(BIN) $(SRC) -DNEON $(CFLAGS)

clean:
	@rm -f $(BIN)

PHONY: avx neon clean