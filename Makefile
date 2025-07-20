BIN = rwkv7
SRC = ./src/rwkv7.c
CC = clang
CFLAGS = -Wall -Wextra -Werror -pedantic -O3 -finput-charset=UTF-8 -lm

default: $(BIN)

$(BIN): $(SRC)
	@$(CC) -o $(BIN) $(SRC) $(CFLAGS)

avx: $(SRC)
	@$(CC) -o $(BIN) $(SRC) -mavx -mfma -DAVX $(CFLAGS)

neon: $(SRC)
	@$(CC) -o $(BIN) $(SRC) -DNEON $(CFLAGS)

fp16: $(SRC)
	@$(CC) -o rwkv7-fp16 $(SRC) -DUSE_FP16 $(CFLAGS)

neon-fp16: $(SRC)
	@$(CC) -o rwkv7-neon-fp16 $(SRC) -march=armv8.2-a+fp16 -DUSE_FP16 -DNEON_FP16 $(CFLAGS)

clean:
	@rm -f $(BIN) rwkv7-fp16 rwkv7-neon-fp16

PHONY: avx neon neon-fp16 clean