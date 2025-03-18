BIN = rwkv7
SRC = ./src/rwkv7.c
CC = clang
CFLAGS = -Wall -Wextra -Werror -pedantic -O3 -finput-charset=UTF-8 -lm

default: $(BIN)

$(BIN): $(SRC)
	@$(CC) -o $(BIN) $(SRC) $(CFLAGS)

avx: $(SRC)
	@$(CC) -o $(BIN) $(SRC) -mavx -mfma -DAVX $(CFLAGS)

avx2: $(SRC)
	@$(CC) -o $(BIN) $(SRC) -mavx2 -mfma -DAVX -DAVX2 $(CFLAGS)

clean:
	@rm -f $(BIN)

PHONY: avx avx2 clean