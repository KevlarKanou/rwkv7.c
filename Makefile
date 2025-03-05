BIN = rwkv7
SRC = ./src/rwkv7.c
CC = clang
CFLAGS = -Wall -Wextra -Werror -pedantic -O3 -finput-charset=UTF-8 -lm

default: $(BIN)

$(BIN): $(SRC)
	@$(CC) $(CFLAGS) -o rwkv7 $(SRC)

clean:
	@rm -f $(BIN)

PHONY: clean