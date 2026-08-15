CC         = gcc
CFLAGS     = -Wall -Wextra -O2 -std=c11
LDLIBS     = -lm -lreadline
READLINE   = -DHAS_READLINE

calc: calc.c
	$(CC) $(CFLAGS) $(READLINE) $< -o $@ $(LDLIBS)

# Build without readline (plain fgets fallback)
calc-noreadline: calc.c
	$(CC) $(CFLAGS) $< -o $@ -lm

clean:
	rm -f calc calc-noreadline

.PHONY: clean
