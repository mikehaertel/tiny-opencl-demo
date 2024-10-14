CFLAGS=-O
N=1024
matmul: matmul.c
	$(CC) $(CFLAGS) -DN=$N -o $@ $< -lOpenCL
.PHONY: clean
clean:
	rm -f matmul
