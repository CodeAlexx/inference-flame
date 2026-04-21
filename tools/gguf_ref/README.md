# gguf_ref — ground-truth generator for GGUF dequant tests

`gen_ref.c` is a small C program that links against llama.cpp's compiled
`libggml*.a` and calls the canonical `dequantize_row_q{8_0,4_K,5_K,6_K}`
kernels on hand-crafted blocks. It prints the raw block bytes and the
resulting `float` outputs as Rust-formatted `const` arrays that are pasted
into `tests/gguf.rs`.

This exists so the K-quant tests are not self-consistency tautologies:
the Rust dequant code is validated against byte-for-byte output from the
reference C implementation, not against its own derivation of the formula.

## Build

Requires llama.cpp cloned and built at `/home/alex/llama.cpp` (release
build produces `build/ggml/src/libggml.a` + friends).

```
cd tools/gguf_ref
gcc -O2 \
    -I/home/alex/llama.cpp/ggml/include \
    -I/home/alex/llama.cpp/ggml/src \
    gen_ref.c \
    /home/alex/llama.cpp/build/ggml/src/libggml.a \
    /home/alex/llama.cpp/build/ggml/src/libggml-cpu.a \
    /home/alex/llama.cpp/build/ggml/src/libggml-base.a \
    -lm -lpthread -ldl -lstdc++ \
    -o gen_ref
./gen_ref > ref_values.rs
```

Then paste the contents into `tests/gguf.rs`.
