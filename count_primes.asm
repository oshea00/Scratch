; count_primes.asm - counts primes up to N using a bit sieve of Eratosthenes
; Build: nasm -f elf64 count_primes.asm && gcc -no-pie -o count_primes count_primes.o

default rel

section .data
    N       equ 10000000
    NBYTES  equ ((N + 7) / 8)   ; one bit per number
    msg     db "Number of primes up to 10000000: ", 0
    newline db 10

section .bss
    sieve   resb NBYTES
    buf     resb 32

section .text
    global main

main:
    push rbp
    mov  rbp, rsp
    and  rsp, -16

    ; sieve[i>>3] bit i&7 == 0 means i is prime (only odds stored conceptually,
    ; but we keep full byte map for simplicity)
    mov  rdi, sieve
    mov  ecx, NBYTES
    mov  al, 0
    rep  stosb
    ; mark 0 and 1 as composite
    mov  byte [sieve + 0], 0x03   ; bits 0,1 set -> composite

    ; iterate p from 2..sqrt(N)
    mov  r8d, 2
.p_loop:
    mov  eax, r8d
    imul eax, eax
    cmp  eax, N
    jge  .count
    ; if p prime?
    bt   [sieve], r8d
    jc   .next_p
    ; mark multiples of p starting at p*p
    mov  r9d, eax
.mark_loop:
    bts  [sieve], r9d
    add  r9d, r8d
    cmp  r9d, N
    jl   .mark_loop
.next_p:
    inc  r8d
    jmp  .p_loop

.count:
    ; count cleared bits (primes)
    xor  eax, eax
    mov  edx, N
.loop:
    dec  edx
    js   .done
    bt   [sieve], edx
    jc   .loop
    inc  eax
    jmp  .loop
.done:
    mov  r12d, eax

    ; print message
    mov  eax, 1
    mov  edi, 1
    lea  rsi, [msg]
    mov  edx, 33
    syscall

    ; convert count to decimal in buf
    mov  eax, r12d
    lea  rdi, [buf + 31]
    mov  byte [rdi], 0
    mov  ecx, 10
.conv:
    xor  edx, edx
    div  ecx
    add  dl, '0'
    dec  rdi
    mov  [rdi], dl
    test eax, eax
    jnz  .conv

    ; print number
    lea  rax, [buf + 31]
    sub  rax, rdi
    mov  rdx, rax
    mov  rsi, rdi
    mov  eax, 1
    mov  edi, 1
    syscall

    ; print newline
    mov  eax, 1
    mov  edi, 1
    lea  rsi, [newline]
    mov  edx, 1
    syscall

    xor  eax, eax
    leave
    ret
