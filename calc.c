/*
 * calc.c - CLI calculator with REPL
 *
 * Features:
 *  - Scientific functions: trig, factorial, exponential
 *  - Arithmetic (+ - / * %) with correct precedence, left-assoc
 *  - Unary +/- with consecutive-sign collapsing (--3 == 3)
 *  - Power operator ** (right-assoc), XOR ^
 *  - Logic (&& ||) and bitwise (& | ^ << >>) operations
 *  - Variable assignment and usage
 *  - readline history (up/down arrows) with fgets fallback
 *  - help facility
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <setjmp.h>

/* ------------------------- readline with fallback ------------------------- */
#ifdef HAS_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#else
static char *my_readline(const char *prompt);
static void my_add_history(const char *line);
#define readline(p) my_readline(p)
#define add_history(l) my_add_history(l)
#endif

/* ------------------------------ tokenizer -------------------------------- */
typedef enum {
    T_NUM, T_IDENT, T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PERCENT,
    T_POW, T_BANG, T_LPAREN, T_RPAREN, T_COMMA,
    T_EQ, T_EQEQ, T_NE, T_LT, T_LE, T_GT, T_GE,
    T_AND, T_OR, T_XOR, T_AMP, T_PIPE, T_SHL, T_SHR, T_EOF
} TokType;

typedef struct {
    TokType type;
    double  num;
    char    ident[64];
} Token;

typedef struct {
    const char *src;
    size_t      pos;
    Token       cur;
    Token       look;   /* one-token lookahead */
} Lexer;

static double str_val(const char *s, size_t len) {
    char buf[128];
    if (len >= sizeof buf) len = sizeof buf - 1;
    memcpy(buf, s, len);
    buf[len] = '\0';
    return strtod(buf, NULL);
}

static int tok_is_ident_start(char c) {
    return isalpha((unsigned char)c) || c == '_';
}
static int tok_is_ident_char(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

static void lex_next(Lexer *lx) {
    const char *s = lx->src;
    size_t p = lx->pos;
    while (s[p] && isspace((unsigned char)s[p])) p++;

    if (!s[p]) {
        lx->look.type = T_EOF;
        lx->pos = p;
        return;
    }
    char c = s[p];
    if (isdigit((unsigned char)c) || (c == '.' && isdigit((unsigned char)s[p+1]))) {
        size_t st = p;
        while (isdigit((unsigned char)s[p])) p++;
        if (s[p] == '.') {
            p++;
            while (isdigit((unsigned char)s[p])) p++;
        }
        if (s[p] == 'e' || s[p] == 'E') {
            size_t q = p + 1;
            if (s[q] == '+' || s[q] == '-') q++;
            if (isdigit((unsigned char)s[q])) {
                p = q;
                while (isdigit((unsigned char)s[p])) p++;
            }
        }
        lx->look.type = T_NUM;
        lx->look.num  = str_val(s + st, p - st);
        lx->pos = p;
        return;
    }
    if (tok_is_ident_start(c)) {
        size_t st = p;
        while (tok_is_ident_char(s[p])) p++;
        size_t n = p - st;
        if (n >= sizeof lx->look.ident) n = sizeof lx->look.ident - 1;
        memcpy(lx->look.ident, s + st, n);
        lx->look.ident[n] = '\0';
        lx->look.type = T_IDENT;
        lx->pos = p;
        return;
    }
    p++;
    switch (c) {
        case '+': lx->look.type = T_PLUS;    break;
        case '-': lx->look.type = T_MINUS;   break;
        case '*':
            if (s[p] == '*') { p++; lx->look.type = T_POW; }
            else lx->look.type = T_STAR;
            break;
        case '/': lx->look.type = T_SLASH;   break;
        case '%': lx->look.type = T_PERCENT; break;
        case '!':
            if (s[p] == '=') { p++; lx->look.type = T_NE; }
            else lx->look.type = T_BANG;
            break;
        case '(': lx->look.type = T_LPAREN;  break;
        case ')': lx->look.type = T_RPAREN;  break;
        case ',': lx->look.type = T_COMMA;   break;
        case '=':
            if (s[p] == '=') { p++; lx->look.type = T_EQEQ; }
            else lx->look.type = T_EQ;
            break;
        case '<':
            if (s[p] == '=')      { p++; lx->look.type = T_LE; }
            else if (s[p] == '<') { p++; lx->look.type = T_SHL; }
            else lx->look.type = T_LT;
            break;
        case '>':
            if (s[p] == '=')      { p++; lx->look.type = T_GE; }
            else if (s[p] == '>') { p++; lx->look.type = T_SHR; }
            else lx->look.type = T_GT;
            break;
        case '&':
            if (s[p] == '&') { p++; lx->look.type = T_AND; }
            else lx->look.type = T_AMP;
            break;
        case '|':
            if (s[p] == '|') { p++; lx->look.type = T_OR; }
            else lx->look.type = T_PIPE;
            break;
        case '^': lx->look.type = T_XOR;     break;
        default:
            fprintf(stderr, "error: unexpected character '%c'\n", c);
            lx->look.type = T_EOF;
            break;
    }
    lx->pos = p;
}

static void lex_init(Lexer *lx, const char *src) {
    lx->src = src;
    lx->pos = 0;
    lex_next(lx);           /* look = token 1 */
    lx->cur = lx->look;     /* cur  = token 1 */
    lex_next(lx);           /* look = token 2 */
}
static void lex_advance(Lexer *lx) {
    lx->cur = lx->look;     /* cur = next token */
    lex_next(lx);           /* look = token after */
}

/* ------------------------------- variables -------------------------------- */
#define MAX_VARS 256
typedef struct { char name[64]; double val; int used; } Var;
static Var  g_vars[MAX_VARS];
static int  g_nvars = 0;

static double *var_lookup(const char *name) {
    int i;
    for (i = 0; i < g_nvars; i++)
        if (g_vars[i].used && strcmp(g_vars[i].name, name) == 0)
            return &g_vars[i].val;
    return NULL;
}

/* ------------------------- error handling / parser ------------------------ */
static jmp_buf g_jmp;
#define PARSE_ERR(...) do { fprintf(stderr, "error: "); fprintf(stderr, __VA_ARGS__); \
                            fprintf(stderr, "\n"); longjmp(g_jmp, 1); } while (0)

static double expr(Lexer *lx);

/* primary: number, ident (const/var/func call), ( expr ) */
static double primary(Lexer *lx) {
    double v;
    if (lx->cur.type == T_NUM) {
        v = lx->cur.num;
        lex_advance(lx);
        return v;
    }
    if (lx->cur.type == T_IDENT) {
        char name[64];
        strcpy(name, lx->cur.ident);
        lex_advance(lx);

        if (lx->cur.type == T_LPAREN) {   /* function call */
            lex_advance(lx);
            if (lx->cur.type == T_RPAREN) {
                lex_advance(lx);
                if (!strcmp(name, "pi")) return 3.14159265358979323846;
                if (!strcmp(name, "e"))  return 2.71828182845904523536;
                PARSE_ERR("function '%s' requires arguments", name);
            }
            double a = expr(lx);
            double b = 0;
            int has_b = 0;
            if (lx->cur.type == T_COMMA) {
                lex_advance(lx);
                b = expr(lx);
                has_b = 1;
            }
            if (lx->cur.type != T_RPAREN) PARSE_ERR("expected ')'");
            lex_advance(lx);

            if (!strcmp(name, "sin"))    return sin(a);
            if (!strcmp(name, "cos"))    return cos(a);
            if (!strcmp(name, "tan"))    return tan(a);
            if (!strcmp(name, "asin"))   return asin(a);
            if (!strcmp(name, "acos"))   return acos(a);
            if (!strcmp(name, "atan"))   return atan(a);
            if (!strcmp(name, "sinh"))   return sinh(a);
            if (!strcmp(name, "cosh"))   return cosh(a);
            if (!strcmp(name, "tanh"))   return tanh(a);
            if (!strcmp(name, "exp"))    return exp(a);
            if (!strcmp(name, "ln"))     return log(a);
            if (!strcmp(name, "log"))    return log10(a);
            if (!strcmp(name, "log10"))  return log10(a);
            if (!strcmp(name, "log2"))   return log2(a);
            if (!strcmp(name, "sqrt"))   return sqrt(a);
            if (!strcmp(name, "cbrt"))   return cbrt(a);
            if (!strcmp(name, "fact"))   { if (a < 0 || floor(a) != a) PARSE_ERR("factorial requires non-negative integer"); return tgamma(a + 1); }
            if (!strcmp(name, "atan2"))  { if (!has_b) PARSE_ERR("atan2 requires 2 arguments"); return atan2(a, b); }
            if (!strcmp(name, "pow"))    { if (!has_b) PARSE_ERR("pow requires 2 arguments"); return pow(a, b); }
            if (!strcmp(name, "max"))    { if (!has_b) PARSE_ERR("max requires 2 arguments"); return a > b ? a : b; }
            if (!strcmp(name, "min"))    { if (!has_b) PARSE_ERR("min requires 2 arguments"); return a < b ? a : b; }
            PARSE_ERR("unknown function '%s'", name);
        }
        /* variable reference (or named constant) */
        if (!strcmp(name, "pi")) return 3.14159265358979323846;
        if (!strcmp(name, "e"))  return 2.71828182845904523536;
        {
            double *v = var_lookup(name);
            if (!v) PARSE_ERR("undefined variable '%s'", name);
            return *v;
        }
    }
    if (lx->cur.type == T_LPAREN) {
        lex_advance(lx);
        v = expr(lx);
        if (lx->cur.type != T_RPAREN) PARSE_ERR("expected ')'");
        lex_advance(lx);
        return v;
    }
    PARSE_ERR("unexpected token in expression");
    return 0;
}

/* postfix: factorial ! (right side) */
static double postfix(Lexer *lx) {
    double v = primary(lx);
    while (lx->cur.type == T_BANG) {
        lex_advance(lx);
        if (v < 0 || floor(v) != v) PARSE_ERR("factorial requires non-negative integer");
        v = tgamma(v + 1);
    }
    return v;
}

/* power ** : right-assoc, binds tighter than unary */
static double power(Lexer *lx) {
    double base = postfix(lx);
    if (lx->cur.type == T_POW) {
        lex_advance(lx);
        int sign = 1;
        while (lx->cur.type == T_PLUS || lx->cur.type == T_MINUS) {
            if (lx->cur.type == T_MINUS) sign = -sign;
            lex_advance(lx);
        }
        double ex = power(lx);
        if (sign < 0) ex = -ex;
        base = pow(base, ex);
    }
    return base;
}

/* unary: + - with consecutive-sign collapsing */
static double unary(Lexer *lx) {
    int sign = 1;
    while (lx->cur.type == T_PLUS || lx->cur.type == T_MINUS) {
        if (lx->cur.type == T_MINUS) sign = -sign;
        lex_advance(lx);
    }
    double v = power(lx);
    return sign < 0 ? -v : v;
}

static double mul(Lexer *lx) {
    double v = unary(lx);
    for (;;) {
        if (lx->cur.type == T_STAR) {
            lex_advance(lx);
            v = v * unary(lx);
        } else if (lx->cur.type == T_SLASH) {
            lex_advance(lx);
            double d = unary(lx);
            if (d == 0) PARSE_ERR("division by zero");
            v = v / d;
        } else if (lx->cur.type == T_PERCENT) {
            lex_advance(lx);
            double d = unary(lx);
            if (d == 0) PARSE_ERR("modulo by zero");
            v = fmod(v, d);
        } else {
            return v;
        }
    }
}

static double add(Lexer *lx) {
    double v = mul(lx);
    for (;;) {
        if (lx->cur.type == T_PLUS) {
            lex_advance(lx);
            v = v + mul(lx);
        } else if (lx->cur.type == T_MINUS) {
            lex_advance(lx);
            v = v - mul(lx);
        } else {
            return v;
        }
    }
}

static double shift(Lexer *lx) {
    double v = add(lx);
    for (;;) {
        long iv;
        if (lx->cur.type == T_SHL) {
            lex_advance(lx);
            iv = (long)add(lx);
            v = (double)((long long)v << iv);
        } else if (lx->cur.type == T_SHR) {
            lex_advance(lx);
            iv = (long)add(lx);
            v = (double)((long long)v >> iv);
        } else {
            return v;
        }
    }
}

static double relational(Lexer *lx) {
    double v = shift(lx);
    for (;;) {
        if (lx->cur.type == T_LT)  { lex_advance(lx); v = v <  shift(lx); }
        else if (lx->cur.type == T_LE) { lex_advance(lx); v = v <= shift(lx); }
        else if (lx->cur.type == T_GT) { lex_advance(lx); v = v >  shift(lx); }
        else if (lx->cur.type == T_GE) { lex_advance(lx); v = v >= shift(lx); }
        else return v;
    }
}

static double equality(Lexer *lx) {
    double v = relational(lx);
    for (;;) {
        if (lx->cur.type == T_EQEQ) { lex_advance(lx); v = (v == relational(lx)); }
        else if (lx->cur.type == T_NE) { lex_advance(lx); v = (v != relational(lx)); }
        else return v;
    }
}

static double logic_and(Lexer *lx);
static double bit_or(Lexer *lx);
static double bit_xor(Lexer *lx);
static double bit_and(Lexer *lx);

static double logic_or(Lexer *lx) {
    double v = logic_and(lx);
    while (lx->cur.type == T_OR) {
        lex_advance(lx);
        v = (v != 0) || (logic_and(lx) != 0);
    }
    return v;
}

static double logic_and(Lexer *lx) {
    double v = bit_or(lx);
    while (lx->cur.type == T_AND) {
        lex_advance(lx);
        v = (v != 0) && (bit_or(lx) != 0);
    }
    return v;
}

static double bit_or(Lexer *lx) {  /* single | */
    double v = bit_xor(lx);
    while (lx->cur.type == T_PIPE) {
        lex_advance(lx);
        v = (double)((long long)v | (long long)bit_xor(lx));
    }
    return v;
}

static double bit_xor(Lexer *lx) {  /* single ^ */
    double v = bit_and(lx);
    while (lx->cur.type == T_XOR) {
        lex_advance(lx);
        v = (double)((long long)v ^ (long long)bit_and(lx));
    }
    return v;
}

static double bit_and(Lexer *lx) {  /* single & */
    double v = equality(lx);
    while (lx->cur.type == T_AMP) {
        lex_advance(lx);
        v = (double)((long long)v & (long long)equality(lx));
    }
    return v;
}

static double expr(Lexer *lx) {
    return logic_or(lx);
}

/* top-level: handle assignment */
static double statement(Lexer *lx) {
    if (lx->cur.type == T_IDENT) {
        /* lookahead: ident = expr  -> assignment */
        if (lx->look.type == T_EQ) {
            char name[64];
            strcpy(name, lx->cur.ident);
            lex_advance(lx); /* ident */
            lex_advance(lx); /* '=' */
            double v = expr(lx);
            double *slot = var_lookup(name);
            if (!slot) {
                if (g_nvars >= MAX_VARS) PARSE_ERR("too many variables");
                slot = &g_vars[g_nvars].val;
                strcpy(g_vars[g_nvars].name, name);
                g_vars[g_nvars].used = 1;
                g_nvars++;
            }
            *slot = v;
            return v;
        }
    }
    return expr(lx);
}

/* -------------------------------- help ------------------------------------ */
static void print_help(void) {
    fputs(
"CLI Calculator\n"
"==============\n"
"\n"
"Enter an expression and press Enter.  Ctrl-D or 'quit'/'exit' to leave.\n"
"\n"
"Arithmetic (standard precedence, left-assoc at equal level):\n"
"  2 + 3 * 4            -> 14        ( * / % bind tighter than + - )\n"
"  (1 + 2) * 3          -> 9\n"
"  7 % 3                -> 1\n"
"\n"
"Unary minus (higher precedence; consecutive signs collapse):\n"
"  -5 + 3               -> -2\n"
"  --3                  -> 3\n"
"  ---4                 -> -4\n"
"  -2 ** 2              -> -4       (power binds tighter than unary minus)\n"
"\n"
"Exponentiation:  **   (right-associative)\n"
"  2 ** 3 ** 2          -> 512\n"
"  9 ** 0.5             -> 3\n"
"\n"
"Factorial:  n!  or  fact(n)\n"
"  5!                   -> 120\n"
"  fact(0)              -> 1\n"
"\n"
"Functions:\n"
"  sin cos tan asin acos atan atan2(y,x)\n"
"  sinh cosh tanh\n"
"  exp ln log log10 log2 sqrt cbrt\n"
"  pow(x,y) max(a,b) min(a,b)\n"
"\n"
"Constants:  pi  e\n"
"\n"
"Bitwise (operate on integer part):\n"
"  5 & 3                -> 1        (AND)\n"
"  5 | 3                -> 7        (OR)\n"
"  5 ^ 3                -> 6        (XOR)\n"
"  1 << 4               -> 16       (left shift)\n"
"  16 >> 2              -> 4        (right shift)\n"
"\n"
"Logic:  ==  !=  <  <=  >  >=  &&  ||\n"
"  1 < 2 && 3 > 2       -> 1\n"
"\n"
"Variables:  name = expression\n"
"  x = 10               -> 10\n"
"  x * 2                -> 20\n"
"  x = x + 5            -> 15\n"
"\n"
"Commands:\n"
"  help        show this help\n"
"  quit / exit leave the calculator\n"
"\n"
"History: use Up/Down arrows to recall previous commands.\n"
"\n", stdout);
}

/* ------------------------------- readline --------------------------------- */
#ifndef HAS_READLINE
static char *my_readline(const char *prompt) {
    char buf[1024];
    printf("%s", prompt);
    fflush(stdout);
    if (!fgets(buf, sizeof buf, stdin)) return NULL;
    size_t n = strlen(buf);
    if (n && buf[n-1] == '\n') buf[n-1] = '\0';
    {
        char *s = malloc(n + 1);
        if (s) memcpy(s, buf, n + 1);
        return s;
    }
}
static void my_add_history(const char *line) { (void)line; }
#endif

/* -------------------------------- main ------------------------------------ */
int main(void) {
    char *line;
    printf("CLI Calculator.  Type 'help' for help, 'quit' to exit.\n");
    while ((line = readline("calc> ")) != NULL) {
        if (line[0] == '\0') continue;

        if (!strcmp(line, "quit") || !strcmp(line, "exit")) {
            free(line);
            break;
        }
        if (!strcmp(line, "help")) {
            print_help();
            add_history(line);
            free(line);
            continue;
        }

        add_history(line);

        Lexer lx;
        lex_init(&lx, line);
        if (setjmp(g_jmp)) {
            /* parse error: already reported */
            free(line);
            continue;
        }
        double r = statement(&lx);
        if (lx.cur.type != T_EOF)
            PARSE_ERR("unexpected trailing input");
        printf("%g\n", r);
        free(line);
    }
    printf("\n");
    return 0;
}
