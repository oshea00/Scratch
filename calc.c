#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <setjmp.h>
#include <math.h>
#include <readline/readline.h>
#include <readline/history.h>

#define MAX_VARS   256
#define MAX_NAME   64
#define MAX_TOKENS 4096
#define MAX_ARGS   8

typedef enum { TOK_NUM, TOK_IDENT, TOK_OP, TOK_END } TokType;

typedef struct {
    TokType type;
    double  num;
    char    name[MAX_NAME];
    char    op[4];
} Token;

typedef struct {
    char   name[MAX_NAME];
    double value;
} Var;

static Token tokbuf[MAX_TOKENS + 1];
static int   tokn = 0;
static int   tokp = 0;

static Var  vars[MAX_VARS];
static int  varn = 0;

static jmp_buf parse_jmp;
static char    errmsg[512];

static void perr(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(errmsg, sizeof errmsg, fmt, ap);
    va_end(ap);
    longjmp(parse_jmp, 1);
}

static int   lex(const char *s);
static double parse_expr(void);
static double parse_statement(char *varname, const char **aop);
static double lookup_name(const char *name);
static void   set_var(const char *name, double value);
static Var   *find_var(const char *name);
static double factorial_value(double x);
static double call_function(const char *name, double *args, int nargs);
static const char *tok_desc(const Token *t);
static const char *fmt(double v);

/* ------------------------------ lexer ------------------------------ */

static int lex(const char *s)
{
    tokn = 0;
    tokp = 0;
    while (*s) {
        unsigned char c = (unsigned char)*s;
        if (isspace(c)) { s++; continue; }
        if (tokn >= MAX_TOKENS) {
            snprintf(errmsg, sizeof errmsg, "line too long");
            return -1;
        }
        Token *t = &tokbuf[tokn];
        memset(t, 0, sizeof *t);
        if (isalpha(c) || c == '_') {
            int i = 0;
            while (isalnum((unsigned char)s[i]) || s[i] == '_') {
                if (i < MAX_NAME - 1)
                    t->name[i] = s[i];
                i++;
            }
            t->name[MAX_NAME - 1] = '\0';
            t->type = TOK_IDENT;
            s += i;
        } else if (isdigit(c) || (c == '.' && isdigit((unsigned char)s[1]))) {
            char *end = NULL;
            double v = strtod(s, &end);
            if (end == s) {
                snprintf(errmsg, sizeof errmsg, "bad number at '%s'", s);
                return -1;
            }
            t->type = TOK_NUM;
            t->num = v;
            s = end;
        } else {
            static const char *two[] = {
                "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "%="
            };
            int matched = 0;
            for (int i = 0; i < 11; i++) {
                if (strncmp(s, two[i], 2) == 0) {
                    memcpy(t->op, two[i], 3);
                    t->type = TOK_OP;
                    s += 2;
                    matched = 1;
                    break;
                }
            }
            if (!matched) {
                if (strchr("+ - * / % ^ ! ( ) , ; < > =", c)) {
                    t->op[0] = (char)c;
                    t->op[1] = '\0';
                    t->type = TOK_OP;
                    s++;
                } else {
                    snprintf(errmsg, sizeof errmsg, "unexpected character '%c'", c);
                    return -1;
                }
            }
        }
        tokn++;
    }
    tokbuf[tokn].type = TOK_END;
    tokn++;
    return 0;
}

/* ---------------------------- variables ---------------------------- */

static void copy_name(char *dst, const char *src)
{
    size_t i = 0;
    while (src[i] && i < MAX_NAME - 1) {
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\0';
}

static Var *find_var(const char *name)
{
    for (int i = 0; i < varn; i++)
        if (strcmp(vars[i].name, name) == 0)
            return &vars[i];
    return NULL;
}

static double lookup_name(const char *name)
{
    Var *v = find_var(name);
    if (v)
        return v->value;
    if (strcmp(name, "pi") == 0) return M_PI;
    if (strcmp(name, "e")  == 0) return M_E;
    perr("unknown variable '%s'", name);
    return 0.0;
}

static void set_var(const char *name, double value)
{
    Var *v = find_var(name);
    if (v) {
        v->value = value;
        return;
    }
    if (varn >= MAX_VARS)
        perr("too many variables");
    copy_name(vars[varn].name, name);
    vars[varn].value = value;
    varn++;
}

/* --------------------------- functions ----------------------------- */

static double factorial_value(double x)
{
    if (x < 0.0 || x != floor(x))
        perr("factorial needs a non-negative integer");
    if (x > 170.0)
        perr("factorial too large (max is 170!)");
    double r = 1.0;
    for (double i = 2.0; i <= x; i++)
        r *= i;
    return r;
}

typedef double (*Fn1)(double);
typedef double (*Fn2)(double, double);

static double fact_fn(double x)  { return factorial_value(x); }
static double min2(double a, double b) { return a < b ? a : b; }
static double max2(double a, double b) { return a < b ? b : a; }

struct fentry {
    const char *name;
    int         arity;
    Fn1         f1;
    Fn2         f2;
};

static const struct fentry ftable[] = {
    { "sin", 1, sin, NULL },
    { "cos", 1, cos, NULL },
    { "tan", 1, tan, NULL },
    { "asin", 1, asin, NULL },
    { "acos", 1, acos, NULL },
    { "atan", 1, atan, NULL },
    { "sinh", 1, sinh, NULL },
    { "cosh", 1, cosh, NULL },
    { "tanh", 1, tanh, NULL },
    { "exp", 1, exp, NULL },
    { "ln", 1, log, NULL },
    { "log", 1, log, NULL },
    { "log2", 1, log2, NULL },
    { "log10", 1, log10, NULL },
    { "sqrt", 1, sqrt, NULL },
    { "abs", 1, fabs, NULL },
    { "ceil", 1, ceil, NULL },
    { "floor", 1, floor, NULL },
    { "round", 1, round, NULL },
    { "fact", 1, fact_fn, NULL },
    { "factorial", 1, fact_fn, NULL },
    { "gamma", 1, tgamma, NULL },
    { "pow", 2, NULL, pow },
    { "atan2", 2, NULL, atan2 },
    { "min", 2, NULL, min2 },
    { "max", 2, NULL, max2 },
    { "hypot", 2, NULL, hypot },
    { NULL, 0, NULL, NULL }
};

static double call_function(const char *name, double *args, int nargs)
{
    for (int i = 0; ftable[i].name; i++) {
        if (strcmp(ftable[i].name, name) != 0)
            continue;
        if (nargs != ftable[i].arity)
            perr("function '%s' takes %d argument%s", name,
                 ftable[i].arity, ftable[i].arity == 1 ? "" : "s");
        double r = ftable[i].arity == 1
                 ? ftable[i].f1(args[0])
                 : ftable[i].f2(args[0], args[1]);
        if (isnan(r))
            perr("domain error in '%s'", name);
        if (isinf(r))
            perr("overflow in '%s'", name);
        return r;
    }
    perr("unknown function '%s'", name);
    return 0.0;
}

/* ----------------------------- parser ------------------------------ */

static const char *tok_desc(const Token *t)
{
    static char buf[128];
    if (t->type == TOK_NUM) {
        snprintf(buf, sizeof buf, "%g", t->num);
        return buf;
    }
    if (t->type == TOK_IDENT) return t->name;
    if (t->type == TOK_END)   return "end of line";
    return t->op;
}

static int at_op(const char *op)
{
    return tokp < tokn && tokbuf[tokp].type == TOK_OP &&
           strcmp(tokbuf[tokp].op, op) == 0;
}

static double parse_primary(void)
{
    Token t = tokbuf[tokp];
    if (t.type == TOK_NUM) {
        tokp++;
        return t.num;
    }
    if (t.type == TOK_IDENT) {
        tokp++;
        if (at_op("(")) {
            tokp++;
            double args[MAX_ARGS];
            int nargs = 0;
            if (!at_op(")")) {
                for (;;) {
                    if (nargs >= MAX_ARGS)
                        perr("too many function arguments");
                    args[nargs++] = parse_expr();
                    if (!at_op(","))
                        break;
                    tokp++;
                }
            }
            if (!at_op(")"))
                perr("expected ')' after arguments");
            tokp++;
            return call_function(t.name, args, nargs);
        }
        return lookup_name(t.name);
    }
    if (t.type == TOK_OP && strcmp(t.op, "(") == 0) {
        tokp++;
        double v = parse_expr();
        if (!at_op(")"))
            perr("expected ')'");
        tokp++;
        return v;
    }
    if (t.type == TOK_END)
        perr("unexpected end of expression");
    perr("unexpected '%s'", tok_desc(&t));
    return 0.0;
}

static double parse_postfix(void)
{
    double a = parse_primary();
    while (at_op("!")) {
        tokp++;
        a = factorial_value(a);
    }
    return a;
}

static double parse_unary_fwd(void);

static double parse_power(void)
{
    double base = parse_postfix();
    if (at_op("^")) {
        tokp++;
        double ex = parse_unary_fwd();
        return pow(base, ex);
    }
    return base;
}

static double parse_unary_fwd(void)
{
    if (at_op("-")) { tokp++; return -parse_unary_fwd(); }
    if (at_op("+")) { tokp++; return parse_unary_fwd(); }
    if (at_op("!")) { tokp++; return parse_unary_fwd() != 0.0 ? 0.0 : 1.0; }
    return parse_power();
}

static double parse_term(void)
{
    double a = parse_unary_fwd();
    for (;;) {
        if (at_op("*")) {
            tokp++;
            a *= parse_unary_fwd();
        } else if (at_op("/")) {
            tokp++;
            double b = parse_unary_fwd();
            if (b == 0.0)
                perr("division by zero");
            a /= b;
        } else if (at_op("%")) {
            tokp++;
            double b = parse_unary_fwd();
            if (b == 0.0)
                perr("modulo by zero");
            a = fmod(a, b);
        } else {
            return a;
        }
    }
}

static double parse_add(void)
{
    double a = parse_term();
    for (;;) {
        if (at_op("+")) {
            tokp++;
            a += parse_term();
        } else if (at_op("-")) {
            tokp++;
            a -= parse_term();
        } else {
            return a;
        }
    }
}

static double parse_cmp(void)
{
    double a = parse_add();
    static const char *ops[] = { "==", "!=", "<=", ">=", "<", ">" };
    for (int i = 0; i < 6; i++) {
        if (at_op(ops[i])) {
            tokp++;
            double b = parse_add();
            switch (i) {
                case 0: return a == b;
                case 1: return a != b;
                case 2: return a <= b;
                case 3: return a >= b;
                case 4: return a < b;
                default: return a > b;
            }
        }
    }
    return a;
}

static double parse_and(void)
{
    double a = parse_cmp();
    while (at_op("&&")) {
        tokp++;
        double b = parse_cmp();
        a = (a != 0.0) && (b != 0.0);
    }
    return a;
}

static double parse_or(void)
{
    double a = parse_and();
    while (at_op("||")) {
        tokp++;
        double b = parse_and();
        a = (a != 0.0) || (b != 0.0);
    }
    return a;
}

static double parse_expr(void)
{
    return parse_or();
}

static int is_assign_op(const char *op)
{
    return strcmp(op, "=") == 0 || strcmp(op, "+=") == 0 ||
           strcmp(op, "-=") == 0 || strcmp(op, "*=") == 0 ||
           strcmp(op, "/=") == 0 || strcmp(op, "%=") == 0;
}

static double parse_statement(char *varname, const char **aop)
{
    varname[0] = '\0';
    *aop = NULL;
    if (tokbuf[tokp].type == TOK_IDENT && tokp + 1 < tokn &&
        tokbuf[tokp + 1].type == TOK_OP &&
        is_assign_op(tokbuf[tokp + 1].op)) {
        copy_name(varname, tokbuf[tokp].name);
        *aop = tokbuf[tokp + 1].op;
        tokp += 2;
    }
    return parse_expr();
}

/* ----------------------------- output ------------------------------ */

static const char *fmt(double v)
{
    static char buf[64];
    if (isnan(v)) return "nan";
    if (isinf(v)) return v > 0.0 ? "inf" : "-inf";
    if (v == floor(v) && fabs(v) < 1e16)
        snprintf(buf, sizeof buf, "%.0f", v);
    else
        snprintf(buf, sizeof buf, "%.10g", v);
    return buf;
}

/* ------------------------------ help ------------------------------- */

static const char *H_COMMANDS =
"Commands:\n"
"  help [topic]      show help (topics: commands, operators, functions, examples)\n"
"  vars              list defined variables\n"
"  unset <name>      remove a variable\n"
"  reset             remove all variables\n"
"  history           show command history\n"
"  clear             clear the screen\n"
"  quit | exit       leave the calculator\n"
"  tip: up/down arrows scroll history, Ctrl-R reverse searches it\n";

static const char *H_OPERATORS =
"Operators, low to high precedence (same level evaluates left to right):\n"
"  = += -= *= /= %=    variable assignment\n"
"  ||                   logical OR\n"
"  &&                   logical AND\n"
"  == != < > <= >=      comparison\n"
"  + -                  addition, subtraction\n"
"  * / %                multiplication, division, modulo\n"
"  ^                    power, right-associative (2^3^2 = 512)\n"
"  - + !                unary minus, plus, not (chains: --3 = 3, ---3 = -3)\n"
"  !                    postfix factorial (3! = 6, -3! = -6)\n"
"  ( )                  grouping\n"
"  ;                    separate statements on one line\n";

static const char *H_FUNCTIONS =
"Functions (trig arguments in radians):\n"
"  trig:       sin cos tan asin acos atan\n"
"  hyperbolic: sinh cosh tanh\n"
"  exp/log:    exp ln log log2 log10\n"
"  misc:       sqrt abs ceil floor round gamma\n"
"  two-arg:    pow atan2 min max hypot\n"
"  factorial:  fact(n) or n!\n"
"Constants:    pi, e\n";

static const char *H_EXAMPLES =
"Examples:\n"
"  2 + 3 * 4        -> 14\n"
"  (2 + 3) * 4      -> 20\n"
"  --3              -> 3\n"
"  2^-3             -> 0.125\n"
"  5!               -> 120\n"
"  sin(pi/2)        -> 1\n"
"  exp(ln(10))      -> 10\n"
"  x = 10; x * 2    -> x = 10, then 20\n"
"  1 && 0           -> 0\n"
"  3 > 2            -> 1\n";

static void print_help(const char *topic)
{
    if (!topic) {
        fputs(H_COMMANDS, stdout);
        fputs(H_OPERATORS, stdout);
        fputs(H_FUNCTIONS, stdout);
        fputs(H_EXAMPLES, stdout);
        return;
    }
    if (!strcasecmp(topic, "commands"))
        fputs(H_COMMANDS, stdout);
    else if (!strcasecmp(topic, "operators"))
        fputs(H_OPERATORS, stdout);
    else if (!strcasecmp(topic, "functions"))
        fputs(H_FUNCTIONS, stdout);
    else if (!strcasecmp(topic, "examples"))
        fputs(H_EXAMPLES, stdout);
    else
        printf("Unknown topic '%s'. Available: commands, operators, functions, examples.\n", topic);
}

/* ------------------------------ repl ------------------------------- */

static void eval_line(const char *line)
{
    if (lex(line) < 0) {
        printf("Error: %s\n", errmsg);
        return;
    }
    if (tokn == 1)
        return;
    if (setjmp(parse_jmp) == 0) {
        for (;;) {
            if (tokp >= tokn - 1)
                break;
            char varname[MAX_NAME];
            const char *aop;
            double v = parse_statement(varname, &aop);
            if (aop) {
                double old = 0.0;
                Var *vv = find_var(varname);
                if (vv)
                    old = vv->value;
                else if (strcmp(aop, "=") != 0)
                    perr("unknown variable '%s'", varname);
                switch (aop[0]) {
                case '=':
                    break;
                case '+':
                    v = old + v;
                    break;
                case '-':
                    v = old - v;
                    break;
                case '*':
                    v = old * v;
                    break;
                case '/':
                    if (v == 0.0)
                        perr("division by zero");
                    v = old / v;
                    break;
                case '%':
                    if (v == 0.0)
                        perr("modulo by zero");
                    v = fmod(old, v);
                    break;
                }
                set_var(varname, v);
                printf("%s = %s\n", varname, fmt(v));
            } else {
                printf("%s\n", fmt(v));
            }
            if (at_op(";")) {
                tokp++;
                continue;
            }
            break;
        }
        if (tokp < tokn - 1)
            perr("unexpected '%s'", tok_desc(&tokbuf[tokp]));
    } else {
        printf("Error: %s\n", errmsg);
    }
}

static int handle_line(char *raw)
{
    while (isspace((unsigned char)*raw))
        raw++;
    size_t len = strlen(raw);
    while (len > 0 && isspace((unsigned char)raw[len - 1]))
        raw[--len] = '\0';
    if (len == 0)
        return 1;

    char cmd[128] = "";
    char arg[512] = "";
    if (sscanf(raw, "%127s %511[^\n]", cmd, arg) < 2)
        arg[0] = '\0';

    if (!strcmp(cmd, "quit") || !strcmp(cmd, "exit") || !strcmp(cmd, "q"))
        return 0;
    if (!strcmp(cmd, "help") || !strcmp(cmd, "h") || !strcmp(cmd, "?")) {
        print_help(arg[0] ? arg : NULL);
        return 1;
    }
    if (!strcmp(cmd, "vars")) {
        if (varn == 0)
            printf("no variables defined\n");
        else
            for (int i = 0; i < varn; i++)
                printf("%s = %s\n", vars[i].name, fmt(vars[i].value));
        return 1;
    }
    if (!strcmp(cmd, "unset")) {
        if (!arg[0]) {
            printf("usage: unset <name>\n");
            return 1;
        }
        int found = 0;
        for (int i = 0; i < varn; i++) {
            if (!strcmp(vars[i].name, arg)) {
                memmove(&vars[i], &vars[i + 1], (size_t)(varn - i - 1) * sizeof(Var));
                varn--;
                found = 1;
                break;
            }
        }
        if (!found)
            printf("Error: unknown variable '%s'\n", arg);
        return 1;
    }
    if (!strcmp(cmd, "reset")) {
        varn = 0;
        printf("all variables cleared\n");
        return 1;
    }
    if (!strcmp(cmd, "history")) {
        if (history_length == 0)
            printf("empty history\n");
        for (int i = 0; i < history_length; i++) {
            HIST_ENTRY *he = history_get(i);
            if (he)
                printf("%4d  %s\n", i + 1, he->line);
        }
        return 1;
    }
    if (!strcmp(cmd, "clear")) {
        fputs("\033[2J\033[H", stdout);
        fflush(stdout);
        return 1;
    }
    eval_line(raw);
    return 1;
}

int main(void)
{
    rl_instream = stdin;
    rl_outstream = stdout;

    printf("SciCalc - scientific calculator (type 'help' for help, 'quit' to exit)\n");
    for (;;) {
        char *line = readline("calc> ");
        if (!line) {
            printf("\nbye\n");
            break;
        }
        if (line[0]) {
            add_history(line);
            if (!handle_line(line))
                break;
        }
        free(line);
    }
    return 0;
}
