/*
 * Functions for the emulation tests. Each one exercises one thing the
 * verifier design (docs/verifier.md) says can make two compilations of the
 * same function look different when run, or one thing the harness has to
 * handle to run them at all. Built at -O0 and -O3 by build.sh; the tests
 * read addresses and signatures from the DLLs' own DWARF.
 *
 * Nothing here is meant to be good C. Some of it is undefined on purpose.
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* --- plain values ------------------------------------------------------ */

int add3(int a, int b, int c) { return a * 3 + b - c; }

uint64_t mix64(uint64_t a, uint32_t b) {
    return (a ^ ((uint64_t)b << 17)) * 0x9E3779B97F4A7C15ull;
}

/* --- risk 1: the return register beyond the return type --------------- */

void fill(unsigned char *p, int n, unsigned char v) {
    for (int i = 0; i < n; i++) p[i] = (unsigned char)(v + i);
}

signed char low_byte(int x) { return (signed char)(x * 7 + 3); }

unsigned short low_word(unsigned int x) { return (unsigned short)(x ^ 0xBEEFu); }

_Bool is_odd(int x) { return x & 1; }

/* --- input through a pointer: the garbage-rule correction ------------- */

int sum(const int *p, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += p[i] * (i + 1);
    return s;
}

size_t count_nonzero(const unsigned char *p, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) k += p[i] != 0;
    return k;
}

/* --- risk 2: reading an uninitialised local --------------------------- */

int uninit(int flag) {
    int x;
    if (flag) x = 5;
    return x;           /* undefined when flag == 0 */
}

/* --- floating point and the Win64 register slots ---------------------- */

double scale(double x, int k) { return x * k + 0.5; }

float fma3(float a, float b, float c) { return a * b + c; }

/* int, double, int, double: RCX, XMM1, R8, XMM3 - slots go by position */
double mixed(int a, double b, int c, double d) { return a * b - c * d; }

/* --- more than four arguments: the stack ------------------------------ */

long long six(long long a, long long b, long long c,
              long long d, long long e, long long f) {
    return a - b + c * 2 - d + e * 3 - f;
}

double five_mixed(double a, int b, double c, int d, double e) {
    return a + b * c - d + e * 2.0;
}

/* --- risk 4: calls, inlined at one level and not the other ------------ */

static int helper(int x) { return x * x + 1; }

int uses_helper(int a, int b) { return helper(a) - helper(b); }

void copy(char *d, const char *s, size_t n) { memcpy(d, s, n); }

char *duplicate(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

/* Wrappers that each lean on one string function, to exercise its stub. */

size_t length_of(const char *s) { return strlen(s); }

int compare_strings(const char *a, const char *b) { return strcmp(a, b); }

int compare_n(const char *a, const char *b, size_t n) { return strncmp(a, b, n); }

char *find_char(const char *s, int c) { return strchr(s, c); }

char *find_sub(const char *s, const char *needle) { return strstr(s, needle); }

void copy_string(char *d, const char *s) { strcpy(d, s); }

size_t concat_len(char *d, const char *s) { strcat(d, s); return strlen(d); }

/* --- risk 7: running out of time -------------------------------------- */

int spin(int n) {
    volatile int x = 0;
    while (n) x++;      /* never returns for n != 0 */
    return x;
}

/* --- faults ------------------------------------------------------------ */

int divide(int a, int b) { return a / b; }

int null_read(int k) {
    volatile int *p = 0;
    return p[k];
}

/* --- risk 9: globals, the known blind spot ---------------------------- */

int counter;

int bump(int k) {
    counter += k;
    return k * 2;
}

/* Writes into a global array - the array lives in the binary's own data, at
 * a different address at -O0 and -O3, so the write must be excluded from the
 * comparison (risk 9). The return still agrees. */
static int global_table[8];

int record(int index, int value) {
    global_table[index & 7] = value;
    return global_table[index & 7] + 1;
}

/* Returns a pointer into the binary's own .rdata - a static string. Its
 * address differs at -O0 and -O3, so the returned pointer must not be
 * compared by value (F18). */
__attribute__((noinline))
const char *banner(void) {
    return "elenchus fixture banner v1";
}

/* Writes the address of a global into the caller's buffer. The address is
 * different at -O0 and -O3, so the written pointer value must not be
 * compared as data (F23); the count written beside it must. */
void publish(const char **out, int *count) {
    out[0] = banner();
    count[0] = 2;
}

/* --- -O3, a saved XMM register, and the shadow space ------------------ */

/*
 * XMM6-XMM15 are callee-saved on Win64. keeps_across_call holds a double
 * across a call, so at -O3 the value lives in XMM6 and the prologue saves
 * XMM6 to the stack, restoring it after - a spill an -O0 build never makes.
 * The call also uses the 32 bytes of shadow space the convention reserves.
 * If the harness does not lay the stack out the way the convention says,
 * this function is where it shows.
 *
 * The store MinGW emits is movups, not the aligned movaps some ABIs use
 * (checked in the disassembly, not assumed), so stack alignment alone does
 * not fault here. Two earlier attempts exercised nothing and were dropped:
 * a local double[4] stayed in registers, and a direct call let GCC keep the
 * value in a volatile register. The call goes through a volatile pointer so
 * the callee is opaque and the value must be saved.
 */
__attribute__((noinline)) double ext_scale(double x) { return x * 1.5; }

double (*volatile scale_fn)(double) = ext_scale;

double keeps_across_call(double a, double b) {
    double u = a * b;
    double v = scale_fn(a);
    return u + v;
}

/* --- signatures the harness must decline ------------------------------- */

struct pair { int a, b; };
struct big { long long a, b, c; };

int pair_sum(struct pair p) { return p.a + p.b; }

long long big_sum(struct big b) { return b.a + b.b + b.c; }

struct big make_big(int x) {
    struct big b = { x, x * 2, x * 3 };
    return b;
}

int vsum(int n, ...) {
    va_list ap;
    va_start(ap, n);
    int s = 0;
    for (int i = 0; i < n; i++) s += va_arg(ap, int);
    va_end(ap);
    return s;
}

long double wide(long double x) { return x * 2; }

/* --- types that resolve through names ---------------------------------- */

typedef unsigned int u32;
typedef u32 word;

enum colour { RED, GREEN, BLUE };

word through_typedef(word x) { return x + 1; }

int colour_value(enum colour c) { return (int)c * 10; }

int apply(int (*f)(int), int x) { return f(x); }

const char *first_char(const char *const *strings) { return strings[0]; }
