/* flatppl-bridge plugin ABI, version 1. See docs/plugin-abi.md. */
#ifndef FLATPPL_BRIDGE_H
#define FLATPPL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Arrow C Data Interface, verbatim from the Arrow specification. */
#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;
  void (*release)(struct ArrowSchema*);
  void* private_data;
};

struct ArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;
  void (*release)(struct ArrowArray*);
  void* private_data;
};

#endif /* ARROW_C_DATA_INTERFACE */

#define FPB_ABI_VERSION 1

/* Operations. */
#define FPB_OP_APPLY 0
#define FPB_OP_LOGDENSITY 1
#define FPB_OP_SAMPLE 2
#define FPB_OP_TOUNIFORM 3
#define FPB_OP_FROMUNIFORM 4
#define FPB_OP_TONORMAL 5
#define FPB_OP_FROMNORMAL 6

/* Derivative kinds. */
#define FPB_DERIV_NONE 0
#define FPB_DERIV_GRADIENT 1
#define FPB_DERIV_VJP 2

/* Derivative targets, a bit mask. */
#define FPB_WRT_INPUT 1
#define FPB_WRT_POINT 2

/* RNG state kinds. */
#define FPB_RNG_NONE 0
#define FPB_RNG_SEED 1
#define FPB_RNG_STATE 2

/* Status codes of call. */
#define FPB_OK 0
#define FPB_INVALID 1     /* bad input values: HTTP 400 */
#define FPB_UNSUPPORTED 2 /* unsupported query: HTTP 422 */
#define FPB_FAILURE 3     /* box failure: HTTP 500 */

/* An Arrow array with its schema. */
typedef struct fpb_array {
  struct ArrowSchema schema;
  struct ArrowArray array;
} fpb_array;

/* A decoded call. Every input array has length `rows`. The plugin may move an
   input array out (copy it and set the original's release to NULL). The core
   releases every input array left in place after the call returns. */
typedef struct fpb_call {
  uint32_t op;
  const char* member;
  int64_t rows;
  fpb_array* input;     /* NULL for a measure member */
  fpb_array* point;     /* logdensity and transports; NULL otherwise */
  const uint64_t* sizes;   /* sample sizes, op FPB_OP_SAMPLE only */
  size_t n_sizes;
  uint32_t rng_kind;
  const uint8_t* rng;
  size_t rng_len;
  uint32_t deriv;
  uint32_t wrt;
  fpb_array* cotangent; /* FPB_DERIV_VJP only */
} fpb_call;

/* A call result. The plugin fills it, and the core takes the arrays. */
typedef struct fpb_result {
  fpb_array value;
  fpb_array d_input;  /* if wrt has FPB_WRT_INPUT */
  fpb_array d_point;  /* if wrt has FPB_WRT_POINT */
  uint8_t* rng;       /* new RNG state for FPB_OP_SAMPLE, freed by release */
  size_t rng_len;
  char* error;        /* message if status is not FPB_OK, freed by release */
  void* private_data;
} fpb_result;

/* The function table. A plugin may serve calls from several threads at once. */
typedef struct fpb_plugin {
  uint32_t abi_version;
  void* state;
  /* FlatPPL describe text, valid until destroy. NULL on failure. */
  const char* (*describe)(void* state);
  int32_t (*call)(void* state, const fpb_call* call, fpb_result* out);
  /* Free rng, error and private_data of a result. Arrays are released by the core. */
  void (*release)(void* state, fpb_result* out);
  void (*destroy)(void* state);
} fpb_plugin;

/* The entry points of a plugin library. `config` is a plugin-defined string.
   On failure, init returns NULL and sets *error to a message, which the core
   frees with fpb_plugin_free_error_v1. */
typedef const fpb_plugin* (*fpb_init_fn)(const char* config, char** error);
typedef void (*fpb_free_error_fn)(char* error);

#define FPB_INIT_SYMBOL "fpb_plugin_init_v1"
#define FPB_FREE_ERROR_SYMBOL "fpb_plugin_free_error_v1"

#ifdef __cplusplus
}
#endif

#endif /* FLATPPL_BRIDGE_H */
