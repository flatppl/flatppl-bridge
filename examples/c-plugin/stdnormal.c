/* A minimal C plugin: the standard normal measure, log density only.
   Build: cc -shared -fPIC -I include examples/c-plugin/stdnormal.c -o libstdnormal.so */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "flatppl_bridge.h"

static const char* DESCRIBE =
    "{\"stdnormal\": {\"kind\": \"measure\", \"queries\": [\"builtin_logdensityof\"], "
    "\"gradient\": false, \"vjp\": false, \"variate\": \"reals\", \"normalized\": true}}";

static const char* describe(void* state) {
  (void)state;
  return DESCRIBE;
}

static void release_array(struct ArrowArray* array) {
  free((void*)array->buffers[1]);
  free(array->buffers);
  array->release = NULL;
}

static void release_schema(struct ArrowSchema* schema) { schema->release = NULL; }

static int32_t call(void* state, const fpb_call* call, fpb_result* out) {
  (void)state;
  if (call->op != FPB_OP_LOGDENSITY || strcmp(call->member, "stdnormal") != 0 || call->deriv != FPB_DERIV_NONE) {
    out->error = "only builtin_logdensityof on stdnormal";
    return FPB_UNSUPPORTED;
  }
  const struct ArrowSchema* schema = &call->point->schema;
  const struct ArrowArray* point = &call->point->array;
  if (strcmp(schema->format, "g") != 0 || point->null_count != 0) {
    out->error = "the point must be a real scalar";
    return FPB_INVALID;
  }
  const double* x = (const double*)point->buffers[1] + point->offset;
  double* y = malloc(sizeof(double) * (size_t)call->rows);
  for (int64_t i = 0; i < call->rows; i++) y[i] = -0.5 * x[i] * x[i] - 0.5 * log(2.0 * M_PI);

  const void** buffers = malloc(2 * sizeof(void*));
  buffers[0] = NULL;
  buffers[1] = y;
  out->value.array = (struct ArrowArray){.length = call->rows, .n_buffers = 2, .buffers = buffers, .release = release_array};
  out->value.schema = (struct ArrowSchema){.format = "g", .name = "", .release = release_schema};
  return FPB_OK;
}

/* The error strings are static, so release has nothing to free. */
static void release(void* state, fpb_result* out) {
  (void)state;
  (void)out;
}

static void destroy(void* state) { (void)state; }

static const fpb_plugin PLUGIN = {FPB_ABI_VERSION, NULL, describe, call, release, destroy};

const fpb_plugin* fpb_plugin_init_v1(const char* config, char** error) {
  (void)config;
  (void)error;
  return &PLUGIN;
}

void fpb_plugin_free_error_v1(char* error) { free(error); }
