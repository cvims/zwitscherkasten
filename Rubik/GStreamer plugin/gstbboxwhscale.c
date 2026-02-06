#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <string.h>
#include <ctype.h>

#ifndef PACKAGE
#define PACKAGE "bboxwhscale"
#endif

#ifndef VERSION
#define VERSION "1.0"
#endif

typedef struct _GstBboxWhScale {
  GstBaseTransform parent;
  gint width;
  gint height;
  gboolean inverse;   // FALSE: multiply (norm->px), TRUE: divide (px->norm)
} GstBboxWhScale;

typedef struct _GstBboxWhScaleClass {
  GstBaseTransformClass parent_class;
} GstBboxWhScaleClass;

#define GST_TYPE_BBOXWHSCALE (gst_bboxwhscale_get_type())
G_DEFINE_TYPE(GstBboxWhScale, gst_bboxwhscale, GST_TYPE_BASE_TRANSFORM)

enum {
  PROP_0,
  PROP_WIDTH,
  PROP_HEIGHT,
  PROP_INVERSE
};

static void gst_bboxwhscale_set_property(GObject *obj, guint prop_id,
                                         const GValue *value, GParamSpec *pspec) {
  GstBboxWhScale *s = (GstBboxWhScale*)obj;
  switch (prop_id) {
    case PROP_WIDTH:   s->width   = g_value_get_int(value);      break;
    case PROP_HEIGHT:  s->height  = g_value_get_int(value);      break;
    case PROP_INVERSE: s->inverse = g_value_get_boolean(value);  break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec); break;
  }
}

static void gst_bboxwhscale_get_property(GObject *obj, guint prop_id,
                                         GValue *value, GParamSpec *pspec) {
  GstBboxWhScale *s = (GstBboxWhScale*)obj;
  switch (prop_id) {
    case PROP_WIDTH:   g_value_set_int(value, s->width); break;
    case PROP_HEIGHT:  g_value_set_int(value, s->height); break;
    case PROP_INVERSE: g_value_set_boolean(value, s->inverse); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec); break;
  }
}

static inline gboolean is_num_start(char c) {
  return (c == '-' || c == '+' || c == '.' || (c >= '0' && c <= '9'));
}

static gboolean parse_next_double_with_span(const char **p, const char *end,
                                            const char **nstart, const char **nend,
                                            double *out) {
  const char *q = *p;
  while (q < end && !is_num_start(*q)) q++;
  if (q >= end) return FALSE;

  char *e = NULL;
  double v = g_ascii_strtod(q, &e);
  if (!e || e == q) return FALSE;

  *nstart = q;
  *nend   = e;
  *out    = v;
  *p      = e;
  return TRUE;
}

static inline double clamp01(double v) {
  if (v < 0.0) return 0.0;
  if (v > 1.0) return 1.0;
  return v;
}

static GstFlowReturn gst_bboxwhscale_transform(GstBaseTransform *bt,
                                              GstBuffer *inbuf,
                                              GstBuffer *outbuf) {
  GstBboxWhScale *s = (GstBboxWhScale*)bt;

  GstMapInfo inmap;
  if (!gst_buffer_map(inbuf, &inmap, GST_MAP_READ))
    return GST_FLOW_ERROR;

  const char *in  = (const char*)inmap.data;
  const char *end = in + inmap.size;

  GString *out = g_string_sized_new(inmap.size + 128);

  const char *cur = in;
  while (cur < end) {
    const char *rect = g_strstr_len(cur, end - cur, "rectangle");
    if (!rect) {
      g_string_append_len(out, cur, end - cur);
      break;
    }

    // copy up to "rectangle"
    g_string_append_len(out, cur, rect - cur);

    // find next '<' after rectangle
    const char *lt = memchr(rect, '<', end - rect);
    if (!lt) {
      g_string_append_len(out, rect, end - rect);
      break;
    }

    // copy from "rectangle" up to and including '<'
    g_string_append_len(out, rect, (lt + 1) - rect);

    // parse 4 numbers after '<'
    const char *p = lt + 1;
    const char *s1,*e1,*s2,*e2,*s3,*e3,*s4,*e4;
    double x,y,w,h;

    if (!parse_next_double_with_span(&p, end, &s1,&e1,&x) ||
        !parse_next_double_with_span(&p, end, &s2,&e2,&y) ||
        !parse_next_double_with_span(&p, end, &s3,&e3,&w) ||
        !parse_next_double_with_span(&p, end, &s4,&e4,&h)) {
      // couldn't parse -> continue after '<'
      cur = lt + 1;
      continue;
    }

    // scale depending on inverse
    if (s->width <= 0 || s->height <= 0) {
      // just passthrough of numbers if invalid configuration
    } else if (!s->inverse) {
      // normalized -> pixels
      x *= (double)s->width;
      w *= (double)s->width;
      y *= (double)s->height;
      h *= (double)s->height;
    } else {
      // pixels -> normalized
      x /= (double)s->width;
      w /= (double)s->width;
      y /= (double)s->height;
      h /= (double)s->height;

      // optional: keep in sane range for overlay consumers
      x = clamp01(x);
      y = clamp01(y);
      w = clamp01(w);
      h = clamp01(h);
    }

    char tmp[64];

    g_string_append_len(out, lt + 1, s1 - (lt + 1));
    g_ascii_dtostr(tmp, sizeof(tmp), x);
    g_string_append(out, tmp);

    g_string_append_len(out, e1, s2 - e1);
    g_ascii_dtostr(tmp, sizeof(tmp), y);
    g_string_append(out, tmp);

    g_string_append_len(out, e2, s3 - e2);
    g_ascii_dtostr(tmp, sizeof(tmp), w);
    g_string_append(out, tmp);

    g_string_append_len(out, e3, s4 - e3);
    g_ascii_dtostr(tmp, sizeof(tmp), h);
    g_string_append(out, tmp);

    cur = e4; // continue after 4th number
  }

  gst_buffer_unmap(inbuf, &inmap);

  // Resize outbuf to exact size and write
  gst_buffer_set_size(outbuf, out->len);
  gst_buffer_fill(outbuf, 0, out->str, out->len);

  g_string_free(out, TRUE);
  return GST_FLOW_OK;
}

static void gst_bboxwhscale_class_init(GstBboxWhScaleClass *klass) {
  GObjectClass *gobj = G_OBJECT_CLASS(klass);
  GstElementClass *eclass = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass *bt = GST_BASE_TRANSFORM_CLASS(klass);

  gobj->set_property = gst_bboxwhscale_set_property;
  gobj->get_property = gst_bboxwhscale_get_property;

  g_object_class_install_property(gobj, PROP_WIDTH,
    g_param_spec_int("width", "width", "Video width in pixels",
                     1, 8192, 640, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobj, PROP_HEIGHT,
    g_param_spec_int("height", "height", "Video height in pixels",
                     1, 8192, 480, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobj, PROP_INVERSE,
    g_param_spec_boolean("inverse", "inverse",
                         "If true: divide by width/height (pixel -> normalized). "
                         "If false: multiply (normalized -> pixel).",
                         FALSE,
                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_metadata(eclass,
    "bboxwhscale", "Filter/Text",
    "Scale rectangle <x,y,w,h> by width/height (norm<->px) with inverse option",
    "local");

  GstCaps *caps = gst_caps_from_string("text/x-raw,format=utf8");
  GstPadTemplate *sink_t = gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps);
  GstPadTemplate *src_t  = gst_pad_template_new("src",  GST_PAD_SRC,  GST_PAD_ALWAYS, caps);
  gst_element_class_add_pad_template(eclass, sink_t);
  gst_element_class_add_pad_template(eclass, src_t);
  gst_caps_unref(caps);

  bt->transform = GST_DEBUG_FUNCPTR(gst_bboxwhscale_transform);

  // NOT in-place, because output string length can change
}

static void gst_bboxwhscale_init(GstBboxWhScale *s) {
  s->width = 640;
  s->height = 480;
  s->inverse = FALSE;

  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(s), FALSE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(s), FALSE);
}

static gboolean plugin_init(GstPlugin *p) {
  return gst_element_register(p, "bboxwhscale", GST_RANK_NONE, gst_bboxwhscale_get_type());
}

GST_PLUGIN_DEFINE(
  GST_VERSION_MAJOR, GST_VERSION_MINOR,
  bboxwhscale,
  "Scale bbox xywh by width/height",
  plugin_init,
  VERSION,
  "LGPL",
  "bboxwhscale",
  "local"
)
