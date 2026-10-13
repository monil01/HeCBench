#ifndef STB_IMAGE_H
#define STB_IMAGE_H
#include <stdio.h>
#include <stdlib.h>

static const char *stbi_failure_reason(void) { return "unsupported or invalid BMP"; }

static unsigned short stbi_u16(FILE *fp)
{
  int b0 = fgetc(fp), b1 = fgetc(fp);
  return (unsigned short)(b0 | (b1 << 8));
}

static unsigned int stbi_u32(FILE *fp)
{
  unsigned int b0 = fgetc(fp), b1 = fgetc(fp), b2 = fgetc(fp), b3 = fgetc(fp);
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static unsigned char *stbi_load(const char *filename, int *x, int *y, int *comp, int req_comp)
{
  (void)req_comp;
  FILE *fp = fopen(filename, "rb");
  if (!fp) return NULL;
  if (fgetc(fp) != 'B' || fgetc(fp) != 'M') {
    fclose(fp);
    return NULL;
  }
  (void)stbi_u32(fp);
  (void)stbi_u16(fp);
  (void)stbi_u16(fp);
  unsigned int data_offset = stbi_u32(fp);
  unsigned int dib_size = stbi_u32(fp);
  if (dib_size < 40) {
    fclose(fp);
    return NULL;
  }
  int w = (int)stbi_u32(fp);
  int h = (int)stbi_u32(fp);
  if (stbi_u16(fp) != 1) {
    fclose(fp);
    return NULL;
  }
  unsigned short bpp = stbi_u16(fp);
  unsigned int compression = stbi_u32(fp);
  if (bpp != 24 || compression != 0 || w <= 0 || h == 0) {
    fclose(fp);
    return NULL;
  }
  int top_down = h < 0;
  if (top_down) h = -h;
  unsigned int row_stride = ((unsigned int)w * 3u + 3u) & ~3u;
  unsigned char *img = (unsigned char *)malloc((size_t)w * h * 3u);
  if (!img) {
    fclose(fp);
    return NULL;
  }
  fseek(fp, (long)data_offset, SEEK_SET);
  unsigned char *row = (unsigned char *)malloc(row_stride);
  if (!row) {
    free(img);
    fclose(fp);
    return NULL;
  }
  for (int r = 0; r < h; ++r) {
    if (fread(row, 1, row_stride, fp) != row_stride) {
      free(row);
      free(img);
      fclose(fp);
      return NULL;
    }
    int dst_r = top_down ? r : (h - 1 - r);
    for (int c = 0; c < w; ++c) {
      unsigned char b = row[c * 3 + 0];
      unsigned char g = row[c * 3 + 1];
      unsigned char rr = row[c * 3 + 2];
      size_t idx = ((size_t)dst_r * w + c) * 3u;
      img[idx + 0] = rr;
      img[idx + 1] = g;
      img[idx + 2] = b;
    }
  }
  free(row);
  fclose(fp);
  *x = w;
  *y = h;
  *comp = 3;
  return img;
}

#endif

