#ifndef STB_IMAGE_WRITE_H
#define STB_IMAGE_WRITE_H
#include <stdio.h>

static void stbi_put_u16(FILE *fp, unsigned int v)
{
  fputc(v & 255u, fp);
  fputc((v >> 8) & 255u, fp);
}

static void stbi_put_u32(FILE *fp, unsigned int v)
{
  fputc(v & 255u, fp);
  fputc((v >> 8) & 255u, fp);
  fputc((v >> 16) & 255u, fp);
  fputc((v >> 24) & 255u, fp);
}

static int stbi_write_bmp(const char *filename, int w, int h, int comp, const void *data)
{
  if (comp < 3 || w <= 0 || h <= 0) return 0;
  FILE *fp = fopen(filename, "wb");
  if (!fp) return 0;
  unsigned int row_stride = ((unsigned int)w * 3u + 3u) & ~3u;
  unsigned int image_size = row_stride * (unsigned int)h;
  unsigned int file_size = 54u + image_size;
  const unsigned char *img = (const unsigned char *)data;
  fputc('B', fp); fputc('M', fp);
  stbi_put_u32(fp, file_size);
  stbi_put_u16(fp, 0); stbi_put_u16(fp, 0);
  stbi_put_u32(fp, 54);
  stbi_put_u32(fp, 40);
  stbi_put_u32(fp, (unsigned int)w);
  stbi_put_u32(fp, (unsigned int)h);
  stbi_put_u16(fp, 1);
  stbi_put_u16(fp, 24);
  stbi_put_u32(fp, 0);
  stbi_put_u32(fp, image_size);
  stbi_put_u32(fp, 2835); stbi_put_u32(fp, 2835);
  stbi_put_u32(fp, 0); stbi_put_u32(fp, 0);
  for (int r = h - 1; r >= 0; --r) {
    for (int c = 0; c < w; ++c) {
      const unsigned char *p = img + ((size_t)r * w + c) * comp;
      fputc(p[2], fp); fputc(p[1], fp); fputc(p[0], fp);
    }
    for (unsigned int pad = (unsigned int)w * 3u; pad < row_stride; ++pad) fputc(0, fp);
  }
  fclose(fp);
  return 1;
}

#endif

