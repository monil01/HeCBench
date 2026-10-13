#include <chrono>
#include <iostream>
#include "bitcracker.h"
int   *h_found;
char  *h_pswd_char;       // used only for printing purpose, never moved to device
int     pswd_per_thread = 1;               // only for printing
char  printed_pswd[PSWD_NUM_CHAR + 1];   // only for printing

static int check_match() {
  int  out_pswd_ind;
  if (*h_found >= 0){
    out_pswd_ind = *h_found;
    snprintf((char*)printed_pswd, PSWD_NUM_CHAR, "%s", (char *)(h_pswd_char + out_pswd_ind * PSWD_NUM_CHAR));
    for (int i = 0; i < PSWD_NUM_CHAR; i++) {
      if (printed_pswd[i] == (char)0x80 || printed_pswd[i] == (char)0xffffff80) {
        printed_pswd[i] = '\0';
      }
    }
    return 1;
  }
  return 0;
}

void attack(
    char *dname,
    iris::DMem &d_w_words_uint32,
    unsigned char* encryptedVMK,
    unsigned char* nonce,
    unsigned char* encryptedMAC,
    int gridBlocks)
{
  FILE      *fp;
  int        match = 0;
  int        h_w_words_uint32[4];
  uint32_t   num_read_pswd;
  long long  tot_num_read_pswd = 0;
  uint8_t vmkIV[IV_SIZE];
  uint8_t macIV[IV_SIZE];
  uint8_t    computedMacIV[IV_SIZE];

  uint32_t* h_pswd_uint32;     // uint32 representation of passwords in host

  if(dname == NULL || encryptedVMK == NULL)
  {
    fprintf(stderr, "Attack input error\n");
    return;
  }

  if(max_num_pswd_per_read <= 0)
  {
    fprintf(stderr, "Attack tot passwords error: %d\n", max_num_pswd_per_read);
    return;
  }

  //-------- vmkIV setup ------
  memset(vmkIV, 0, IV_SIZE);
  vmkIV[0] = (unsigned char)(IV_SIZE - 1 - NONCE_SIZE - 1);
  memcpy(vmkIV + 1, nonce, NONCE_SIZE);
  if(IV_SIZE - 1 - NONCE_SIZE - 1 < 0)
  {
    fprintf(stderr, "Attack nonce error\n");
    return;
  }
  vmkIV[IV_SIZE - 1] = 1; 
  // -----------------------


  //-------- macIV setup ------
  memset(macIV, 0, IV_SIZE);
  macIV[0] = (unsigned char)(IV_SIZE - 1 - NONCE_SIZE - 1);
  memcpy(macIV + 1, nonce, NONCE_SIZE);
  if(IV_SIZE - 1 - NONCE_SIZE - 1 < 0)
  {
    fprintf(stderr, "Attack nonce error\n");
    return;
  }
  macIV[IV_SIZE - 1] = 0; 
  // -----------------------

  //-------- computedMacIV setup ------
  memset(computedMacIV, 0, IV_SIZE);
  computedMacIV[0] = 0x3a;
  memcpy(computedMacIV + 1, nonce, NONCE_SIZE);
  if(IV_SIZE - 1 - NONCE_SIZE - 1 < 0)
  {
    fprintf(stderr, "Attack nonce error\n");
    return;
  }
  computedMacIV[IV_SIZE - 1] = 0x2c;

  // ---- Open File Dictionary ----
  if (!memcmp(dname, "-\0", 2)) {
    fp = stdin;
  } else {
    fp = fopen(dname, "r");
    if (!fp) {
      fprintf(stderr, "Can't open dictionary file %s.\n", dname);
      return;
    }
  }
  // -------------------------------

  std::vector<char> passwords(max_num_pswd_per_read * PSWD_NUM_CHAR);
  std::vector<uint32_t> password_words(max_num_pswd_per_read * PSWD_NUM_UINT32);
  int found = -1;
  h_found = &found;
  h_pswd_char = passwords.data();
  h_pswd_uint32 = password_words.data();
  // Salt words equal the first four schedule words (CUDA copies them from W).
  for (int i = 0; i < 4; ++i)
    h_w_words_uint32[i] = uint32_t(salt[4*i]) << 24 | uint32_t(salt[4*i+1]) << 16 |
                         uint32_t(salt[4*i+2]) << 8 | uint32_t(salt[4*i+3]);
  iris::DMem d_vmk(encryptedVMK, VMK_FULL_SIZE), d_vmkIV(vmkIV, IV_SIZE);
  iris::DMem d_mac(encryptedMAC, MAC_SIZE), d_macIV(macIV, IV_SIZE);
  iris::DMem d_computedMacIV(computedMacIV, IV_SIZE), d_found(&found, sizeof(found));
  iris::DMem d_pswd_uint32(password_words.data(), password_words.size() * sizeof(uint32_t));

  printf("Type of attack: %s\n", "User Password");
  printf("Psw per thread: %d\n", pswd_per_thread);
  printf("max_num_pswd_per_read: %d\n", max_num_pswd_per_read);
  printf("Dictionary: %s\n", (fp == stdin) ? "standard input" : dname);
  printf("MAC Comparison (-m): %s\n", "Yes");
  printf("\n");

  auto v0 = h_w_words_uint32[0];
  auto v1 = h_w_words_uint32[1];
  auto v2 = h_w_words_uint32[2];
  auto v3 = h_w_words_uint32[3];

  uint32_t s0 =  ((uint32_t)salt[ 0]) << 24 | ((uint32_t)salt[ 1]) << 16 | ((uint32_t)salt[ 2]) <<  8 | ((uint32_t)salt[ 3]); 
  uint32_t s1 =  ((uint32_t)salt[ 4]) << 24 | ((uint32_t)salt[ 5]) << 16 | ((uint32_t)salt[ 6]) <<  8 | ((uint32_t)salt[ 7]); 
  uint32_t s2 =  ((uint32_t)salt[ 8]) << 24 | ((uint32_t)salt[ 9]) << 16 | ((uint32_t)salt[10]) <<  8 | ((uint32_t)salt[11]);
  uint32_t s3 =  ((uint32_t)salt[12]) << 24 | ((uint32_t)salt[13]) << 16 | ((uint32_t)salt[14]) <<  8 | ((uint32_t)salt[15]);

  double total_time = 0;
  int iter = 0;
  while(true) {
    iter++;

    // populate h_pswd_uint32 and h_pswd_char
    num_read_pswd = read_password(&h_pswd_uint32, &h_pswd_char, max_num_pswd_per_read, fp);
    if(num_read_pswd <= 0) {
      break;
    }
    std::cout <<"\nIter: " << iter<< ", num passwords read: " << num_read_pswd << std::endl;

    iris_check(d_pswd_uint32.update(password_words.data()));
    auto start = std::chrono::steady_clock::now();
    iris::Task task;
    void *args[] = { &num_read_pswd, &d_found, &d_vmk, &d_vmkIV, &d_mac, &d_macIV,
                    &d_computedMacIV, &v0, &v1, &v2, &v3, &s0, &s1, &s2, &s3,
                    &d_pswd_uint32, &d_w_words_uint32 };
    int info[] = { sizeof(num_read_pswd), iris_rw, iris_r, iris_r, iris_r, iris_r,
                   iris_r, sizeof(v0), sizeof(v1), sizeof(v2), sizeof(v3),
                   sizeof(s0), sizeof(s1), sizeof(s2), sizeof(s3), iris_r, iris_r };
    size_t gws[] = { (size_t(num_read_pswd) + 255) / 256 * 256 }, lws[] = {256};
    iris_check(task.kernel("decrypt_vmk_with_mac", 1, nullptr, gws, lws, 17, args, info));
    iris_check(task.flush_out(d_found));
    iris_check(task.submit(iris_gpu, nullptr, true));
    auto end = std::chrono::steady_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // copy d_found from device to h_found in host


#ifdef DEBUG
    printf("Kernel execution:\n"
        "\tEffective passwords: %d\n"
        "\tPasswords Range:\n"
        "\t\t%s\n"
        "\t\t.....\n"
        "\t\t%s\n",
        num_read_pswd,
        (char *)(h_pswd_char),                                          // first password in batch
        (char *)(h_pswd_char + ((num_read_pswd - 1) * PSWD_NUM_CHAR)));  // last password in batch
    std::cout << "--------------------\n";
#endif

    tot_num_read_pswd += num_read_pswd;
    match = check_match();  // uses h_found
    if(match || feof(fp)) {
      break;
    }
  }
  std::cout << "Total kernel time (decrypt_vmk_with_mac): " << total_time * 1e-9f << " s\n\n";

  std::cout << "================================================\n"
            << "Bitcracker attack completed\n"
            << "Total passwords evaluated: " << tot_num_read_pswd << std::endl;
  if (match == 1) std::cout << "Password found: " << printed_pswd << std::endl;
  else std::cout << "Password not found!\n";
  std::cout << "================================================\n";

  // close file
  if (fp != stdin)
    fclose(fp);

  if (getenv("BITCRACKER_VERIFY")) printf("FOUND_RAW %d\n", *h_found);
}
