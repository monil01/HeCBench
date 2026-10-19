void sortGraph() {
  float max = -99999999999999.f;
  int maxi, i, j;
  float tmp;

  for (j = 0; j < HIGHEST - 1; j++) {
    max = maxScore[j];
    maxi = j;
    for (i = j + 1; i < HIGHEST; i++) {
      if (maxScore[i] > max) {
        max = maxScore[i];
        maxi = i;
      }
    }

    swap(j, maxi);
    tmp = maxScore[j];
    maxScore[j] = max;
    maxScore[maxi] = tmp;
  }
}

void swap(int a, int b) {
  int i, j;
  bool tmp;

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++) {

      tmp = bestGraph[a][i][j];
      bestGraph[a][i][j] = bestGraph[b][i][j];
      bestGraph[b][i][j] = tmp;
    }
  }
}

void initial() {
  int i, j, tmp, a, b, r;
  bool tmpd;
  tmp = 1;
  for (i = 1; i <= 4; i++) {
    tmp += C(NODE_N - 1, i);
  }
  sizepernode = tmp;
  tmp *= NODE_N;

  localscore = (float*) malloc(tmp * sizeof(float));
  localscore_ref = (float*) malloc(tmp * sizeof(float));

  for (i = 0; i < tmp; i++)
    localscore_ref[i] = localscore[i] = 0;

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++)
      orders[i][j] = 0;
  }
  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < i; j++)
      orders[i][j] = 1;
  }
  r = rand() % 10000;
  for (i = 0; i < r; i++) {
    a = rand() % NODE_N;
    b = rand() % NODE_N;
    for (j = 0; j < NODE_N; j++) {
      tmpd = orders[j][a];
      orders[j][a] = orders[j][b];
      orders[j][b] = tmpd;
    }

    for (j = 0; j < NODE_N; j++) {
      tmpd = orders[a][j];
      orders[a][j] = orders[b][j];
      orders[b][j] = tmpd;
    }
  }

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++) {
      preOrders[i][j] = orders[i][j];
    }
  }
}

// generate ramdom order
int genOrders() {

  int a, b, j;
  bool tmp;
  a = rand() % NODE_N;
  b = rand() % NODE_N;

  for (j = 0; j < NODE_N; j++) {
    tmp = orders[a][j];
    orders[a][j] = orders[b][j];
    orders[b][j] = tmp;
  }
  for (j = 0; j < NODE_N; j++) {
    tmp = orders[j][a];
    orders[j][a] = orders[j][b];
    orders[j][b] = tmp;
  }

  return 1;
}

// decide leave or discard an order
int ConCore() {
  int i, j;
  float tmp;
  tmp = log((rand() % 100000) / 100000.0);
  if (tmp < (score - preScore)) {

    for (i = 0; i < NODE_N; i++) {
      for (j = 0; j < NODE_N; j++) {
        preOrders[i][j] = orders[i][j];
        preGraph[i][j] = graph[i][j];
      }
    }
    preScore = score;

    return 1;
  }

  return 0;
}

void genScore() {
}

void Pre_logGamma() {

  LG = (float*) malloc ((DATA_N + 2) * sizeof(float));

  LG[0] = 0.f; // log(0!), explicit initialized contract
  LG[1] = log(1.0);
  float i;
  for (i = 2; i <= DATA_N + 1; i++) {
    LG[(int)i] = LG[(int)i - 1] + log((float)i);
  }
}

void incr(int *bit, int n) {

  bit[n]++;
  if (bit[n] >= 2) {
    bit[n] = 0;
    incr(bit, n + 1);
  }

  return;
}

void incrS(int *bit, int n) {

  bit[n]++;
  if (bit[n] >= STATE_N) {
    bit[n] = 0;
    incr(bit, n + 1);
  }

  return;
}

bool getState(int parN, int *state, int time) {
  int j = 1;

  j = pow(STATE_N, (float)parN) - 1;

  if (time > j)
    return false;

  if (time >= 1)
    incrS(state, 0);

  return true;
}

int findindex(int *arr, int size) { // reminder: arr[0] has to be 0 && size ==
  // array size-1 && index start from 0
  int i, j, index = 0;

  for (i = 1; i < size; i++) {
    index += C(NODE_N - 1, i);
  }

  for (i = 1; i <= size - 1; i++) {
    for (j = arr[i - 1] + 1; j <= arr[i] - 1; j++) {
      index += C(NODE_N - 1 - j, size - i);
    }
  }

  index += arr[size] - arr[size - 1];

  return index;
}

int C(int n, int a) {
  int i, res = 1, atmp = a;

  for (i = 0; i < atmp; i++) {
    res *= n;
    n--;
  }

  for (i = 0; i < atmp; i++) {
    res /= a;
    a--;
  }

  return res;
}
