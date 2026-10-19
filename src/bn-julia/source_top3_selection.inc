      // store the top HIGHEST highest orders
      if (c < HIGHEST) {
        tmp = 1;
        for (j = 0; j < c; j++) {
          if (maxScore[j] == preScore) {
            tmp = 0;
          }
        }
        if (tmp != 0) {
          maxScore[c] = preScore;
          for (a = 0; a < NODE_N; a++) {
            for (b = 0; b < NODE_N; b++) {
              bestGraph[c][a][b] = preGraph[a][b];
            }
          }
          c++;
        }

      } else if (c == HIGHEST) {
        sortGraph();
        c++;
      } else {

        tmp = 1;
        for (j = 0; j < HIGHEST; j++) {
          if (maxScore[j] == preScore) {
            tmp = 0;
            break;
          }
        }
        if (tmp != 0 && preScore > maxScore[HIGHEST - 1]) {
          maxScore[HIGHEST - 1] = preScore;
          for (a = 0; a < NODE_N; a++) {
            for (b = 0; b < NODE_N; b++) {
              bestGraph[HIGHEST - 1][a][b] = preGraph[a][b];
            }
          }
          b = HIGHEST - 1;
          for (a = HIGHEST - 2; a >= 0; a--) {
            if (maxScore[b] > maxScore[a]) {
              swap(a, b);
              tmpd = maxScore[a];
              maxScore[a] = maxScore[b];
              maxScore[b] = tmpd;
              b = a;
            }
          }
        }
      }
