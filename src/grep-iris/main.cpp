/*
 * Regular expression implementation.
 * Supports only ( | ) * + ?.  No escapes.
 * Compiles to NFA and then simulates NFA
 * using Thompson's algorithm.
 *
 * See also http://swtch.com/~rsc/regexp/ and
 * Thompson, Ken.  Regular Expression Search Algorithm,
 * Communications of the ACM 11(6) (June 1968), pp. 419-422.
 * 
 * Copyright (c) 2007 Russ Cox.
 * 
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the
 * Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall
 * be included in all copies or substantial portions of the
 * Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
 * KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS
 * OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <iris/iris.hpp>
#include <cstring>
#include "pnfa.h"
#include "cycleTimer.h"



int checkCmdLine(int argc, char **argv, char **fileName, char **regexFile, int *time) {
  int visualize, simplified, postfix;
  SimpleReBuilder builder;
  State *start;

  parseCmdLine(argc, argv, &visualize, &postfix, time, &simplified, fileName, regexFile);

  // argv index at which regex is present
  int regexIndex = 1 + visualize + postfix + *time + simplified;
  if (fileName != NULL)
    regexIndex += 2;

  if (argc <= regexIndex) {
    usage (argv[0]);
    exit(EXIT_SUCCESS);
  }

  char * regexBuffer = (char*)malloc(strlen(argv[regexIndex])+1);
  strcpy(regexBuffer, argv[regexIndex]);
  simplifyRe(&regexBuffer, &builder);
  free(regexBuffer);

  char *post = re2post(builder.re);
  if(post == NULL){
    fprintf(stderr, "bad regexp %s\n", argv[regexIndex]);
    return 1;
  }

  if (simplified == 1) {
    char * clean_simplified = stringifyRegex(builder.re);
    printf("\nSimplified Regex: %s\n", clean_simplified);
    free(clean_simplified);
    exit(0);
  }

  /* destruct the simpleRe */
  _simpleReBuilder(&builder);

  if (postfix == 1) {
    char * clean_post = stringifyRegex(post);
    printf("\nPostfix buffer: %s\n", clean_post);
    free(clean_post);
    exit(0);
  }

  if (visualize == 1) { 
    start = post2nfa(post);
    visualize_nfa(start);
    exit(0);
  }

  return regexIndex;

}


int main(int argc, char **argv)
{
  iris::Platform platform;if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) return 1;
  int backend=0,type=0;char device_name[256]={};
  iris_device_info(0,iris_backend,&backend,nullptr);iris_device_info(0,iris_type,&type,nullptr);iris_device_info(0,iris_name,device_name,nullptr);
  fprintf(stderr,"Device backend=%d type=%d name=%s\n",backend,type,device_name);
  if(backend!=3 || type!=256 || !strstr(device_name,"AMD")) return 1;
  {
  int timerOn;
  char *fileName = NULL, *regexFile = NULL, **lines = NULL;
  int num_lines;

  SimpleReBuilder builder;
  double startTime, endReadFile, 
         endSetup, endKernel, endTime;

  int regexIndex = checkCmdLine(argc, argv, &fileName, &regexFile, &timerOn);

  // parallel matching
  if (fileName == NULL) {
    printf("Enter a file \n");
    exit(EXIT_SUCCESS);
  }

  //===============================================================================
  // match just a single regex  on a device
  //===============================================================================
  startTime = CycleTimer::currentSeconds();  
  readFile(fileName, &lines, &num_lines);    
  endReadFile = CycleTimer::currentSeconds();

  char * regexBuffer = (char*)malloc(strlen(argv[regexIndex])+1);
  strcpy(regexBuffer, argv[regexIndex]);
  simplifyRe(&regexBuffer, &builder);
  free(regexBuffer);

  int postsize = (strlen(builder.re) + 1) * sizeof (char);

  iris::DMem device_regex(builder.re,postsize);

  u32 * table = (u32 *) malloc(sizeof(u32) * strlen(*lines));
  table[0] = 0;
  num_lines = 0;

  int len = strlen(lines[0]);
  for (int i = 0; i < len; i++) {
    if ((lines[0])[i] == '\n') {
      table[++num_lines] = i+1;
      lines[0][i] = 0;    

    }
  }

  if((lines[0])[len-1] == '\n')/*if at the end file not '\n', then we not forgot last offset */
    --num_lines;

  iris::DMem device_line_table(table,sizeof(u32)*len);
  iris::DMem device_line(*lines,sizeof(char)*(len+1));

  u32 host_regex_table[1]; /*offsets to regexes on host*/

  host_regex_table[0]=0;   /*in case of one regex offset must be 0*/
  iris::DMem device_regex_table(host_regex_table,sizeof(u32));

  unsigned char *host_result=(unsigned char*)malloc(num_lines*sizeof(unsigned char));
  iris::DMem device_result(host_result,num_lines*sizeof(unsigned char));

  State pmatchstate = { Match };  /* matching state */
  static_assert(sizeof(State)==40 && sizeof(u32)==4,"State/u32 ABI");
  iris::DMem device_match_state(&pmatchstate,sizeof(State));

  endSetup = CycleTimer::currentSeconds();

  // measure kernel execution time
  iris::Task task;
  void* args[]={&device_line,&device_line_table,&num_lines,&device_regex,&device_regex_table,&device_result,&device_match_state};
  int flags[]={iris_r,iris_r,sizeof(num_lines),iris_r,iris_r,iris_w,iris_r};
  size_t gws[]={512*160},lws[]={160};
  if(task.kernel("iris_parallelMatch",1,nullptr,gws,lws,7,args,flags)!=IRIS_SUCCESS ||
     task.flush_out(device_result)!=IRIS_SUCCESS || task.submit(0,nullptr,1)!=IRIS_SUCCESS) return 1;

  endKernel = CycleTimer::currentSeconds();


  // print the "grep" results for verification
  if (!timerOn) {  
    for (int i = 0; i < num_lines; i++) {
      if(host_result[i] == 1) 
        PRINT(timerOn, "%s\n", lines[0] + table[i]);
    }
  }


  free(table);
  free(host_result);
  free(*lines);
  free(lines);

  endTime = CycleTimer::currentSeconds();

  if (timerOn) {
    printf("\nReadFile time %.4f \n", (endReadFile - startTime));
    printf("\nDevice setup time %.4f \n", (endSetup - endReadFile));
    printf("\nKernel execution Time %.4f \n", (endKernel - endSetup));
    printf("\nTotal time %.4f \n\n", (endTime - startTime));
  }

  }
  platform.finalize();return EXIT_SUCCESS;
}
