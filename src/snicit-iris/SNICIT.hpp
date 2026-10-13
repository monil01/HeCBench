#include <chrono>
#include <fstream>
#include <string>
#include "dmem.hpp"



namespace SNICIT_BEY{

class SNICIT{

  private:
    std::vector<SMemory<float>> _dev_Y_hidden; // 2 buffers
    std::vector<SMemory<int>> _dev_hidden_roffw; // hidden layer W row offset
    std::vector<SMemory<int>> _dev_hidden_colsw; // hidden layer W cols
    std::vector<SMemory<float>> _dev_hidden_valsw; // hidden layer W vals
    std::vector<SMemory<float>> _dev_hidden_bias; // hidden layer W vals


    float *Y_input; // on cpu
    SMemory<float> _dev_Y_output_whole; // on gpu

    SMemory<float> _dev_Y_input;
    SMemory<float> _dev_Y_output;
    SMemory<float> _dev_input_weight;
    SMemory<float> _dev_input_bias;
    SMemory<float> _dev_output_weight;
    SMemory<float> _dev_output_bias;
    SMemory<int> _dev_result_label;

    std::string weight_path, bias_path;
    int num_hidden_neurons, num_layers;
    int input_size, num_classes, num_input, batch_size;
    int nnz;
    float density;
    bool is_cifar;

    // SNICIT augmentation data structures
    int threshold;
    SMemory<int> y_star_row;
    int seed_size;
    SMemory<int> centroid_LUT;
    SMemory<bool> ne_record;
    SMemory<int> rowsY;

    void _infer();

    void _preprocess(const std::string& input_path);

    void _weight_bias_alloc_read();

    void _input_alloc_read(const std::string& input_path);

    void _result_alloc_read(const std::string& input_path);

  public:
    SNICIT(
      const std::string& _weight_path,
      const std::string& _bias_path,
      const int _num_hidden_neurons,
      const int _num_layers,
      const float _density,
      const int _seed_size,
      const int _threshold,
      const int _batch_size,
      const int _num_input,
      const bool _is_cifar
    );

    ~SNICIT();

    void infer(const std::string& input_path);

};

SNICIT::SNICIT(
    const std::string& _weight_path,
    const std::string& _bias_path,
    const int _num_hidden_neurons,
    const int _num_layers,
    const float _density,
    const int _seed_size,
    const int _threshold,
    const int _batch_size,
    const int _num_input,
    const bool _is_cifar
) : weight_path(_weight_path),
    bias_path(_bias_path),
    num_hidden_neurons(_num_hidden_neurons),
    num_layers(_num_layers),
    num_classes(10),
    num_input(_num_input),
    batch_size(_batch_size),
    nnz(std::round(_num_hidden_neurons*_num_hidden_neurons*_density)),
    density(_density),
    is_cifar(_is_cifar),
    threshold(_threshold),
    seed_size(_seed_size)
 {
  std::cout<<"Constructing SNICIT method......\n";
  input_size = is_cifar ? _num_hidden_neurons : 784;
}

SNICIT::~SNICIT() {
  for(auto& each_Y : _dev_Y_hidden) {
    each_Y.reset();
  }
  for(auto& each_dev_hidden_roffw : _dev_hidden_roffw) {
    each_dev_hidden_roffw.reset();
  }
  for(auto& each_dev_hidden_colsw : _dev_hidden_colsw) {
    each_dev_hidden_colsw.reset();
  }
  for(auto& each_dev_hidden_valsw : _dev_hidden_valsw) {
    each_dev_hidden_valsw.reset();
  }
  for(auto& each_dev_hidden_bias : _dev_hidden_bias) {
    each_dev_hidden_bias.reset();
  }
  if (!is_cifar) {
    _dev_Y_input.reset();
    _dev_input_weight.reset();
    _dev_input_bias.reset();
  }
  _dev_output_weight.reset();
  _dev_output_bias.reset();

  _dev_Y_output.reset();
  _dev_Y_output_whole.reset();
  _dev_result_label.reset();
  y_star_row.reset();
  centroid_LUT.reset();
  ne_record.reset();
  rowsY.reset();

  delete [] Y_input;
}

void SNICIT::infer(const std::string& input_path) {
  _preprocess(input_path);

  _infer();

}

void SNICIT::_preprocess(const std::string& input_path) {
  std::cout<<"preprocessing......\n";
  auto _tic = std::chrono::steady_clock::now();

  _weight_bias_alloc_read();

  _input_alloc_read(input_path);

  _result_alloc_read(input_path);

  y_star_row=std::make_shared<SBuffer<int>>(seed_size * sizeof(int));

  centroid_LUT=std::make_shared<SBuffer<int>>(batch_size * sizeof(int));
  ne_record=std::make_shared<SBuffer<bool>>(batch_size * sizeof(bool));
  rowsY=std::make_shared<SBuffer<int>>(batch_size * sizeof(int));

  auto _toc = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(_toc - _tic).count();
  std::cout<<"finished preprocessing in "<<duration<< "ms"<<std::endl;
}


void SNICIT::_weight_bias_alloc_read() {
  std::string line;
  std::ifstream MyReadFile;
  int ptr = 0;
  int file_offset;
  if (!is_cifar) {
    file_offset = 2;
    // allocate input layer's weight and bias
    _dev_input_weight=std::make_shared<SBuffer<float>>(input_size * num_hidden_neurons * sizeof(float));
    _dev_input_bias=std::make_shared<SBuffer<float>>(num_hidden_neurons * sizeof(float));
    // read input layer's weight and bias
    float *input_weight;
    float *input_bias;
    input_weight = new float[input_size * num_hidden_neurons];
    input_bias = new float[num_hidden_neurons];

    MyReadFile = std::ifstream(weight_path+"l1-dense.tsv");
    if (MyReadFile.is_open()) {
      while(std::getline(MyReadFile, line)){
        input_weight[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open weight file " << weight_path+"l1-dense.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    MyReadFile = std::ifstream(bias_path+"l1-dense.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;
      while(std::getline(MyReadFile, line)){
        input_bias[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open bias file " << weight_path+"l1-dense.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    // copy input layer's weight and bias
    _dev_input_weight->update(input_weight,input_size * num_hidden_neurons * sizeof(float));
    _dev_input_bias->update(input_bias,num_hidden_neurons * sizeof(float));
    delete [] input_weight;
    delete [] input_bias;
  }
  else {file_offset = 1;}
  for(int hidden_layer = 0; hidden_layer < num_layers; hidden_layer++) {
    int *hidden_roffw = new int[num_hidden_neurons+1](); // value initialization
    int *hidden_colsw = new int[nnz];
    float *hidden_valsw = new float[nnz];
    float *hidden_bias = new float[num_hidden_neurons];

    SMemory<int> dev_cur_roffw;
    SMemory<int> dev_cur_colsw;
    SMemory<float> dev_cur_valsw;
    SMemory<float> dev_cur_bias;

    // allocate hidden layer's weight and bias
    dev_cur_roffw=std::make_shared<SBuffer<int>>((num_hidden_neurons+1) * sizeof(float));
    dev_cur_colsw=std::make_shared<SBuffer<int>>(nnz * sizeof(float));
    dev_cur_valsw=std::make_shared<SBuffer<float>>(nnz * sizeof(float));

    // read hidden layer

    MyReadFile = std::ifstream(weight_path+"l"+std::to_string(hidden_layer+file_offset)+"-sparse.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;

      std::vector<std::string> tokens;
      while(std::getline(MyReadFile, line)){

        std::stringstream lineStream(line);
        std::string token;
        tokens.clear();
        while(std::getline(lineStream, token, '\t')) {
          tokens.push_back(std::move(token));
        }

        hidden_roffw[std::stoi(tokens[0])+1]++;
        hidden_colsw[ptr] = std::stoi(tokens[1]);
        hidden_valsw[ptr] = std::stof(tokens[2]);
        ptr++;
      }
      for (int i = 0; i < num_hidden_neurons; i++)
      {
          hidden_roffw[i + 1] += hidden_roffw[i];
      }
    }
    else {
      std::cout << "ERROR: open weight file " << weight_path+"l"+
        std::to_string(hidden_layer+file_offset)+"-sparse.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    // copy hidden layer
    dev_cur_roffw->update(hidden_roffw,(num_hidden_neurons+1) * sizeof(int));
    dev_cur_colsw->update(hidden_colsw,nnz * sizeof(int));
    dev_cur_valsw->update(hidden_valsw,nnz * sizeof(float));
    _dev_hidden_roffw.emplace_back(dev_cur_roffw);
    _dev_hidden_colsw.emplace_back(dev_cur_colsw);
    _dev_hidden_valsw.emplace_back(dev_cur_valsw);

    dev_cur_bias=std::make_shared<SBuffer<float>>(num_hidden_neurons * sizeof(float));


    MyReadFile = std::ifstream(bias_path+"l"+std::to_string(hidden_layer+file_offset)+"-sparse.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;
      while(std::getline(MyReadFile, line)){
        hidden_bias[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open bias file " << bias_path+"l"+
        std::to_string(hidden_layer+file_offset)+"-sparse.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();

    dev_cur_bias->update(hidden_bias,num_hidden_neurons * sizeof(float));

    _dev_hidden_bias.emplace_back(dev_cur_bias);

    delete [] hidden_roffw;
    delete [] hidden_colsw;
    delete [] hidden_valsw;
    delete [] hidden_bias;
  }

  _dev_output_weight=std::make_shared<SBuffer<float>>(num_hidden_neurons * num_classes * sizeof(float));
  _dev_output_bias=std::make_shared<SBuffer<float>>(num_classes * sizeof(float));
  float *output_weight;
  float *output_bias;
  output_weight = new float[num_hidden_neurons * num_classes];
  output_bias = new float[num_classes];
  MyReadFile = std::ifstream(weight_path+"l"+std::to_string(num_layers+file_offset)+"-dense.tsv");
  ptr = 0;
  while(std::getline(MyReadFile, line)){
    output_weight[ptr++] = std::stof(line);
  }
  MyReadFile.close();
  MyReadFile = std::ifstream(bias_path+"l"+std::to_string(num_layers+file_offset)+"-dense.tsv");
  ptr = 0;
  while(std::getline(MyReadFile, line)){
    output_bias[ptr++] = std::stof(line);
  }
  MyReadFile.close();

  _dev_output_weight->update(output_weight,num_hidden_neurons * num_classes * sizeof(float));
  _dev_output_bias->update(output_bias,num_classes * sizeof(float));

  delete [] output_weight;
  delete [] output_bias;
}

void SNICIT::_input_alloc_read(const std::string& input_path) {

  Y_input = new float[num_input*input_size];
  if (!is_cifar) {
    std::ifstream file(input_path+"t10k-images-idx3-ubyte", std::ios::binary);
    if (file.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      int n_rows=0;
      int n_cols=0;
      file.read((char*)&magic_number,sizeof(magic_number));
      file.read((char*)&number_of_images,sizeof(number_of_images));
      file.read((char*)&n_rows,sizeof(n_rows));
      file.read((char*)&n_cols,sizeof(n_cols));
      for(int i=0;i<10000;++i) // !!!!!!!!!!!!!!!!!! change!!!!!!!!!!!!!
      {
        for(int r=0;r<input_size;++r)
        {
          unsigned char temp=0;
          file.read((char*)&temp,sizeof(temp));
          Y_input[i*input_size+r]= ((float)temp)/255.0;
        }
      }
      _dev_Y_input=std::make_shared<SBuffer<float>>(batch_size*input_size * sizeof(float));
    }
    else {
      std::cout << "ERROR: MNIST input file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }
  else {
    std::ifstream file(input_path+"cifar-input.txt");
    if (file.is_open()) {
      std::string line;
      int ptr = 0;
      while(std::getline(file, line)){
        Y_input[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: CIFAR-10 input file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }

  for(int buff=0; buff<2; buff++) {
    SMemory<float> _dev_buff_Y;
    _dev_buff_Y=std::make_shared<SBuffer<float>>(batch_size*num_hidden_neurons * sizeof(float));
    _dev_Y_hidden.emplace_back(_dev_buff_Y);
  }
  _dev_Y_output_whole=std::make_shared<SBuffer<float>>(num_input*num_classes * sizeof(float));

  _dev_Y_output=std::make_shared<SBuffer<float>>(batch_size*num_classes * sizeof(float));
}

void SNICIT::_result_alloc_read(const std::string& input_path) {
  int *label;
  label = new int[num_input];
  if (!is_cifar) {
    std::ifstream file(input_path+"t10k-labels-idx1-ubyte", std::ios::binary);
    if (file.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      file.read((char*)&magic_number,sizeof(magic_number));
      file.read((char*)&number_of_images,sizeof(number_of_images));
      for(int i = 0; i < 10000; ++i)
      {
        unsigned char temp=0;
        file.read((char*)&temp, sizeof(temp));
        label[i] = (int)temp;
      }
    }
    else {
      std::cout << "ERROR: MNIST result file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }
  else {
    std::ifstream file;
    file.open(input_path+"cifar-label.bin", std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) {
        std::cout << "ERROR: CIFAR-10 result file open failed" << std::endl;
        exit(1);
    }

    auto file_size = file.tellg();
    auto buffer = std::make_unique<char[]>(file_size);

    //Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), file_size);
    file.close();

    for(int i = 0; i < num_input; ++i){
      int l = buffer[i * 3073];
      label[i] = l;
    }
  }

  _dev_result_label=std::make_shared<SBuffer<int>>(num_input * sizeof(int));

  _dev_result_label->update(label,num_input * sizeof(int));

  delete [] label;
}

void SNICIT::_infer() {
  std::cout<<"inferring......\n";
  auto _tic = std::chrono::steady_clock::now();
  double sparse_duration = 0.0;
  double post_duration = 0.0;
  for (int round = 0; round < num_input / batch_size; round++) {
    std::cout<<"[round "<<round<<"] begins: "<<std::endl;
    if (!is_cifar) {
      _dev_Y_input->update(Y_input+round * batch_size * input_size,batch_size*input_size * sizeof(float));
      { void *args[]={&_dev_Y_input->mem, &_dev_input_weight->mem, &_dev_input_bias->mem, &batch_size, &input_size, &num_hidden_neurons, &_dev_Y_hidden[0]->mem};int info[]={iris_r, iris_r, iris_r, sizeof(batch_size), sizeof(input_size), sizeof(num_hidden_neurons), iris_rw};sn_launch("dense_input",batch_size,dim3(num_hidden_neurons, (int)(1024/num_hidden_neurons), 1),7,args,info); }
    }
    else {
      _dev_Y_hidden[0]->update(Y_input+round * batch_size * input_size,batch_size*input_size * sizeof(float));
    }
    auto sparse_tic = std::chrono::steady_clock::now();
    // pre-convergence
    auto pre_tic = std::chrono::steady_clock::now();
    for(int cur_layer = 0; cur_layer < threshold; cur_layer++) { // num_layers-2
      { void *args[]={&_dev_Y_hidden[cur_layer%2]->mem, &_dev_hidden_roffw[cur_layer]->mem, &_dev_hidden_colsw[cur_layer]->mem, &_dev_hidden_valsw[cur_layer]->mem, &_dev_hidden_bias[cur_layer]->mem, &batch_size, &num_hidden_neurons, &num_hidden_neurons, &_dev_Y_hidden[(cur_layer+1)%2]->mem};int info[]={iris_r, iris_r, iris_r, iris_r, iris_r, sizeof(batch_size), sizeof(num_hidden_neurons), sizeof(num_hidden_neurons), iris_rw};sn_launch("sparse_hidden",batch_size,dim3((int)(1024/num_hidden_neurons), num_hidden_neurons, 1),9,args,info); }

      _dev_Y_hidden[cur_layer % 2]->zero(batch_size*num_hidden_neurons*sizeof(float));
    }
    auto pre_toc = std::chrono::steady_clock::now();
    auto pre_duration = std::chrono::duration_cast<std::chrono::microseconds>(pre_toc - pre_tic).count();
    std::cout<<"[**pre conv**] "<< pre_duration/1000.0<< "ms"<<std::endl;
    // y star generation
    auto cluster_tic = std::chrono::steady_clock::now();

    { void *args[]={&_dev_Y_hidden[threshold % 2]->mem, &y_star_row->mem, &batch_size, &num_hidden_neurons, &seed_size};int info[]={iris_r, iris_rw, sizeof(batch_size), sizeof(num_hidden_neurons), sizeof(seed_size)};sn_launch("y_star_gen",1,dim3(1024/seed_size, seed_size, 1),5,args,info);y_star_row->flush(); }


    int y_star_cnt = 0;
    for (int i = 0; i < seed_size; i++) {
      if ((*y_star_row)[i] != -1) {
        (*centroid_LUT)[(*y_star_row)[i]] = -1;
        (*y_star_row)[y_star_cnt++] = (*y_star_row)[i];
        // printf("%drow, label=%d\n", (*y_star_row)[i], _dev_result_label[(*y_star_row)[i]]);
      }
    }
    y_star_row->update_binding();centroid_LUT->update_binding();
    // coarse cluster
    { void *args[]={&_dev_Y_hidden[threshold % 2]->mem, &y_star_row->mem, &ne_record->mem, &y_star_cnt, &centroid_LUT->mem, &num_hidden_neurons};int info[]={iris_rw, iris_r, iris_rw, sizeof(y_star_cnt), iris_rw, sizeof(num_hidden_neurons)};sn_launch("coarse_cluster",batch_size,dim3(1024/y_star_cnt, y_star_cnt, 1),6,args,info);ne_record->flush();centroid_LUT->flush(); }

    int ne_rows = 0;
    for (int i = 0; i < batch_size; i++) {
      if ((*ne_record)[i] != false) {
        (*rowsY)[ne_rows++] = i;
      }
    }
    // std::cout<<"non empty rows in delta Y =  "<< ne_rows <<std::endl;
    rowsY->update_binding();
    auto cluster_toc = std::chrono::steady_clock::now();
    auto cluster_duration = std::chrono::duration_cast<std::chrono::microseconds>(cluster_toc - cluster_tic).count();
    std::cout<<"[**cluster-based conversion**] finished in "<< cluster_duration/1000.0<< "ms"<<std::endl;
    // post convergence
    auto post_tic = std::chrono::steady_clock::now();
    for(int cur_layer = threshold; cur_layer < num_layers; cur_layer++) { // num_layers-2
      auto post_p_tic = std::chrono::steady_clock::now();
      { void *args[]={&rowsY->mem, &_dev_Y_hidden[threshold % 2]->mem, &_dev_hidden_roffw[cur_layer]->mem, &_dev_hidden_colsw[cur_layer]->mem, &_dev_hidden_valsw[cur_layer]->mem, &batch_size, &num_hidden_neurons, &num_hidden_neurons, &_dev_Y_hidden[(1+threshold) % 2]->mem};int info[]={iris_r, iris_r, iris_r, iris_r, iris_r, sizeof(batch_size), sizeof(num_hidden_neurons), sizeof(num_hidden_neurons), iris_rw};sn_launch("sparse_hidden_post",ne_rows,dim3((int)(1024/num_hidden_neurons), num_hidden_neurons, 1),9,args,info); }
      { void *args[]={&rowsY->mem, &centroid_LUT->mem, &_dev_Y_hidden[(1+threshold) % 2]->mem, &_dev_hidden_bias[cur_layer]->mem, &num_hidden_neurons, &ne_record->mem, &_dev_Y_hidden[threshold % 2]->mem};int info[]={iris_r, iris_r, iris_r, iris_r, sizeof(num_hidden_neurons), iris_rw, iris_rw};sn_launch("update_post",ne_rows,dim3(num_hidden_neurons),7,args,info);ne_record->flush(); }

      int new_ne_rows = 0;
      for (int i = 0; i < ne_rows; i++) {
        if ((*ne_record)[(*rowsY)[i]] != false) {
          (*rowsY)[new_ne_rows++] = (*rowsY)[i];
        }
      }
      ne_rows = new_ne_rows;rowsY->update_binding();
      // std::cout<<"non empty rows in delta Y =  "<< ne_rows <<std::endl;
      _dev_Y_hidden[(1+threshold) % 2]->zero(batch_size*num_hidden_neurons*sizeof(float));

      auto post_p_toc = std::chrono::steady_clock::now();
      auto post_p_duration = std::chrono::duration_cast<std::chrono::microseconds>(post_p_toc - post_p_tic).count();
      std::cout<<"[**post convergence**]finished layer "<< cur_layer <<" in "<< post_p_duration/1000.0<< "ms"<<std::endl;
    }
    auto post_toc = std::chrono::steady_clock::now();
    post_duration = std::chrono::duration_cast<std::chrono::microseconds>(post_toc - post_tic).count();
    std::cout<<"[**post convergence**]finished in "<< post_duration/1000.0<< "ms"<<std::endl;
    // recovery
    auto recovery_tic = std::chrono::steady_clock::now();
    { void *args[]={&_dev_Y_hidden[threshold % 2]->mem, &centroid_LUT->mem, &num_hidden_neurons};int info[]={iris_rw, iris_r, sizeof(num_hidden_neurons)};sn_launch("recover",batch_size,dim3(num_hidden_neurons),3,args,info); }
    auto recovery_toc = std::chrono::steady_clock::now();
    auto recovery_duration = std::chrono::duration_cast<std::chrono::microseconds>(recovery_toc - recovery_tic).count();
    std::cout<<"[**recovery**]finished in "<< recovery_duration/1000.0<< "ms"<<std::endl;
    auto sparse_toc = std::chrono::steady_clock::now();
    sparse_duration += std::chrono::duration_cast<std::chrono::microseconds>(sparse_toc - sparse_tic).count();

    // output layer
    { void *args[]={&_dev_Y_hidden[threshold % 2]->mem, &_dev_output_weight->mem, &_dev_output_bias->mem, &batch_size, &num_hidden_neurons, &num_classes, &_dev_Y_output->mem};int info[]={iris_r, iris_r, iris_r, sizeof(batch_size), sizeof(num_hidden_neurons), sizeof(num_classes), iris_rw};sn_launch("dense_output",batch_size,dim3(num_classes, (int)(1024/num_classes), 1),7,args,info); }
    _dev_Y_output->flush();
    memcpy(_dev_Y_output_whole->host.get()+round * batch_size * num_classes,_dev_Y_output->host.get(),batch_size*num_classes * sizeof(float));
    _dev_Y_output_whole->update_binding();
  }
  // std::cout<<"SNICIT runtime: "<< sparse_duration/1000.0<< "ms"<<std::endl;
  SMemory<int> cnt;
  cnt=std::make_shared<SBuffer<int>>(sizeof(int));
  { void *args[]={&_dev_Y_output_whole->mem, &num_classes, &num_input, &_dev_result_label->mem, &cnt->mem};int info[]={iris_rw, sizeof(num_classes), sizeof(num_input), iris_rw, iris_rw};sn_launch("check_acc",1,dim3(1024),5,args,info);cnt->flush(); }


  std::cout<<"SNICIT info: accuracy "<<100*((float)(*cnt)[0]/(float)num_input)<<"%"<<" runtime "<< sparse_duration/1000.0<< "ms"
  <<" avgpost "<<post_duration/(1000*(num_layers-threshold))<< "ms"<<std::endl;
  if(getenv("SNICIT_VERIFY")) {
    _dev_Y_output_whole->flush();
    FILE *file=fopen("scores.bin","wb");if(!file) { perror("scores.bin");exit(1); }
    int header[]={num_input,num_classes,(*cnt)[0]};fwrite(header,sizeof(int),3,file);
    fwrite(_dev_Y_output_whole->host.get(),sizeof(float),size_t(num_input)*num_classes,file);fclose(file);
  }

  auto _toc = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(_toc - _tic).count();
  std::cout<<"[Total] finished inferring in "<<duration/1000.0<< "ms"<<std::endl;

}

}
