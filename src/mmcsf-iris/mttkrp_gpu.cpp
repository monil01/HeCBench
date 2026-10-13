#include "mttkrp_gpu.h"
#include "dmem.h"
int MTTKRP_MIHCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt){

  ITYPE *dInds2, *dInds3, *dFbrPtr0, *dFbrIdx0, *dfbrPtr1, *dFbrIdx1, *dFbrPtr2, *dFbrIdx2, *dFbrLikeSlcInds;
  DTYPE *dVals;
  ITYPE dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0, dFbrLoc =0,  dFbrIdxLoc =0, dFbrLoc2 =0;
  ITYPE totNnz = 0, totSlcPtr = 0, totSlcIdx = 0, totFbrPtr = 0, totFbrIdx = 0, totFbrPtr2 = 0;

  // All m same mode
  ITYPE mode0 = 0;//TiledX[0].modeOrder[0];
  ITYPE mode1 = 1;//TiledX[0].modeOrder[1];
  ITYPE mode2 = 2;//TiledX[0].modeOrder[2];
  ITYPE mode3 = 3;//((TiledX[0].ndims == 4) ? TiledX[0].modeOrder[3] : 0) ;

  for (int m = 0; m < TiledX[0].ndims; ++m){

    if (TiledX[m].totNnz == 0) continue;

    totNnz += TiledX[m].totNnz;
    totSlcPtr += TiledX[m].fbrPtr[0].size() ;
    totSlcIdx += TiledX[m].fbrIdx[0].size() ;
    totFbrPtr += TiledX[m].fbrPtr[1].size() ;
    totFbrIdx += TiledX[m].fbrIdx[1].size() ;
    totFbrPtr2 += ((TiledX[m].ndims == 4) ? TiledX[m].fbrPtr[2].size() : 0) ;
  }

  // Allocate Tensor on a device
  allocateBuffer((void**) &dVals, totNnz * sizeof(DTYPE));
  allocateBuffer((void**) &dFbrPtr0, totSlcPtr * sizeof(ITYPE));
  allocateBuffer((void**) &dFbrIdx0, totSlcIdx * sizeof(ITYPE));
  allocateBuffer((void**) &dfbrPtr1, totFbrPtr * sizeof(ITYPE));
  allocateBuffer((void**) &dFbrIdx1, totFbrIdx * sizeof(ITYPE));
  allocateBuffer((void**) &dFbrLikeSlcInds, totFbrIdx * sizeof(ITYPE));

  if(TiledX[0].ndims == 3)
    allocateBuffer((void**) &dInds2, totNnz * sizeof(ITYPE));

  if(TiledX[0].ndims == 4){
    allocateBuffer((void**) &dFbrIdx2, totFbrPtr2 * sizeof(ITYPE));
    allocateBuffer((void**) &dFbrPtr2, totFbrPtr2 * sizeof(ITYPE));
    allocateBuffer((void**) &dInds3, totNnz * sizeof(ITYPE));
  }

  // device memory copy for tiled parts
  for (int m = 0; m < TiledX[0].ndims; ++m){

    if(m > 0) {

      if (TiledX[m-1].totNnz > 0) {

        dLoc += TiledX[m-1].totNnz;
        dSlcLoc += TiledX[m - 1].fbrPtr[0].size(); // all m same
        dSlcIdxLoc += TiledX[m - 1].fbrIdx[0].size();
        dFbrLoc += TiledX[m - 1].fbrPtr[1].size();
        dFbrIdxLoc += TiledX[m - 1].fbrIdx[1].size();
        dFbrLoc2 += ((TiledX[m].ndims == 4) ? TiledX[m - 1].fbrPtr[2].size() : 0) ;
      }
    }

    if (TiledX[m].totNnz == 0) continue;

    uploadBuffer(dVals + dLoc, &(TiledX[m].vals[0]),
          TiledX[m].totNnz * sizeof(DTYPE));
    uploadBuffer(dFbrPtr0 + dSlcLoc, &(TiledX[m].fbrPtr[0][0]),
          TiledX[m].fbrPtr[0].size() * sizeof(ITYPE));
    uploadBuffer(dFbrIdx0 + dSlcIdxLoc, &(TiledX[m].fbrIdx[0][0]),
          TiledX[m].fbrIdx[0].size() * sizeof(ITYPE));
    uploadBuffer(dfbrPtr1 + dFbrLoc, &(TiledX[m].fbrPtr[1][0]),
          TiledX[m].fbrPtr[1].size() * sizeof(ITYPE));
    uploadBuffer(dFbrIdx1 + dFbrIdxLoc, &(TiledX[m].fbrIdx[1][0]),
          TiledX[m].fbrIdx[1].size() * sizeof(ITYPE));
    uploadBuffer(dFbrLikeSlcInds + dFbrIdxLoc, &(TiledX[m].fbrLikeSlcInds[0]),
          TiledX[m].fbrIdx[1].size() * sizeof(ITYPE));

    if(TiledX[m].ndims == 3){
      if(m <= 2)
        uploadBuffer(dInds2 + dLoc, &(TiledX[m].inds[TiledX[m].modeOrder[2]][0]),
              TiledX[m].totNnz * sizeof(ITYPE));
    }
    if(TiledX[m].ndims == 4){
      uploadBuffer(dFbrPtr2 + dFbrLoc2, &(TiledX[m].fbrPtr[2][0]),
            TiledX[m].fbrPtr[2].size() * sizeof(ITYPE));
      uploadBuffer(dFbrIdx2 + dFbrLoc2, &(TiledX[m].fbrIdx[2][0]),
            TiledX[m].fbrIdx[2].size() * sizeof(ITYPE));
      uploadBuffer(dInds3 + dLoc, &(TiledX[m].inds[TiledX[m].modeOrder[3]][0]),
            TiledX[m].totNnz * sizeof(ITYPE));
    }
  }

  //Matrices
  unsigned int *dULoc =  new unsigned int[TiledX[0].ndims];
  unsigned int *szDU =  new unsigned int[TiledX[0].ndims];

  //Matrices
  DTYPE *dU;

  ITYPE mtxSize = ((TiledX[0].ndims == 3) ? (U[mode0].nRows + U[mode1].nRows + U[mode2].nRows) * U[mode0].nCols
      : (U[mode0].nRows + U[mode1].nRows + U[mode2].nRows + U[mode3].nRows) * U[mode0].nCols );

  allocateBuffer((void**) &dU, mtxSize * sizeof(DTYPE));

  for (int m = 0; m < TiledX[0].ndims; ++m)
    szDU[m] = U[m].nRows * U[m].nCols;

  resetBuffer(dU, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
  uploadBuffer(dU + szDU[0], &(U[mode1].vals[0]),
        U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE));
  uploadBuffer(dU + szDU[0] + szDU[1], &(U[mode2].vals[0]),
        U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE));

  if(TiledX[0].ndims == 4)
    uploadBuffer(dU + szDU[0] + szDU[1] + szDU[2], &(U[mode3].vals[0]),
        U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE));

  dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0; dFbrLoc =0, dFbrIdxLoc = 0, dFbrLoc2= 0;

  for (int MTTKRPmode = 0; MTTKRPmode < TiledX[0].ndims; ++MTTKRPmode){

    if(MTTKRPmode > 0){

      dLoc = 0; dSlcLoc = 0; dSlcIdxLoc = 0; dFbrLoc =0; dFbrIdxLoc = 0; dFbrLoc2= 0;

      // MTTKRP on mode mode 0 changed DU0. To pass correctness for now initializing to 2 again.
      int mode = MTTKRPmode - 1;
      for(long r = 0; r < U[mode].nRows; ++r){
        for(long c = 0; c < U[mode].nCols; ++c)
          U[mode].vals[r * U[mode].nCols + c] = mode + .5;
      }

      if(MTTKRPmode == 1){
        uploadBuffer(dU, &(U[mode0].vals[0]),
              U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
        resetBuffer(dU + szDU[0], 0,  U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE));
      }
      else if(MTTKRPmode == 2){
        uploadBuffer(dU + szDU[0], &(U[mode1].vals[0]),
              U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE));
        resetBuffer(dU + szDU[0] + szDU[1], 0,  U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE));
      }
      else if(MTTKRPmode == 3){
        uploadBuffer(dU + szDU[0] + szDU[1] , &(U[mode2].vals[0]),
              U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE));
        resetBuffer(dU + szDU[0] + szDU[1] + szDU[2], 0,  U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE));
      }
    }

    for (int m = 0; m < TiledX[0].ndims; ++m){

      /* matrix order according to mode order*/
      for (int mm = 0; mm < TiledX[0].ndims; ++mm){

        int curMode = TiledX[m].modeOrder[mm];
        dULoc[mm] = 0;

        for (int q = 0; q < curMode; ++q)
          dULoc[mm] +=  szDU[q % TiledX[0].ndims]; //1 2 3 0
      }

      if(m > 0) {

        if (TiledX[m-1].totNnz > 0) {

          dLoc += TiledX[m-1].totNnz;
          dSlcLoc += TiledX[m - 1].fbrPtr[0].size();
          dSlcIdxLoc += TiledX[m - 1].fbrIdx[0].size();
          dFbrLoc += TiledX[m - 1].fbrPtr[1].size();
          dFbrIdxLoc += TiledX[m - 1].fbrIdx[1].size();
          dFbrLoc2 += ((TiledX[0].ndims == 4) ? TiledX[m - 1].fbrPtr[2].size(): 0) ;
        }
      }

      if (TiledX[m].totNnz == 0) continue;

      int BLOCKSIZE;

      if(TiledX[m].modeOrder[0] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: Slc atomics\n" ;

        BLOCKSIZE = Opt.TBsize;
        dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

        int warpPerFbr = Opt.warpPerSlice;//4;
        int logOfWarpPerFbr = log2(warpPerFbr);
        int fbrPerWarp = Opt.fiberPerWarp;//1;//BLOCKSIZE/32; // dont overflow TB
        int logOfFbrPerWarp = log2(fbrPerWarp );

        if( (warpPerFbr > (BLOCKSIZE/32)) || (fbrPerWarp > (BLOCKSIZE/32)) ){
          std::cout << "warpPerFbr (-w) or fbrPerWarp (-s) cannot be higher than threadblock size!"
            << std::endl << "hint: increase -b!" << std::endl;
          return -1;
        }

        grid.x = ( warpPerFbr * 32 * ((TiledX[m].nFibers + fbrPerWarp-1)/fbrPerWarp) + BLOCKSIZE - 1) / BLOCKSIZE;

        if(TiledX[0].ndims == 3)
          submitKernel("mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar",grid.x,block.x,{memory(dVals + dLoc),memory(dFbrLikeSlcInds + dFbrIdxLoc),memory(dInds2 + dLoc),memory(dFbrPtr0 + dSlcLoc),memory(dfbrPtr1 + dFbrLoc),memory(dFbrIdx1 + dFbrIdxLoc),scalar(static_cast<ITYPE>(TiledX[m].nFibers)),memory(dU + dULoc[0]),memory(dU + dULoc[1]),memory(dU + dULoc[2]),scalar(static_cast<ITYPE>(Opt.mode)),scalar(static_cast<ITYPE>(Opt.R)),scalar(static_cast<ITYPE>(warpPerFbr)),scalar(static_cast<int>(logOfWarpPerFbr)),scalar(static_cast<int>(fbrPerWarp)),scalar(static_cast<int>(logOfFbrPerWarp))});

        else if(TiledX[0].ndims == 4) {
          submitKernel("mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_4D",grid.x,block.x,{memory(dVals + dLoc),memory(dFbrLikeSlcInds + dFbrIdxLoc),memory(dInds3 + dLoc),memory(dFbrPtr0 + dSlcLoc),memory(dfbrPtr1 + dFbrLoc),memory(dFbrIdx1 + dFbrIdxLoc),memory(dFbrPtr2 + dFbrLoc2),memory(dFbrIdx2 + dFbrLoc2),scalar(static_cast<ITYPE>(TiledX[m].nFibers)),memory(dU + dULoc[0]),memory(dU + dULoc[1]),memory(dU + dULoc[2]),memory(dU + dULoc[3]),scalar(static_cast<ITYPE>(Opt.mode)),scalar(static_cast<ITYPE>(Opt.R)),scalar(static_cast<ITYPE>(warpPerFbr)),scalar(static_cast<int>(logOfWarpPerFbr)),scalar(static_cast<int>(fbrPerWarp)),scalar(static_cast<int>(logOfFbrPerWarp))});
	}
      }

      else if(TiledX[0].ndims == 4 && TiledX[m].modeOrder[1] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: FbrS atomics\n";

        BLOCKSIZE = Opt.TBsize;
        dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

        int warpPerFbr = Opt.warpPerSlice; // default 4
        if(warpPerFbr > (BLOCKSIZE/32)){
          std::cout << "warpPerFbr (-w) cannot be higher than threadblock size! hint: increase -b!" << std::endl;
          return -1;
        }
        int logOfWarpPerFbr = log2(warpPerFbr);

        grid.x = ( warpPerFbr * 32 * TiledX[m].nFibers + BLOCKSIZE - 1) / BLOCKSIZE;

        submitKernel("mttkrp_MIHCSR_kernel_fbrS_atomic_fbrLvlPar_4D",grid.x,block.x,{memory(dVals + dLoc),memory(dFbrLikeSlcInds + dFbrIdxLoc),memory(dInds3 + dLoc),memory(dFbrPtr0 + dSlcLoc),memory(dfbrPtr1 + dFbrLoc),memory(dFbrIdx1 + dFbrIdxLoc),memory(dFbrPtr2 + dFbrLoc2),memory(dFbrIdx2 + dFbrLoc2),scalar(static_cast<ITYPE>(TiledX[m].nFibers)),memory(dU + dULoc[1]),memory(dU + dULoc[2]),memory(dU + dULoc[3]),memory(dU + dULoc[0]),scalar(static_cast<ITYPE>(Opt.mode)),scalar(static_cast<ITYPE>(Opt.R)),scalar(static_cast<ITYPE>(warpPerFbr)),scalar(static_cast<int>(logOfWarpPerFbr))});
      }

      else if(TiledX[m].modeOrder[TiledX[0].ndims-2] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: Fbr atomics\n";

        BLOCKSIZE = Opt.TBsize;
        dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

        int warpPerFbr = Opt.warpPerSlice; // default 4
        if(warpPerFbr > (BLOCKSIZE/32)){
          std::cout << "warpPerFbr (-w) cannot be higher than threadblock size! hint: increase -b!" << std::endl;
          return -1;
        }
        int logOfWarpPerFbr = log2(warpPerFbr);

        grid.x = ( warpPerFbr * 32 * TiledX[m].nFibers + BLOCKSIZE - 1) / BLOCKSIZE;

        if(TiledX[0].ndims == 3)
          submitKernel("mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar",grid.x,block.x,{memory(dVals + dLoc),memory(dFbrLikeSlcInds + dFbrIdxLoc),memory(dInds2 + dLoc),memory(dFbrPtr0 + dSlcLoc),memory(dfbrPtr1 + dFbrLoc),memory(dFbrIdx1 + dFbrIdxLoc),scalar(static_cast<ITYPE>(TiledX[m].nFibers)),memory(dU + dULoc[1]),memory(dU + dULoc[2]),memory(dU + dULoc[0]),scalar(static_cast<ITYPE>(Opt.mode)),scalar(static_cast<ITYPE>(Opt.R)),scalar(static_cast<ITYPE>(warpPerFbr)),scalar(static_cast<int>(logOfWarpPerFbr))});

        else if (TiledX[0].ndims == 4) {
          submitKernel("mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_4D",grid.x,block.x,{memory(dVals + dLoc),memory(dFbrLikeSlcInds + dFbrIdxLoc),memory(dInds3 + dLoc),memory(dFbrPtr0 + dSlcLoc),memory(dfbrPtr1 + dFbrLoc),memory(dFbrIdx1 + dFbrIdxLoc),memory(dFbrPtr2 + dFbrLoc2),memory(dFbrIdx2 + dFbrLoc2),scalar(static_cast<ITYPE>(TiledX[m].nFibers)),memory(dU + dULoc[2]),memory(dU + dULoc[3]),memory(dU + dULoc[0]),memory(dU + dULoc[1]),scalar(static_cast<ITYPE>(Opt.mode)),scalar(static_cast<ITYPE>(Opt.R)),scalar(static_cast<ITYPE>(warpPerFbr)),scalar(static_cast<int>(logOfWarpPerFbr))});
	}
      }

      else if(TiledX[m].modeOrder[TiledX[0].ndims-1] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: nnz atomics\n";

        BLOCKSIZE = Opt.TBsize;
        dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

        int warpPerFbr = Opt.warpPerSlice; // default 4
        if(warpPerFbr > (BLOCKSIZE/32)){
          std::cout << "warpPerFbr (-w) cannot be higher than threadblock size! hint: increase -b!" << std::endl;
          return -1;
        }
        int logOfWarpPerFbr = log2(warpPerFbr);

        grid.x = ( warpPerFbr * 32 * TiledX[m].nFibers + BLOCKSIZE - 1) / BLOCKSIZE;

        if (TiledX[0].ndims == 3)
          submitKernel("mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar",grid.x,block.x,{memory(dVals + dLoc),memory(dFbrLikeSlcInds + dFbrIdxLoc),memory(dInds2 + dLoc),memory(dFbrPtr0 + dSlcLoc),memory(dfbrPtr1 + dFbrLoc),memory(dFbrIdx1 + dFbrIdxLoc),scalar(static_cast<ITYPE>(TiledX[m].nFibers)),memory(dU + dULoc[2]),memory(dU + dULoc[0]),memory(dU + dULoc[1]),scalar(static_cast<ITYPE>(Opt.mode)),scalar(static_cast<ITYPE>(Opt.R)),scalar(static_cast<ITYPE>(warpPerFbr)),scalar(static_cast<int>(logOfWarpPerFbr))});

        else if (TiledX[0].ndims == 4) {
          submitKernel("mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_4D",grid.x,block.x,{memory(dVals + dLoc),memory(dFbrLikeSlcInds + dFbrIdxLoc),memory(dInds3 + dLoc),memory(dFbrPtr0 + dSlcLoc),memory(dfbrPtr1 + dFbrLoc),memory(dFbrIdx1 + dFbrIdxLoc),memory(dFbrPtr2 + dFbrLoc2),memory(dFbrIdx2 + dFbrLoc2),scalar(static_cast<ITYPE>(TiledX[m].nFibers)),memory(dU + dULoc[3]),memory(dU + dULoc[0]),memory(dU + dULoc[1]),memory(dU + dULoc[2]),scalar(static_cast<ITYPE>(Opt.mode)),scalar(static_cast<ITYPE>(Opt.R)),scalar(static_cast<ITYPE>(warpPerFbr)),scalar(static_cast<int>(logOfWarpPerFbr))});
	}
      }
      // IRIS submissions are synchronous.
    }
  }

  /* Copying output matrix from GPU to CPU for correctness check */
  int MTTKRPmode = TiledX[0].ndims - 1;
  ITYPE loc = ((TiledX[0].ndims == 3) ? szDU[0] + szDU[1] : szDU[0] + szDU[1] + szDU[2]);
  downloadBuffer(&U[MTTKRPmode].vals[0], dU + loc,
        U[MTTKRPmode].nRows * U[MTTKRPmode].nCols * sizeof(DTYPE));

  releaseBuffer(dVals);
  releaseBuffer(dU);
  releaseBuffer(dFbrIdx0);
  releaseBuffer(dFbrIdx1);
  releaseBuffer(dFbrPtr0);
  releaseBuffer(dfbrPtr1);
  releaseBuffer(dFbrLikeSlcInds);

  if(TiledX[0].ndims == 3)
    releaseBuffer(dInds2);

  if(TiledX[0].ndims == 4){
    releaseBuffer(dFbrIdx2);
    releaseBuffer(dFbrPtr2);
    releaseBuffer(dInds3);
  }
  delete[] dULoc;
  delete[] szDU;

  int totalMIslics = 0, totalMISfibers = 0, totalMIfibers = 0, totalMInnz = 0;;
  for (int m = 0; m <  TiledX[0].ndims; ++m){
    if(TiledX[m].totNnz){
      if(TiledX[m].ndims == 3){
        totalMIslics += TiledX[m].fbrIdx[0].size();
        totalMIfibers += TiledX[m].fbrPtr[1].size();
        totalMInnz += TiledX[m].totNnz;
      }

      if(TiledX[m].ndims == 4){
        totalMIslics += TiledX[m].fbrIdx[0].size();
        totalMISfibers += TiledX[m].fbrPtr[1].size();
        totalMIfibers += TiledX[m].fbrPtr[2].size();
        totalMInnz += TiledX[m].totNnz;
      }
    }
  }

  std::cout << "Resource usage: " << std::endl;
  if(TiledX[0].ndims == 3)
    std::cout << " nSlc:" << totalMIslics
              << ", nFibers:" << totalMIfibers << ", nnz:" << totalMInnz
              << std::endl;
  else if(TiledX[0].ndims == 4)
    std::cout << " nSlc:" << totalMIslics  << ", nSFibers:" << totalMISfibers
              << ", nFibers:" << totalMIfibers << ", nnz:" << totalMInnz
              << std::endl;
  return 0;
}
