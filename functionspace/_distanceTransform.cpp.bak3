#include "../headerspace/WatershedAlg.h"
#include <queue>
#include <cstdlib> 
#include <vector>
#include <string.h>
#define NUMSIZE 8
#define NSIZE 4

using namespace cv;

Array2D<int> &WatershedAlg::distanceTransform(Array2D<int>& matArr, Array2D<int> &markers,int rows,int cols,Array2D<int> &plots,Array2D<bool>& visArr,Array1D &plotx,Array1D &ploty) {

       // queue<int> qx, qy;
        int dx[NUMSIZE]={-1, 1, 0, 0, -1, -1, 1, 1};
        int dy[NUMSIZE]={0, 0, -1, 1, -1,  1, 1, -1};
        int pixelThreshold=55;

int boundcounter=0;
int zerocounter=0;
 

#pragma acc enter data copyin(matArr,visArr,plots,matArr.matImg[:matArr.arows][:matArr.acols],visArr.matImg[:visArr.arows][:visArr.acols],plots.matImg[:plots.arows][:plots.acols],dx[:NUMSIZE],dy[:NUMSIZE])
#pragma acc parallel loop collapse(2)  default(present)

for(int i=0;i<matArr.arows;i++){
   for(int j=0;j<matArr.acols;j++){
     if(matArr(i,j)==ZERO){
  
            // visArr(i,j)=true; 
           //  zerocounter+=1;
	     continue;
            }

        for(int k=0;k<NSIZE;k++){

              
           
                    int nextX = i + dx[k];
                    int nextY = j + dy[k];
              
		    if( nextX < 0 || nextY < 0 || nextX >= matArr.arows || nextY >= matArr.acols ) {
	                if(!visArr(nextX,nextY)) {  
                          if(matArr(nextX,nextY)==ZERO){ //!visArr(arr4D[i][j][0][k],arr4D[i][j][0][k])){  
                             matArr(i,j)=pixelThreshold;
                             visArr(i,j)=true;
		     //  boundcounter+=1;
		             plots(i,j)=1;
                       }
                     //  mat4D[i][j][0][k]=i;
                     //  mat4D[i][j][1][k]=j;
                      // vcounter++;
                 }
	      }
	    }
             // }
	 }
       }


#pragma acc update self(matArr.matImg[:matArr.arows][:matArr.acols],visArr.matImg[:visArr.arows][:visArr.acols],plots.matImg[:plots.arows][:plots.acols])
#pragma acc exit data delete(matArr,visArr,plots,matArr.matImg[:matArr.arows][:matArr.acols],visArr.matImg[:visArr.arows][:visArr.acols],plots.matImg[:plots.arows][:plots.acols])



int i=0;

 int maxVal=0;
        int pcounter=0;
//        #pragma omp parallel for reduction(+:pcounter)
// this is very inefficient on GPU (reduction kernel!) and serial on CPU !!!
        for(int k=0;k<rows;k++){
  //         #pragma omp parallel for
           for(int j=0;j<cols;j++){
             if(plots(k,j)==1){
                  plotx(pcounter)=k;
                  ploty(pcounter)=j;
                 // qx.push(i);
                 // qy.push(j);
                  pcounter++;
             }
            
           }
	}




int qcounter=0;
  


while(plotx(i)!=-1){
            //int crtX = qx.front(); qx.pop();
            //int crtY = qy.front(); qy.pop();

            int crtX=plotx(i);
            int crtY=ploty(i);
             i++;
             qcounter++;
            bool isBigger = true;//check

            for(int h = 0; h < NUMSIZE; h++) {
                int nextX = crtX + dx[h];
                int nextY = crtY + dy[h];

                if( nextX < 0 || nextY < 0 || nextX >= rows || nextY >= cols || matArr(nextX,nextY) == ZERO ) {
                    continue;
                }

                if( matArr(crtX,crtY) <= matArr(nextX,nextY)) {
                    isBigger = false;

                }
                //pick the max local value of some regions

                if( matArr(crtX,crtY) +1< matArr(nextX,nextY)) {
                    visArr(nextX,nextY) = true;
                    matArr(nextX,nextY) =  min((matArr(crtX,crtY)+1), 254);

                    //to get max value for difference between max value image and image
                    if(maxVal<=matArr(nextX,nextY)){
                       maxVal=matArr(nextX,nextY);

                    }
                     //to get max value for difference between max value image and image
 
                    plotx(pcounter)=nextX;
                    ploty(pcounter)=nextY;
                    pcounter++;

                }
                //fout4<<endl;
            }
           //find the max value in local area
            if(isBigger) {
                markers(crtX,crtY)=2;
             }
      }



   pixelThreshold=pixelThreshold-1;

   int pnumThrshold=30;
   int handlingType=0;
   int neighbourType=0;
   removeholesopt(matArr,pnumThrshold,  handlingType, neighbourType, pixelThreshold,rows,cols);

      return matArr;
    }
