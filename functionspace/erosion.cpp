#include "../headerspace/WatershedAlg.h"
using namespace cv;

Mat WatershedAlg::erosion(Mat &image, int **kernel,int rows,int cols) {


    int morph_size = 2;
    Mat element = getStructuringElement(cv::MORPH_RECT, Size(2 * morph_size + 1,2 * morph_size + 1),
    Point(morph_size, morph_size));
    
    Mat dill(image.rows,image.cols,CV_8UC1,Scalar::all(0));
    
    auto*  __restrict dillstart=dill.data;

    dilate(image, dill, element,Point(-1, -1), 1);

    int n = rows;
    int m = cols;
    int drows=dill.rows;
    int dcols=dill.cols;
    image.release();
   vector< vector < int> > shouldBeZeroImage(drows, vector<int>(dcols, 0));
        #pragma acc enter data copyin(dillstart[:drows*dcols],shouldBeZeroImage[:drows][:dcols],kernel[:n][:m])
        #pragma acc parallel loop collapse(2) default(present)
        for(int i = n / 2; i < drows - n / 2; i++) {
           for(int j = m / 2; j < dcols - m / 2; j++) {
                if( (int)dillstart[i*dcols+j]  == ONE) {
                    //bool shouldBeZero = false;
                    int shouldBeZero = 0;
                    for(int crtX = i - n / 2, x = 0; crtX <= i + n / 2; crtX++, x++) {
                         for(int crtY = j - m / 2, y = 0; crtY <= j + m / 2; crtY++, y++) {
                            if((int)dillstart[crtX*dcols+crtY] == ZERO && kernel[x][y] == 1) {
                               // shouldBeZero = true;
                               shouldBeZero=1;
                                break;
                            }
                        }
                    }

                    if(shouldBeZero) {
                       shouldBeZeroImage[i][j]=1;
                    }//if
                }//if
            }//for
         }//for
             #pragma acc parallel loop collapse(2) default(present) 
             for(int i = 0; i < drows; i++) {
               for(int j = 0; j < dcols; j++) {

               if(shouldBeZeroImage[i][j]==1) {


                       dillstart[i*dcols+j] = ZERO;
                 }
              }
           }
           #pragma acc update self(dillstart[:drows*dcols])
           #pragma acc exit data delete (shouldBeZeroImage[:drows][:dcols],kernel[:n][:m],dillstart[:drows*dcols])



    return dill;

    }
